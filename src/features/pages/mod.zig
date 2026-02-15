//! Pages Module
//!
//! Dashboard/UI pages with radix-tree URL routing, template rendering,
//! and support for both static and dynamic content.
//!
//! Architecture:
//! - Radix tree for O(path_segments) page matching with params and wildcards
//! - Single-pass {{variable}} template substitution
//! - Per-page flags for auth requirement and cache TTL
//! - RwLock for concurrent page lookups

const std = @import("std");
const core_config = @import("../../core/config/pages.zig");
const sync = @import("../../services/shared/sync.zig");

pub const PagesConfig = core_config.PagesConfig;

pub const PagesError = error{
    FeatureDisabled,
    PageNotFound,
    DuplicatePage,
    TooManyPages,
    InvalidPath,
    TemplateError,
    OutOfMemory,
};

pub const HttpMethod = enum { GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS };

pub const MetadataEntry = struct {
    key: []const u8 = "",
    value: []const u8 = "",
};

pub const TemplateVar = struct {
    key: []const u8 = "",
    value: []const u8 = "",
};

pub const TemplateRef = struct {
    source: []const u8 = "",
    default_vars: [8]TemplateVar = [_]TemplateVar{.{}} ** 8,
    var_count: u8 = 0,
};

pub const PageContent = union(enum) {
    static: []const u8,
    template: TemplateRef,
};

pub const Page = struct {
    path: []const u8 = "/",
    title: []const u8 = "",
    content: PageContent = .{ .static = "" },
    layout: []const u8 = "default",
    method: HttpMethod = .GET,
    require_auth: bool = false,
    cache_ttl_ms: u64 = 0,
    metadata: [4]MetadataEntry = [_]MetadataEntry{.{}} ** 4,
    metadata_count: u8 = 0,
};

pub const PageMatch = struct {
    page: Page,
    params: [8]Param = [_]Param{.{}} ** 8,
    param_count: u8 = 0,

    pub const Param = struct {
        name: []const u8 = "",
        value: []const u8 = "",
    };

    pub fn getParam(self: *const PageMatch, name: []const u8) ?[]const u8 {
        for (self.params[0..self.param_count]) |p| {
            if (std.mem.eql(u8, p.name, name)) return p.value;
        }
        return null;
    }
};

pub const RenderResult = struct {
    title: []const u8 = "",
    body: []u8 = &.{},
    layout: []const u8 = "default",
    body_owned: bool = false,

    pub fn deinit(self: *RenderResult, allocator: std.mem.Allocator) void {
        if (self.body_owned) {
            allocator.free(self.body);
        }
        self.* = undefined;
    }
};

pub const PagesStats = struct {
    total_pages: u32 = 0,
    total_renders: u64 = 0,
    static_pages: u32 = 0,
    template_pages: u32 = 0,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: PagesConfig,

    pub fn init(allocator: std.mem.Allocator, config: PagesConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ── Internal Types ─────────────────────────────────────────────────────

const PageEntry = struct {
    page: Page,
    path_owned: []u8,
    title_owned: []u8,
};

/// Radix tree node for page matching.
const RadixNode = struct {
    segment: []const u8 = "",
    is_param: bool = false,
    param_name: []const u8 = "",
    is_wildcard: bool = false,
    page_idx: ?u32 = null,
    children: std.ArrayListUnmanaged(*RadixNode) = .empty,

    fn deinitRecursive(self: *RadixNode, allocator: std.mem.Allocator) void {
        for (self.children.items) |child| {
            child.deinitRecursive(allocator);
            allocator.destroy(child);
        }
        self.children.deinit(allocator);
    }

    fn findChild(self: *const RadixNode, segment: []const u8) ?*RadixNode {
        for (self.children.items) |child| {
            if (!child.is_param and !child.is_wildcard and
                std.mem.eql(u8, child.segment, segment))
            {
                return child;
            }
        }
        return null;
    }

    fn findParamChild(self: *const RadixNode) ?*RadixNode {
        for (self.children.items) |child| {
            if (child.is_param) return child;
        }
        return null;
    }

    fn findWildcardChild(self: *const RadixNode) ?*RadixNode {
        for (self.children.items) |child| {
            if (child.is_wildcard) return child;
        }
        return null;
    }
};

// ── Module State ───────────────────────────────────────────────────────

var pg_state: ?*PagesState = null;

const PagesState = struct {
    allocator: std.mem.Allocator,
    config: PagesConfig,
    pages: std.ArrayListUnmanaged(PageEntry),
    radix_root: *RadixNode,
    rw_lock: sync.RwLock,

    // Stats
    stat_total_renders: std.atomic.Value(u64),

    fn create(allocator: std.mem.Allocator, config: PagesConfig) !*PagesState {
        const root = try allocator.create(RadixNode);
        root.* = .{};

        const s = try allocator.create(PagesState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .pages = .empty,
            .radix_root = root,
            .rw_lock = sync.RwLock.init(),
            .stat_total_renders = std.atomic.Value(u64).init(0),
        };
        return s;
    }

    fn destroy(self: *PagesState) void {
        const allocator = self.allocator;

        for (self.pages.items) |entry| {
            allocator.free(entry.path_owned);
            allocator.free(entry.title_owned);
        }
        self.pages.deinit(allocator);

        self.radix_root.deinitRecursive(allocator);
        allocator.destroy(self.radix_root);

        allocator.destroy(self);
    }

    fn insertRadixPage(
        self: *PagesState,
        path: []const u8,
        page_idx: u32,
    ) !void {
        var current = self.radix_root;
        var segments = splitPath(path);

        while (segments.next()) |segment| {
            if (segment.len > 0 and segment[0] == '*') {
                const child = try self.allocator.create(RadixNode);
                child.* = .{
                    .is_wildcard = true,
                    .page_idx = page_idx,
                };
                try current.children.append(self.allocator, child);
                return;
            }

            if (segment.len > 2 and segment[0] == '{' and segment[segment.len - 1] == '}') {
                const param_name = segment[1 .. segment.len - 1];
                if (current.findParamChild()) |child| {
                    current = child;
                } else {
                    const child = try self.allocator.create(RadixNode);
                    child.* = .{
                        .is_param = true,
                        .param_name = param_name,
                    };
                    try current.children.append(self.allocator, child);
                    current = child;
                }
            } else {
                if (current.findChild(segment)) |child| {
                    current = child;
                } else {
                    const child = try self.allocator.create(RadixNode);
                    child.* = .{ .segment = segment };
                    try current.children.append(self.allocator, child);
                    current = child;
                }
            }
        }

        current.page_idx = page_idx;
    }
};

fn splitPath(path: []const u8) std.mem.SplitIterator(u8, .scalar) {
    const trimmed = if (path.len > 0 and path[0] == '/') path[1..] else path;
    return std.mem.splitScalar(u8, trimmed, '/');
}

fn matchPathNode(
    node: *const RadixNode,
    segments: *std.mem.SplitIterator(u8, .scalar),
    result: *PageMatch,
) bool {
    const segment = segments.next() orelse {
        if (node.page_idx != null) return true;
        return false;
    };

    // Try exact match first
    if (node.findChild(segment)) |child| {
        var child_segments = segments.*;
        if (matchPathNode(child, &child_segments, result)) {
            segments.* = child_segments;
            return true;
        }
    }

    // Try param match
    if (node.findParamChild()) |child| {
        var child_segments = segments.*;
        if (matchPathNode(child, &child_segments, result)) {
            if (result.param_count < 8) {
                result.params[result.param_count] = .{
                    .name = child.param_name,
                    .value = segment,
                };
                result.param_count += 1;
            }
            segments.* = child_segments;
            return true;
        }
    }

    // Try wildcard match
    if (node.findWildcardChild()) |child| {
        if (child.page_idx != null) return true;
    }

    return false;
}

/// Render a template string by substituting {{key}} placeholders.
fn renderTemplate(
    allocator: std.mem.Allocator,
    template: []const u8,
    vars: []const TemplateVar,
) PagesError![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    var pos: usize = 0;
    while (pos < template.len) {
        // Look for {{
        if (pos + 1 < template.len and
            template[pos] == '{' and template[pos + 1] == '{')
        {
            // Find closing }}
            const key_start = pos + 2;
            var key_end = key_start;
            while (key_end + 1 < template.len) {
                if (template[key_end] == '}' and template[key_end + 1] == '}') {
                    break;
                }
                key_end += 1;
            }

            if (key_end + 1 < template.len and
                template[key_end] == '}' and template[key_end + 1] == '}')
            {
                const key = template[key_start..key_end];
                // Look up the variable
                const value = lookupVar(vars, key) orelse "";
                result.appendSlice(allocator, value) catch return error.OutOfMemory;
                pos = key_end + 2;
            } else {
                // Malformed — output literal
                result.append(allocator, template[pos]) catch return error.OutOfMemory;
                pos += 1;
            }
        } else {
            result.append(allocator, template[pos]) catch return error.OutOfMemory;
            pos += 1;
        }
    }

    return result.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn lookupVar(vars: []const TemplateVar, key: []const u8) ?[]const u8 {
    for (vars) |v| {
        if (std.mem.eql(u8, v.key, key)) return v.value;
    }
    return null;
}

// ── Public API ─────────────────────────────────────────────────────────

pub fn init(allocator: std.mem.Allocator, config: PagesConfig) PagesError!void {
    if (pg_state != null) return;
    pg_state = PagesState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit() void {
    if (pg_state) |s| {
        s.destroy();
        pg_state = null;
    }
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return pg_state != null;
}

pub fn addPage(page: Page) PagesError!void {
    const s = pg_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.pages.items.len >= s.config.max_pages) return error.TooManyPages;
    if (page.path.len == 0) return error.InvalidPath;

    // Check for duplicate path
    for (s.pages.items) |entry| {
        if (std.mem.eql(u8, entry.path_owned, page.path)) {
            return error.DuplicatePage;
        }
    }

    const page_idx: u32 = @intCast(s.pages.items.len);

    const path_owned = s.allocator.dupe(u8, page.path) catch return error.OutOfMemory;
    const title_owned = s.allocator.dupe(u8, page.title) catch {
        s.allocator.free(path_owned);
        return error.OutOfMemory;
    };

    s.pages.append(s.allocator, .{
        .page = .{
            .path = path_owned,
            .title = title_owned,
            .content = page.content,
            .layout = page.layout,
            .method = page.method,
            .require_auth = page.require_auth,
            .cache_ttl_ms = page.cache_ttl_ms,
            .metadata = page.metadata,
            .metadata_count = page.metadata_count,
        },
        .path_owned = path_owned,
        .title_owned = title_owned,
    }) catch {
        s.allocator.free(path_owned);
        s.allocator.free(title_owned);
        return error.OutOfMemory;
    };

    // Insert into radix tree — roll back on failure
    s.insertRadixPage(path_owned, page_idx) catch {
        _ = s.pages.pop();
        s.allocator.free(path_owned);
        s.allocator.free(title_owned);
        return error.OutOfMemory;
    };
}

pub fn removePage(path: []const u8) PagesError!bool {
    const s = pg_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    for (s.pages.items, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.path_owned, path)) {
            s.allocator.free(entry.path_owned);
            s.allocator.free(entry.title_owned);
            _ = s.pages.orderedRemove(i);

            // Rebuild radix tree
            s.radix_root.deinitRecursive(s.allocator);
            s.radix_root.* = .{};
            for (s.pages.items, 0..) |remaining, new_idx| {
                s.insertRadixPage(remaining.path_owned, @intCast(new_idx)) catch |err| {
                    std.log.err("pages: radix rebuild failed after removePage: {t}", .{err});
                    return error.OutOfMemory;
                };
            }
            return true;
        }
    }
    return false;
}

pub fn getPage(path: []const u8) ?Page {
    const s = pg_state orelse return null;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    for (s.pages.items) |entry| {
        if (std.mem.eql(u8, entry.path_owned, path)) {
            return entry.page;
        }
    }
    return null;
}

pub fn matchPage(path: []const u8) PagesError!?PageMatch {
    const s = pg_state orelse return error.FeatureDisabled;
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    var result = PageMatch{ .page = .{} };
    var segments = splitPath(path);

    if (matchPathNode(s.radix_root, &segments, &result)) {
        // Find the matched page via radix tree traversal
        // Re-match to get the page_idx from terminal node
        var segments2 = splitPath(path);
        var current = s.radix_root;
        var found_idx: ?u32 = null;

        while (segments2.next()) |seg| {
            if (current.findChild(seg)) |child| {
                current = child;
            } else if (current.findParamChild()) |child| {
                current = child;
            } else if (current.findWildcardChild()) |child| {
                found_idx = child.page_idx;
                break;
            } else {
                return null;
            }
        }

        if (found_idx == null) found_idx = current.page_idx;

        if (found_idx) |idx| {
            if (idx < s.pages.items.len) {
                result.page = s.pages.items[idx].page;
                return result;
            }
        }
    }

    return null;
}

pub fn renderPage(
    allocator: std.mem.Allocator,
    path: []const u8,
    vars: []const TemplateVar,
) PagesError!RenderResult {
    const s = pg_state orelse return error.FeatureDisabled;

    _ = s.stat_total_renders.fetchAdd(1, .monotonic);

    // Find the page
    const page = getPage(path) orelse return error.PageNotFound;

    switch (page.content) {
        .static => |content| {
            return .{
                .title = page.title,
                .body = @constCast(content),
                .layout = page.layout,
                .body_owned = false,
            };
        },
        .template => |tmpl| {
            // Merge default vars with provided vars (provided override defaults)
            var merged = std.ArrayListUnmanaged(TemplateVar).empty;
            defer merged.deinit(allocator);

            // Add defaults first
            for (tmpl.default_vars[0..tmpl.var_count]) |v| {
                merged.append(allocator, v) catch return error.OutOfMemory;
            }
            // Override with provided vars
            for (vars) |v| {
                var found = false;
                for (merged.items) |*m| {
                    if (std.mem.eql(u8, m.key, v.key)) {
                        m.value = v.value;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    merged.append(allocator, v) catch return error.OutOfMemory;
                }
            }

            const body = try renderTemplate(allocator, tmpl.source, merged.items);
            return .{
                .title = page.title,
                .body = body,
                .layout = page.layout,
                .body_owned = true,
            };
        },
    }
}

pub fn listPages() []const Page {
    const s = pg_state orelse return &.{};
    // Return a simple empty slice — callers should iterate via getPage
    _ = s;
    return &.{};
}

pub fn stats() PagesStats {
    const s = pg_state orelse return .{};

    var static_count: u32 = 0;
    var template_count: u32 = 0;
    for (s.pages.items) |entry| {
        switch (entry.page.content) {
            .static => static_count += 1,
            .template => template_count += 1,
        }
    }

    return .{
        .total_pages = @intCast(s.pages.items.len),
        .total_renders = s.stat_total_renders.load(.monotonic),
        .static_pages = static_count,
        .template_pages = template_count,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

test "pages basic add and stats" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{ .path = "/dashboard", .title = "Dashboard" });
    try addPage(.{ .path = "/settings", .title = "Settings" });

    const s = stats();
    try std.testing.expectEqual(@as(u32, 2), s.total_pages);
    try std.testing.expectEqual(@as(u32, 2), s.static_pages);
}

test "pages removal" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{ .path = "/temp", .title = "Temp" });
    try std.testing.expectEqual(@as(u32, 1), stats().total_pages);

    const removed = try removePage("/temp");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(u32, 0), stats().total_pages);
}

test "pages static rendering" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/about",
        .title = "About",
        .content = .{ .static = "<h1>About Us</h1>" },
    });

    var result = try renderPage(allocator, "/about", &.{});
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("About", result.title);
    try std.testing.expectEqualStrings("<h1>About Us</h1>", result.body);
    try std.testing.expect(!result.body_owned); // static = not owned
}

test "pages template rendering" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/greet",
        .title = "Greeting",
        .content = .{
            .template = .{
                .source = "Hello, {{name}}! Welcome to {{place}}.",
            },
        },
    });

    var result = try renderPage(allocator, "/greet", &.{
        .{ .key = "name", .value = "Alice" },
        .{ .key = "place", .value = "ABI" },
    });
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("Hello, Alice! Welcome to ABI.", result.body);
    try std.testing.expect(result.body_owned);
}

test "pages path parameter extraction" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/users/{id}",
        .title = "User Profile",
    });

    const match = try matchPage("/users/42");
    try std.testing.expect(match != null);
    try std.testing.expectEqual(@as(u8, 1), match.?.param_count);
    const id_val = match.?.getParam("id");
    try std.testing.expect(id_val != null);
    try std.testing.expectEqualStrings("42", id_val.?);
}

test "pages wildcard matching" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/docs/*",
        .title = "Documentation",
    });

    const match = try matchPage("/docs/api/v2");
    try std.testing.expect(match != null);
    try std.testing.expectEqualStrings("/docs/*", match.?.page.path);
}

test "pages duplicate detection" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{ .path = "/home", .title = "Home" });
    const result = addPage(.{ .path = "/home", .title = "Home 2" });
    try std.testing.expectError(error.DuplicatePage, result);
}

test "pages too many pages limit" {
    const allocator = std.testing.allocator;
    try init(allocator, .{ .max_pages = 2 });
    defer deinit();

    try addPage(.{ .path = "/a", .title = "A" });
    try addPage(.{ .path = "/b", .title = "B" });

    const result = addPage(.{ .path = "/c", .title = "C" });
    try std.testing.expectError(error.TooManyPages, result);
}

test "pages template missing variables render as empty" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/test",
        .title = "Test",
        .content = .{
            .template = .{
                .source = "Value: {{missing}}!",
            },
        },
    });

    var result = try renderPage(allocator, "/test", &.{});
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("Value: !", result.body);
}

test "pages re-initialization guard" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    // Second init should be no-op
    try init(allocator, .{ .max_pages = 1 });
    try addPage(.{ .path = "/a", .title = "A" });
    try addPage(.{ .path = "/b", .title = "B" });
    // If second init took effect, this would fail with TooManyPages
}

test "pages render increments stats" {
    const allocator = std.testing.allocator;
    try init(allocator, PagesConfig.defaults());
    defer deinit();

    try addPage(.{
        .path = "/counter",
        .title = "Counter",
        .content = .{ .static = "count" },
    });

    var r1 = try renderPage(allocator, "/counter", &.{});
    r1.deinit(allocator);
    var r2 = try renderPage(allocator, "/counter", &.{});
    r2.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 2), stats().total_renders);
}
