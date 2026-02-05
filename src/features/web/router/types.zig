//! Router Types
//!
//! Core types for HTTP request routing.

const std = @import("std");

/// HTTP method enum (re-exported from std).
pub const Method = std.http.Method;

/// Route handler function signature.
pub const HandlerFn = *const fn (ctx: *anyopaque) anyerror!void;

/// A registered route.
pub const Route = struct {
    /// HTTP method for this route.
    method: Method,
    /// URL pattern (supports :param syntax).
    pattern: []const u8,
    /// Handler function.
    handler: HandlerFn,
    /// Route name (for reverse routing).
    name: ?[]const u8 = null,

    /// Checks if this route matches the given method and path.
    pub fn matches(self: *const Route, method: Method, path: []const u8) bool {
        if (self.method != method) return false;
        return matchPattern(self.pattern, path);
    }
};

/// Result of a route match.
pub const RouteMatch = struct {
    /// The matched route.
    route: *const Route,
    /// Extracted path parameters.
    params: std.StringHashMap([]const u8),
    /// Allocator used for params.
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RouteMatch) void {
        self.params.deinit();
    }

    /// Gets a path parameter by name.
    pub fn getParam(self: *const RouteMatch, name: []const u8) ?[]const u8 {
        return self.params.get(name);
    }
};

/// Route group for organizing routes with common prefix/middleware.
pub const RouteGroup = struct {
    prefix: []const u8,
    routes: std.ArrayList(Route),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, prefix: []const u8) RouteGroup {
        return .{
            .prefix = prefix,
            .routes = std.ArrayList(Route).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RouteGroup) void {
        self.routes.deinit();
    }

    pub fn addRoute(self: *RouteGroup, method: Method, pattern: []const u8, handler: HandlerFn) !void {
        try self.routes.append(.{
            .method = method,
            .pattern = pattern,
            .handler = handler,
        });
    }
};

/// Checks if a pattern matches a path.
pub fn matchPattern(pattern: []const u8, path: []const u8) bool {
    var pattern_parts = std.mem.splitScalar(u8, pattern, '/');
    var path_parts = std.mem.splitScalar(u8, path, '/');

    while (true) {
        const pattern_part = pattern_parts.next();
        const path_part = path_parts.next();

        // Both exhausted = match
        if (pattern_part == null and path_part == null) {
            return true;
        }

        // One exhausted before other = no match
        if (pattern_part == null or path_part == null) {
            return false;
        }

        const pp = pattern_part.?;
        const pa = path_part.?;

        // Empty parts (from leading/trailing slashes)
        if (pp.len == 0 and pa.len == 0) {
            continue;
        }

        // Parameter matches anything non-empty
        if (pp.len > 0 and pp[0] == ':') {
            if (pa.len == 0) return false;
            continue;
        }

        // Wildcard matches rest of path
        if (std.mem.eql(u8, pp, "*")) {
            return true;
        }

        // Double wildcard matches any depth
        if (std.mem.eql(u8, pp, "**")) {
            return true;
        }

        // Exact match required
        if (!std.mem.eql(u8, pp, pa)) {
            return false;
        }
    }
}

/// Extracts parameters from a path using a pattern.
pub fn extractParams(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    path: []const u8,
) !std.StringHashMap([]const u8) {
    var params = std.StringHashMap([]const u8).init(allocator);
    errdefer params.deinit();

    var pattern_parts = std.mem.splitScalar(u8, pattern, '/');
    var path_parts = std.mem.splitScalar(u8, path, '/');

    while (pattern_parts.next()) |pattern_part| {
        const path_part = path_parts.next() orelse break;

        if (pattern_part.len > 0 and pattern_part[0] == ':') {
            const param_name = pattern_part[1..];
            try params.put(param_name, path_part);
        }
    }

    return params;
}

/// Normalizes a path (removes trailing slash, etc.).
pub fn normalizePath(path: []const u8) []const u8 {
    if (path.len == 0) return "/";
    if (path.len == 1) return path;

    // Remove trailing slash (except for root)
    if (path[path.len - 1] == '/') {
        return path[0 .. path.len - 1];
    }

    return path;
}

/// Joins path segments.
pub fn joinPath(allocator: std.mem.Allocator, segments: []const []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    for (segments) |segment| {
        if (segment.len == 0) continue;

        // Add separator if needed
        if (result.items.len > 0 and result.items[result.items.len - 1] != '/') {
            if (segment[0] != '/') {
                try result.append('/');
            }
        }

        // Skip leading slash if we already have one
        const start: usize = if (result.items.len > 0 and
            result.items[result.items.len - 1] == '/' and
            segment[0] == '/') 1 else 0;

        try result.appendSlice(segment[start..]);
    }

    if (result.items.len == 0) {
        try result.append('/');
    }

    return result.toOwnedSlice();
}

test "matchPattern basic" {
    try std.testing.expect(matchPattern("/", "/"));
    try std.testing.expect(matchPattern("/users", "/users"));
    try std.testing.expect(!matchPattern("/users", "/posts"));
    try std.testing.expect(!matchPattern("/users", "/users/123"));
}

test "matchPattern with params" {
    try std.testing.expect(matchPattern("/users/:id", "/users/123"));
    try std.testing.expect(matchPattern("/users/:id", "/users/abc"));
    try std.testing.expect(!matchPattern("/users/:id", "/users/"));
    try std.testing.expect(matchPattern("/users/:id/posts/:post_id", "/users/1/posts/2"));
}

test "matchPattern with wildcard" {
    try std.testing.expect(matchPattern("/api/*", "/api/anything"));
    try std.testing.expect(matchPattern("/api/**", "/api/deep/nested/path"));
}

test "extractParams" {
    const allocator = std.testing.allocator;

    var params = try extractParams(allocator, "/users/:id/posts/:post_id", "/users/123/posts/456");
    defer params.deinit();

    try std.testing.expectEqualStrings("123", params.get("id").?);
    try std.testing.expectEqualStrings("456", params.get("post_id").?);
}

test "normalizePath" {
    try std.testing.expectEqualStrings("/", normalizePath(""));
    try std.testing.expectEqualStrings("/", normalizePath("/"));
    try std.testing.expectEqualStrings("/users", normalizePath("/users"));
    try std.testing.expectEqualStrings("/users", normalizePath("/users/"));
}

test "joinPath" {
    const allocator = std.testing.allocator;

    const path1 = try joinPath(allocator, &.{ "/api", "users" });
    defer allocator.free(path1);
    try std.testing.expectEqualStrings("/api/users", path1);

    const path2 = try joinPath(allocator, &.{ "/api/", "/users" });
    defer allocator.free(path2);
    try std.testing.expectEqualStrings("/api/users", path2);

    const path3 = try joinPath(allocator, &.{});
    defer allocator.free(path3);
    try std.testing.expectEqualStrings("/", path3);
}
