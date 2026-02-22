const std = @import("std");
const app_paths = @import("../../../../../services/shared/app_paths.zig");

pub const PluginKind = enum {
    http,
    native,

    pub fn fromString(value: []const u8) ?PluginKind {
        if (std.mem.eql(u8, value, "http")) return .http;
        if (std.mem.eql(u8, value, "native")) return .native;
        return null;
    }

    pub fn label(self: PluginKind) []const u8 {
        return switch (self) {
            .http => "http",
            .native => "native",
        };
    }
};

pub const PluginEntry = struct {
    id: []u8,
    kind: PluginKind,
    enabled: bool = true,
    base_url: ?[]u8 = null,
    model: ?[]u8 = null,
    api_key_env: ?[]u8 = null,
    library_path: ?[]u8 = null,
    symbol: ?[]u8 = null,

    pub fn deinit(self: *PluginEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        freeOptional(allocator, &self.base_url);
        freeOptional(allocator, &self.model);
        freeOptional(allocator, &self.api_key_env);
        freeOptional(allocator, &self.library_path);
        freeOptional(allocator, &self.symbol);
        self.* = undefined;
    }
};

pub const Manifest = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(PluginEntry) = .empty,

    pub fn init(allocator: std.mem.Allocator) Manifest {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Manifest) void {
        for (self.entries.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn findIndex(self: *const Manifest, id: []const u8) ?usize {
        for (self.entries.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.id, id)) return i;
        }
        return null;
    }

    pub fn find(self: *const Manifest, id: []const u8) ?*const PluginEntry {
        const idx = self.findIndex(id) orelse return null;
        return &self.entries.items[idx];
    }

    pub fn findPtr(self: *Manifest, id: []const u8) ?*PluginEntry {
        const idx = self.findIndex(id) orelse return null;
        return &self.entries.items[idx];
    }

    pub fn addOrUpdateHttp(
        self: *Manifest,
        id: []const u8,
        base_url: []const u8,
        model: ?[]const u8,
        api_key_env: ?[]const u8,
    ) !void {
        if (self.findPtr(id)) |entry| {
            freeOptional(self.allocator, &entry.base_url);
            freeOptional(self.allocator, &entry.model);
            freeOptional(self.allocator, &entry.api_key_env);
            freeOptional(self.allocator, &entry.library_path);
            freeOptional(self.allocator, &entry.symbol);

            entry.kind = .http;
            entry.base_url = try self.allocator.dupe(u8, base_url);
            entry.model = if (model) |v| try self.allocator.dupe(u8, v) else null;
            entry.api_key_env = if (api_key_env) |v| try self.allocator.dupe(u8, v) else null;
            entry.enabled = true;
            return;
        }

        try self.entries.append(self.allocator, .{
            .id = try self.allocator.dupe(u8, id),
            .kind = .http,
            .enabled = true,
            .base_url = try self.allocator.dupe(u8, base_url),
            .model = if (model) |v| try self.allocator.dupe(u8, v) else null,
            .api_key_env = if (api_key_env) |v| try self.allocator.dupe(u8, v) else null,
            .library_path = null,
            .symbol = null,
        });
    }

    pub fn addOrUpdateNative(
        self: *Manifest,
        id: []const u8,
        library_path: []const u8,
        symbol: ?[]const u8,
    ) !void {
        if (self.findPtr(id)) |entry| {
            freeOptional(self.allocator, &entry.base_url);
            freeOptional(self.allocator, &entry.model);
            freeOptional(self.allocator, &entry.api_key_env);
            freeOptional(self.allocator, &entry.library_path);
            freeOptional(self.allocator, &entry.symbol);

            entry.kind = .native;
            entry.library_path = try self.allocator.dupe(u8, library_path);
            entry.symbol = if (symbol) |v| try self.allocator.dupe(u8, v) else null;
            entry.enabled = true;
            return;
        }

        try self.entries.append(self.allocator, .{
            .id = try self.allocator.dupe(u8, id),
            .kind = .native,
            .enabled = true,
            .base_url = null,
            .model = null,
            .api_key_env = null,
            .library_path = try self.allocator.dupe(u8, library_path),
            .symbol = if (symbol) |v| try self.allocator.dupe(u8, v) else null,
        });
    }

    pub fn setEnabled(self: *Manifest, id: []const u8, enabled: bool) bool {
        if (self.findPtr(id)) |entry| {
            entry.enabled = enabled;
            return true;
        }
        return false;
    }

    pub fn remove(self: *Manifest, id: []const u8) bool {
        const idx = self.findIndex(id) orelse return false;
        var removed = self.entries.orderedRemove(idx);
        removed.deinit(self.allocator);
        return true;
    }
};

pub fn cloneEntry(allocator: std.mem.Allocator, entry: PluginEntry) !PluginEntry {
    return .{
        .id = try allocator.dupe(u8, entry.id),
        .kind = entry.kind,
        .enabled = entry.enabled,
        .base_url = if (entry.base_url) |v| try allocator.dupe(u8, v) else null,
        .model = if (entry.model) |v| try allocator.dupe(u8, v) else null,
        .api_key_env = if (entry.api_key_env) |v| try allocator.dupe(u8, v) else null,
        .library_path = if (entry.library_path) |v| try allocator.dupe(u8, v) else null,
        .symbol = if (entry.symbol) |v| try allocator.dupe(u8, v) else null,
    };
}

pub fn defaultPath(allocator: std.mem.Allocator) ![]u8 {
    return app_paths.resolvePath(allocator, "llm_plugins.json");
}

pub fn loadDefault(allocator: std.mem.Allocator) !Manifest {
    const path = defaultPath(allocator) catch return Manifest.init(allocator);
    defer allocator.free(path);
    return loadFromFile(allocator, path) catch Manifest.init(allocator);
}

pub fn loadDefaultFromPaths(allocator: std.mem.Allocator, primary_path: []const u8, legacy_path: []const u8) !Manifest {
    return loadFromFile(allocator, primary_path) catch {
        return loadFromFile(allocator, legacy_path) catch Manifest.init(allocator);
    };
}

pub fn saveDefault(manifest: *const Manifest) !void {
    const path = try defaultPath(manifest.allocator);
    defer manifest.allocator.free(path);
    try saveToFile(manifest, path);
}

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Manifest {
    var out = Manifest.init(allocator);
    errdefer out.deinit();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(raw);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return out;
    const plugins = parsed.value.object.get("plugins") orelse return out;
    if (plugins != .array) return out;

    for (plugins.array.items) |entry_value| {
        if (entry_value != .object) continue;
        const obj = entry_value.object;

        const id_val = obj.get("id") orelse continue;
        if (id_val != .string) continue;

        const kind_val = obj.get("kind") orelse continue;
        if (kind_val != .string) continue;
        const kind = PluginKind.fromString(kind_val.string) orelse continue;

        const enabled = if (obj.get("enabled")) |v| (if (v == .bool) v.bool else true) else true;

        var plugin = PluginEntry{
            .id = try allocator.dupe(u8, id_val.string),
            .kind = kind,
            .enabled = enabled,
            .base_url = null,
            .model = null,
            .api_key_env = null,
            .library_path = null,
            .symbol = null,
        };
        errdefer plugin.deinit(allocator);

        if (obj.get("base_url")) |v| {
            if (v == .string) plugin.base_url = try allocator.dupe(u8, v.string);
        }
        if (obj.get("model")) |v| {
            if (v == .string) plugin.model = try allocator.dupe(u8, v.string);
        }
        if (obj.get("api_key_env")) |v| {
            if (v == .string) plugin.api_key_env = try allocator.dupe(u8, v.string);
        }
        if (obj.get("library_path")) |v| {
            if (v == .string) plugin.library_path = try allocator.dupe(u8, v.string);
        }
        if (obj.get("symbol")) |v| {
            if (v == .string) plugin.symbol = try allocator.dupe(u8, v.string);
        }

        try out.entries.append(allocator, plugin);
    }

    return out;
}

pub fn saveToFile(manifest: *const Manifest, path: []const u8) !void {
    var io_backend = std.Io.Threaded.init(manifest.allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    if (std.mem.lastIndexOfScalar(u8, path, '/')) |idx| {
        if (idx > 0) {
            const dir_path = path[0..idx];
            std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};
        }
    }

    var json = std.ArrayListUnmanaged(u8).empty;
    defer json.deinit(manifest.allocator);

    try json.appendSlice(manifest.allocator, "{\"plugins\":[");
    for (manifest.entries.items, 0..) |entry, i| {
        if (i > 0) try json.append(manifest.allocator, ',');

        try json.appendSlice(manifest.allocator, "{\"id\":\"");
        try appendEscaped(manifest.allocator, &json, entry.id);
        try json.appendSlice(manifest.allocator, "\",\"kind\":\"");
        try json.appendSlice(manifest.allocator, entry.kind.label());
        try json.appendSlice(manifest.allocator, "\",\"enabled\":");
        try json.appendSlice(manifest.allocator, if (entry.enabled) "true" else "false");

        if (entry.base_url) |value| {
            try json.appendSlice(manifest.allocator, ",\"base_url\":\"");
            try appendEscaped(manifest.allocator, &json, value);
            try json.append(manifest.allocator, '"');
        }
        if (entry.model) |value| {
            try json.appendSlice(manifest.allocator, ",\"model\":\"");
            try appendEscaped(manifest.allocator, &json, value);
            try json.append(manifest.allocator, '"');
        }
        if (entry.api_key_env) |value| {
            try json.appendSlice(manifest.allocator, ",\"api_key_env\":\"");
            try appendEscaped(manifest.allocator, &json, value);
            try json.append(manifest.allocator, '"');
        }
        if (entry.library_path) |value| {
            try json.appendSlice(manifest.allocator, ",\"library_path\":\"");
            try appendEscaped(manifest.allocator, &json, value);
            try json.append(manifest.allocator, '"');
        }
        if (entry.symbol) |value| {
            try json.appendSlice(manifest.allocator, ",\"symbol\":\"");
            try appendEscaped(manifest.allocator, &json, value);
            try json.append(manifest.allocator, '"');
        }

        try json.append(manifest.allocator, '}');
    }

    try json.appendSlice(manifest.allocator, "]}\n");

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, json.items);
}

fn appendEscaped(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), value: []const u8) !void {
    for (value) |ch| {
        switch (ch) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, ch),
        }
    }
}

fn freeOptional(allocator: std.mem.Allocator, value: *?[]u8) void {
    if (value.*) |slice| {
        allocator.free(slice);
        value.* = null;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "Manifest.init creates empty manifest" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try std.testing.expectEqual(@as(usize, 0), m.entries.items.len);
}

test "addOrUpdateHttp adds new entry" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("test-http", "http://localhost:8080", "gpt-4", "MY_KEY");

    try std.testing.expectEqual(@as(usize, 1), m.entries.items.len);
    const entry = m.entries.items[0];
    try std.testing.expectEqual(PluginKind.http, entry.kind);
    try std.testing.expect(entry.enabled);
    try std.testing.expectEqualStrings("test-http", entry.id);
    try std.testing.expectEqualStrings("http://localhost:8080", entry.base_url.?);
    try std.testing.expectEqualStrings("gpt-4", entry.model.?);
    try std.testing.expectEqualStrings("MY_KEY", entry.api_key_env.?);
    try std.testing.expect(entry.library_path == null);
}

test "addOrUpdateHttp updates existing entry" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("plugin1", "http://old.host", "old-model", null);
    try m.addOrUpdateHttp("plugin1", "http://new.host", "new-model", "KEY_ENV");

    // Should still be 1 entry, not 2
    try std.testing.expectEqual(@as(usize, 1), m.entries.items.len);
    const entry = m.entries.items[0];
    try std.testing.expectEqualStrings("http://new.host", entry.base_url.?);
    try std.testing.expectEqualStrings("new-model", entry.model.?);
    try std.testing.expectEqualStrings("KEY_ENV", entry.api_key_env.?);
}

test "addOrUpdateNative adds new entry" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateNative("test-native", "/usr/lib/plugin.dylib", "my_symbol");

    try std.testing.expectEqual(@as(usize, 1), m.entries.items.len);
    const entry = m.entries.items[0];
    try std.testing.expectEqual(PluginKind.native, entry.kind);
    try std.testing.expect(entry.enabled);
    try std.testing.expectEqualStrings("/usr/lib/plugin.dylib", entry.library_path.?);
    try std.testing.expectEqualStrings("my_symbol", entry.symbol.?);
    try std.testing.expect(entry.base_url == null);
}

test "addOrUpdateNative updates existing entry" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateNative("p1", "/old/path.dylib", null);
    try m.addOrUpdateNative("p1", "/new/path.dylib", "new_sym");

    try std.testing.expectEqual(@as(usize, 1), m.entries.items.len);
    try std.testing.expectEqualStrings("/new/path.dylib", m.entries.items[0].library_path.?);
    try std.testing.expectEqualStrings("new_sym", m.entries.items[0].symbol.?);
}

test "find returns entry when present" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("alpha", "http://a", null, null);
    try m.addOrUpdateHttp("beta", "http://b", null, null);

    const found = m.find("beta");
    try std.testing.expect(found != null);
    try std.testing.expectEqualStrings("beta", found.?.id);

    try std.testing.expect(m.find("nonexistent") == null);
}

test "findIndex returns correct index" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("first", "http://1", null, null);
    try m.addOrUpdateHttp("second", "http://2", null, null);
    try m.addOrUpdateHttp("third", "http://3", null, null);

    try std.testing.expectEqual(@as(?usize, 0), m.findIndex("first"));
    try std.testing.expectEqual(@as(?usize, 1), m.findIndex("second"));
    try std.testing.expectEqual(@as(?usize, 2), m.findIndex("third"));
    try std.testing.expectEqual(@as(?usize, null), m.findIndex("missing"));
}

test "remove removes entry and returns true" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("a", "http://a", null, null);
    try m.addOrUpdateHttp("b", "http://b", null, null);

    try std.testing.expect(m.remove("a"));
    try std.testing.expectEqual(@as(usize, 1), m.entries.items.len);
    try std.testing.expectEqualStrings("b", m.entries.items[0].id);

    // Removing non-existent returns false
    try std.testing.expect(!m.remove("a"));
}

test "setEnabled toggles enabled flag" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("p", "http://p", null, null);
    try std.testing.expect(m.entries.items[0].enabled);

    try std.testing.expect(m.setEnabled("p", false));
    try std.testing.expect(!m.entries.items[0].enabled);

    try std.testing.expect(m.setEnabled("p", true));
    try std.testing.expect(m.entries.items[0].enabled);

    // Non-existent returns false
    try std.testing.expect(!m.setEnabled("nope", false));
}

test "PluginKind.fromString and label round-trip" {
    try std.testing.expectEqual(PluginKind.http, PluginKind.fromString("http").?);
    try std.testing.expectEqual(PluginKind.native, PluginKind.fromString("native").?);
    try std.testing.expect(PluginKind.fromString("unknown") == null);

    try std.testing.expectEqualStrings("http", PluginKind.http.label());
    try std.testing.expectEqualStrings("native", PluginKind.native.label());
}

test "cloneEntry produces independent copy" {
    const allocator = std.testing.allocator;
    var m = Manifest.init(allocator);
    defer m.deinit();

    try m.addOrUpdateHttp("orig", "http://example.com", "model-1", "KEY");

    var clone = try cloneEntry(allocator, m.entries.items[0]);
    defer clone.deinit(allocator);

    try std.testing.expectEqualStrings("orig", clone.id);
    try std.testing.expectEqualStrings("http://example.com", clone.base_url.?);
    try std.testing.expectEqualStrings("model-1", clone.model.?);
    try std.testing.expectEqualStrings("KEY", clone.api_key_env.?);
    try std.testing.expectEqual(PluginKind.http, clone.kind);
    // Verify it's an independent copy (different pointer)
    try std.testing.expect(clone.id.ptr != m.entries.items[0].id.ptr);
}

test "loadDefaultFromPaths falls back to legacy when primary is missing" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const sub_path_len = std.mem.indexOfScalar(u8, tmp.sub_path[0..], 0) orelse tmp.sub_path.len;
    const base = try std.fmt.allocPrint(allocator, ".zig-cache/{s}", .{tmp.sub_path[0..sub_path_len]});
    defer allocator.free(base);

    const primary_path = try std.fs.path.join(allocator, &.{ base, "llm_plugins.primary.json" });
    defer allocator.free(primary_path);
    const legacy_path = try std.fs.path.join(allocator, &.{ base, "llm_plugins.legacy.json" });
    defer allocator.free(legacy_path);

    var legacy_manifest = Manifest.init(allocator);
    defer legacy_manifest.deinit();
    try legacy_manifest.addOrUpdateHttp("legacy-http", "http://localhost:9000", "qwen", "ABI_TEST_KEY");
    try saveToFile(&legacy_manifest, legacy_path);

    var loaded = try loadDefaultFromPaths(allocator, primary_path, legacy_path);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 1), loaded.entries.items.len);
    try std.testing.expectEqualStrings("legacy-http", loaded.entries.items[0].id);
    try std.testing.expectEqual(PluginKind.http, loaded.entries.items[0].kind);
}

test {
    std.testing.refAllDecls(@This());
}
