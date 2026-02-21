const std = @import("std");
const connectors = @import("../../../../../services/connectors/mod.zig");

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
    const home = try connectors.getFirstEnvOwned(allocator, &.{ "HOME", "USERPROFILE" }) orelse return error.NoHomeDirectory;
    defer allocator.free(home);

    return std.fmt.allocPrint(allocator, "{s}/.abi/llm_plugins.json", .{home});
}

pub fn loadDefault(allocator: std.mem.Allocator) !Manifest {
    const path = defaultPath(allocator) catch {
        return Manifest.init(allocator);
    };
    defer allocator.free(path);

    return loadFromFile(allocator, path) catch {
        return Manifest.init(allocator);
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
