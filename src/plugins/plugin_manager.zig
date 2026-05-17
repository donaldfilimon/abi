const std = @import("std");
const sync = @import("../foundation/sync.zig");

pub const ManifestSchemaError = error{
    MissingName,
    MissingVersion,
    InvalidVersion,
    MissingEntryPoint,
    InvalidJson,
};

pub const PluginManifest = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    entry_point: []const u8,
};

pub const PluginInfo = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    entry_point: []const u8,
    path: []const u8,
    loaded: bool,
};

pub const PluginLoadError = error{
    FileNotFound,
    InvalidManifest,
    OutOfMemory,
    AlreadyLoaded,
    NotLoaded,
};

const LoadedPlugin = struct {
    info: PluginInfo,
};

pub const PluginManager = struct {
    allocator: std.mem.Allocator,
    plugins: std.StringArrayHashMapUnmanaged(LoadedPlugin),
    lock: sync.RwLock,

    pub fn init(allocator: std.mem.Allocator) PluginManager {
        return .{
            .allocator = allocator,
            .plugins = .{},
            .lock = .{},
        };
    }

    pub fn deinit(self: *PluginManager) void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        var it = self.plugins.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            const plugin = entry.value_ptr.*;
            self.allocator.free(plugin.info.name);
            self.allocator.free(plugin.info.version);
            self.allocator.free(plugin.info.description);
            self.allocator.free(plugin.info.entry_point);
            self.allocator.free(plugin.info.path);
        }
        self.plugins.deinit(self.allocator);
    }

    pub fn validatePlugin(allocator: std.mem.Allocator, manifest_json: []const u8) !PluginManifest {
        var stream = std.json.TokenStream.init(manifest_json);
        const parsed = std.json.parseFromTokenSource(std.json.Value, allocator, &stream, .{}) catch return ManifestSchemaError.InvalidJson;
        defer parsed.deinit();

        const obj = parsed.value.object;

        const name_entry = obj.get("name") orelse return ManifestSchemaError.MissingName;
        const name = if (name_entry == .string) name_entry.string else return ManifestSchemaError.MissingName;
        if (name.len == 0) return ManifestSchemaError.MissingName;

        const version_entry = obj.get("version") orelse return ManifestSchemaError.MissingVersion;
        const version = if (version_entry == .string) version_entry.string else return ManifestSchemaError.InvalidVersion;
        if (version.len == 0) return ManifestSchemaError.InvalidVersion;

        const description_entry = obj.get("description") orelse return .{ .string = "" };
        const description = if (description_entry == .string) description_entry.string else "";

        const entry_entry = obj.get("entry_point") orelse return ManifestSchemaError.MissingEntryPoint;
        const entry_point = if (entry_entry == .string) entry_entry.string else return ManifestSchemaError.MissingEntryPoint;
        if (entry_point.len == 0) return ManifestSchemaError.MissingEntryPoint;

        return .{
            .name = name,
            .version = version,
            .description = description,
            .entry_point = entry_point,
        };
    }

    pub fn loadPlugin(self: *PluginManager, path: []const u8) !PluginInfo {
        const manifest_path = try std.fs.path.join(self.allocator, &.{ path, "abi-plugin.json" });
        defer self.allocator.free(manifest_path);

        const file = std.fs.openFileAbsolute(manifest_path, .{}) catch |err| {
            if (err == error.FileNotFound) return PluginLoadError.FileNotFound;
            return err;
        };
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 1024 * 64) catch |err| {
            if (err == error.OutOfMemory) return PluginLoadError.OutOfMemory;
            return err;
        };
        defer self.allocator.free(content);

        const manifest = validatePlugin(self.allocator, content) catch |err| {
            switch (err) {
                ManifestSchemaError.InvalidJson,
                ManifestSchemaError.MissingName,
                ManifestSchemaError.MissingVersion,
                ManifestSchemaError.InvalidVersion,
                ManifestSchemaError.MissingEntryPoint,
                => return PluginLoadError.InvalidManifest,
            }
        };

        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        if (self.plugins.get(manifest.name) != null) {
            return PluginLoadError.AlreadyLoaded;
        }

        const owned_name = try self.allocator.dupe(u8, manifest.name);
        errdefer self.allocator.free(owned_name);

        const owned_version = try self.allocator.dupe(u8, manifest.version);
        errdefer self.allocator.free(owned_version);

        const owned_description = try self.allocator.dupe(u8, manifest.description);
        errdefer self.allocator.free(owned_description);

        const owned_entry = try self.allocator.dupe(u8, manifest.entry_point);
        errdefer self.allocator.free(owned_entry);

        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        const plugin_info = PluginInfo{
            .name = owned_name,
            .version = owned_version,
            .description = owned_description,
            .entry_point = owned_entry,
            .path = owned_path,
            .loaded = true,
        };

        const loaded = LoadedPlugin{ .info = plugin_info };
        try self.plugins.put(self.allocator, owned_name, loaded);

        return plugin_info;
    }

    pub fn unloadPlugin(self: *PluginManager, name: []const u8) !void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        const entry = self.plugins.getEntry(name) orelse return PluginLoadError.NotLoaded;

        self.allocator.free(entry.value_ptr.*.info.name);
        self.allocator.free(entry.value_ptr.*.info.version);
        self.allocator.free(entry.value_ptr.*.info.description);
        self.allocator.free(entry.value_ptr.*.info.entry_point);
        self.allocator.free(entry.value_ptr.*.info.path);

        _ = self.plugins.swapRemove(name);
    }

    pub fn getPlugin(self: *PluginManager, name: []const u8) !PluginInfo {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        const entry = self.plugins.getEntry(name) orelse return PluginLoadError.NotLoaded;
        return entry.value_ptr.*.info;
    }

    pub fn listPlugins(self: *PluginManager) ![]const PluginInfo {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        const result = try self.allocator.alloc(PluginInfo, self.plugins.count());
        errdefer self.allocator.free(result);

        for (self.plugins.values(), 0..) |plugin, i| {
            result[i] = plugin.info;
        }

        return result;
    }

    pub fn pluginCount(self: *PluginManager) usize {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        return self.plugins.count();
    }
};

test "plugin manager validates correct manifest" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "description": "A test", "entry_point": "mod.zig"}
    ;
    const parsed = try PluginManager.validatePlugin(std.testing.allocator, manifest_json);
    try std.testing.expectEqualStrings("test-plugin", parsed.name);
    try std.testing.expectEqualStrings("1.0.0", parsed.version);
    try std.testing.expectEqualStrings("mod.zig", parsed.entry_point);
}

test "plugin manager rejects manifest missing name" {
    const manifest_json =
        \\{"version": "1.0.0", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingName, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing version" {
    const manifest_json =
        \\{"name": "test", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingVersion, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing entry_point" {
    const manifest_json =
        \\{"name": "test", "version": "1.0.0"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingEntryPoint, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager load and unload" {
    var manager = PluginManager.init(std.testing.allocator);
    defer manager.deinit();

    const info = try manager.loadPlugin("src/plugins/example-plugin");
    try std.testing.expectEqualStrings("example-plugin", info.name);
    try std.testing.expect(info.loaded);

    try std.testing.expectEqual(@as(usize, 1), manager.pluginCount());

    try manager.unloadPlugin("example-plugin");
    try std.testing.expectEqual(@as(usize, 0), manager.pluginCount());
}

test "plugin manager rejects duplicate load" {
    var manager = PluginManager.init(std.testing.allocator);
    defer manager.deinit();

    _ = try manager.loadPlugin("src/plugins/example-plugin");
    try std.testing.expectError(PluginLoadError.AlreadyLoaded, manager.loadPlugin("src/plugins/example-plugin"));
}

test "plugin manager getPlugin returns not loaded" {
    var manager = PluginManager.init(std.testing.allocator);
    defer manager.deinit();

    try std.testing.expectError(PluginLoadError.NotLoaded, manager.getPlugin("nonexistent"));
}

test "plugin manager listPlugins returns all loaded" {
    var manager = PluginManager.init(std.testing.allocator);
    defer manager.deinit();

    _ = try manager.loadPlugin("src/plugins/example-plugin");
    const list = try manager.listPlugins();
    defer std.testing.allocator.free(list);

    try std.testing.expectEqual(@as(usize, 1), list.len);
    try std.testing.expectEqualStrings("example-plugin", list[0].name);
}
