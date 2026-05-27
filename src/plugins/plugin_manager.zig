const std = @import("std");
const sync = @import("../foundation/sync.zig");

pub const ManifestSchemaError = error{
    MissingName,
    MissingVersion,
    InvalidVersion,
    MissingDescription,
    MissingEntryPoint,
    InvalidEntryPoint,
    MissingTargetFeature,
    InvalidJson,
};

pub const PluginManifest = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
    entry_point: []const u8,

    pub fn deinit(self: PluginManifest, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.description);
        allocator.free(self.target_feature);
        allocator.free(self.entry_point);
    }
};

pub const PluginInfo = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
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

pub fn isSafeEntryPoint(entry_point: []const u8) bool {
    if (entry_point.len == 0) return false;
    if (!std.mem.endsWith(u8, entry_point, ".zig")) return false;
    if (entry_point[0] == '/' or entry_point[0] == '\\') return false;
    if (std.mem.indexOfScalar(u8, entry_point, ':') != null) return false;

    var parts = std.mem.splitAny(u8, entry_point, "/\\");
    while (parts.next()) |part| {
        if (part.len == 0) return false;
        if (std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return true;
}

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
            self.allocator.free(plugin.info.target_feature);
            self.allocator.free(plugin.info.entry_point);
            self.allocator.free(plugin.info.path);
        }
        self.plugins.deinit(self.allocator);
    }

    pub fn validatePlugin(allocator: std.mem.Allocator, manifest_json: []const u8) !PluginManifest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, manifest_json, .{}) catch return ManifestSchemaError.InvalidJson;
        defer parsed.deinit();

        if (parsed.value != .object) return ManifestSchemaError.InvalidJson;

        const obj = parsed.value.object;

        const name_entry = obj.get("name") orelse return ManifestSchemaError.MissingName;
        const name = if (name_entry == .string) name_entry.string else return ManifestSchemaError.MissingName;
        if (name.len == 0) return ManifestSchemaError.MissingName;

        const version_entry = obj.get("version") orelse return ManifestSchemaError.MissingVersion;
        const version = if (version_entry == .string) version_entry.string else return ManifestSchemaError.InvalidVersion;
        if (version.len == 0) return ManifestSchemaError.InvalidVersion;

        const description_entry = obj.get("description") orelse return ManifestSchemaError.MissingDescription;
        const description = if (description_entry == .string) description_entry.string else return ManifestSchemaError.MissingDescription;
        if (description.len == 0) return ManifestSchemaError.MissingDescription;

        const target_entry = obj.get("target_feature") orelse obj.get("targetFeature") orelse return ManifestSchemaError.MissingTargetFeature;
        const target_feature = if (target_entry == .string) target_entry.string else return ManifestSchemaError.MissingTargetFeature;
        if (target_feature.len == 0) return ManifestSchemaError.MissingTargetFeature;

        const entry_entry = obj.get("entry_point") orelse obj.get("entryPoint") orelse return ManifestSchemaError.MissingEntryPoint;
        const entry_point = if (entry_entry == .string) entry_entry.string else return ManifestSchemaError.MissingEntryPoint;
        if (entry_point.len == 0) return ManifestSchemaError.MissingEntryPoint;
        if (!isSafeEntryPoint(entry_point)) return ManifestSchemaError.InvalidEntryPoint;

        const owned_name = try allocator.dupe(u8, name);
        errdefer allocator.free(owned_name);

        const owned_version = try allocator.dupe(u8, version);
        errdefer allocator.free(owned_version);

        const owned_description = try allocator.dupe(u8, description);
        errdefer allocator.free(owned_description);

        const owned_target = try allocator.dupe(u8, target_feature);
        errdefer allocator.free(owned_target);

        const owned_entry = try allocator.dupe(u8, entry_point);
        errdefer allocator.free(owned_entry);

        return .{
            .name = owned_name,
            .version = owned_version,
            .description = owned_description,
            .target_feature = owned_target,
            .entry_point = owned_entry,
        };
    }

    pub fn loadPlugin(self: *PluginManager, path: []const u8) !PluginInfo {
        const manifest_path = try std.fs.path.join(self.allocator, &.{ path, "abi-plugin.json" });
        defer self.allocator.free(manifest_path);

        const content = std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, manifest_path, self.allocator, .limited(1024 * 64)) catch |err| {
            if (err == error.FileNotFound) return PluginLoadError.FileNotFound;
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
                ManifestSchemaError.MissingDescription,
                ManifestSchemaError.MissingEntryPoint,
                ManifestSchemaError.InvalidEntryPoint,
                ManifestSchemaError.MissingTargetFeature,
                => return PluginLoadError.InvalidManifest,
                error.OutOfMemory => return PluginLoadError.OutOfMemory,
            }
        };
        errdefer manifest.deinit(self.allocator);

        const entry_path = try std.fs.path.join(self.allocator, &.{ path, manifest.entry_point });
        defer self.allocator.free(entry_path);
        _ = std.Io.Dir.cwd().statFile(std.Options.debug_io, entry_path, .{}) catch return PluginLoadError.InvalidManifest;

        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        if (self.plugins.get(manifest.name) != null) {
            return PluginLoadError.AlreadyLoaded;
        }

        const owned_key = try self.allocator.dupe(u8, manifest.name);
        errdefer self.allocator.free(owned_key);

        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        const plugin_info = PluginInfo{
            .name = manifest.name,
            .version = manifest.version,
            .description = manifest.description,
            .target_feature = manifest.target_feature,
            .entry_point = manifest.entry_point,
            .path = owned_path,
            .loaded = true,
        };

        const loaded = LoadedPlugin{ .info = plugin_info };
        try self.plugins.put(self.allocator, owned_key, loaded);

        return plugin_info;
    }

    pub fn unloadPlugin(self: *PluginManager, name: []const u8) !void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        const entry = self.plugins.getEntry(name) orelse return PluginLoadError.NotLoaded;
        const owned_key = entry.key_ptr.*;
        const info = entry.value_ptr.*.info;

        _ = self.plugins.swapRemove(name);

        self.allocator.free(owned_key);
        self.allocator.free(info.name);
        self.allocator.free(info.version);
        self.allocator.free(info.description);
        self.allocator.free(info.target_feature);
        self.allocator.free(info.entry_point);
        self.allocator.free(info.path);
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

test {
    std.testing.refAllDecls(@This());
}

test "plugin manager validates correct manifest" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    const parsed = try PluginManager.validatePlugin(std.testing.allocator, manifest_json);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("test-plugin", parsed.name);
    try std.testing.expectEqualStrings("1.0.0", parsed.version);
    try std.testing.expectEqualStrings("A test", parsed.description);
    try std.testing.expectEqualStrings("ai", parsed.target_feature);
    try std.testing.expectEqualStrings("mod.zig", parsed.entry_point);
}

test "plugin manager validates camelCase manifest aliases" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "description": "A test", "targetFeature": "ai", "entryPoint": "nested/mod.zig"}
    ;
    const parsed = try PluginManager.validatePlugin(std.testing.allocator, manifest_json);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("ai", parsed.target_feature);
    try std.testing.expectEqualStrings("nested/mod.zig", parsed.entry_point);
}

test "plugin manager rejects invalid manifest shapes" {
    try std.testing.expectError(ManifestSchemaError.InvalidJson, PluginManager.validatePlugin(std.testing.allocator, "not json"));
    try std.testing.expectError(ManifestSchemaError.InvalidJson, PluginManager.validatePlugin(std.testing.allocator, "[]"));
    try std.testing.expectError(ManifestSchemaError.MissingName, PluginManager.validatePlugin(std.testing.allocator,
        \\{"name": 123, "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.InvalidVersion, PluginManager.validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingDescription, PluginManager.validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingTargetFeature, PluginManager.validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingEntryPoint, PluginManager.validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": ""}
    ));
}

test "plugin manager rejects manifest missing description" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingDescription, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing name" {
    const manifest_json =
        \\{"version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingName, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing version" {
    const manifest_json =
        \\{"name": "test", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingVersion, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing entry_point" {
    const manifest_json =
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "ai"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingEntryPoint, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing target_feature" {
    const manifest_json =
        \\{"name": "test", "version": "1.0.0", "description": "A test", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingTargetFeature, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects unsafe entry points" {
    try std.testing.expect(isSafeEntryPoint("mod.zig"));
    try std.testing.expect(isSafeEntryPoint("nested/mod.zig"));
    try std.testing.expect(!isSafeEntryPoint("../mod.zig"));
    try std.testing.expect(!isSafeEntryPoint("/tmp/mod.zig"));
    try std.testing.expect(!isSafeEntryPoint("mod.so"));

    const manifest_json =
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "../mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.InvalidEntryPoint, PluginManager.validatePlugin(std.testing.allocator, manifest_json));
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
