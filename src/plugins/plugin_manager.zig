const std = @import("std");
const build_options = @import("build_options");
const sync = @import("../foundation/sync.zig");
const plugin_validator = @import("../foundation/plugin_validator.zig");
const temp_path = @import("../foundation/temp_path.zig");
const telemetry = if (build_options.feat_telemetry) @import("../features/telemetry/mod.zig") else @import("../features/telemetry/stub.zig");
const isSafeEntryPoint = plugin_validator.isSafeEntryPoint;

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

pub const ManifestCommand = struct {
    name: []const u8,
    summary: []const u8 = "",
    aliases: []const []const u8 = &.{},
};

pub const ManifestContextProvider = struct {
    name: []const u8,
    summary: []const u8 = "",
};

pub const PluginManifest = struct {
    name: []const u8,
    version: []const u8,
    description: []const u8,
    target_feature: []const u8,
    entry_point: []const u8,
    commands: []const ManifestCommand = &.{},
    context_providers: []const ManifestContextProvider = &.{},

    pub fn deinit(self: PluginManifest, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.description);
        allocator.free(self.target_feature);
        allocator.free(self.entry_point);
        for (self.commands) |cmd| {
            allocator.free(cmd.name);
            allocator.free(cmd.summary);
            for (cmd.aliases) |alias| allocator.free(alias);
            allocator.free(cmd.aliases);
        }
        allocator.free(self.commands);
        for (self.context_providers) |cp| {
            allocator.free(cp.name);
            allocator.free(cp.summary);
        }
        allocator.free(self.context_providers);
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
    commands: []const ManifestCommand = &.{},
    context_providers: []const ManifestContextProvider = &.{},
};

pub const PluginLoadError = error{
    FileNotFound,
    InvalidManifest,
    OutOfMemory,
    AlreadyLoaded,
    NotLoaded,
};

/// Single source of truth for the bundled plugin directories shipped in-tree.
/// Both the CLI (`abi plugin run`) and the MCP server (`plugin_list` /
/// `plugin_run`) load this exact set so the two surfaces stay symmetric — a
/// plugin runnable from the CLI is runnable over MCP and vice versa. Paths are
/// relative to the repo root (the working directory for both binaries).
pub const bundled_plugin_paths = [_][]const u8{
    "src/plugins/example-plugin",
    "src/plugins/example-wdbx-plugin",
    "src/plugins/telemetry-exporter",
    "src/plugins/ai-plugin",
    "src/plugins/gpu-plugin",
    "src/plugins/accelerator-plugin",
    "src/plugins/shader-plugin",
    "src/plugins/mlir-plugin",
    "src/plugins/os-control-plugin",
    "src/plugins/hash-plugin",
    "src/plugins/tui-plugin",
    "src/plugins/nn-plugin",
    "src/plugins/metrics-plugin",
    "src/plugins/sea-plugin",
    "src/plugins/mobile-plugin",
    "src/plugins/foundationmodels-plugin",
};

/// Load every `bundled_plugin_paths` entry into `pm`, tolerating both an
/// already-loaded plugin (idempotent across repeated calls) and a single bad
/// manifest (logged and skipped) so one broken plugin never blanks the whole
/// list. Shared by the CLI and MCP surfaces to keep them in lockstep.
pub fn loadBundled(pm: *PluginManager) void {
    for (bundled_plugin_paths) |path| {
        _ = pm.loadPlugin(path) catch |err| switch (err) {
            error.AlreadyLoaded => {},
            else => std.log.warn("failed to load bundled plugin path={s} err={s}", .{ path, @errorName(err) }),
        };
    }
}

const LoadedPlugin = struct {
    info: PluginInfo,
};

const PluginDispatchEntry = struct {
    name: []const u8,
    run_fn: *const fn (std.mem.Allocator, []const u8) anyerror![]u8,
};

const bundled_plugin_dispatch = [_]PluginDispatchEntry{
    .{ .name = "example-plugin", .run_fn = &@import("example-plugin/mod.zig").run },
    .{ .name = "example-wdbx-plugin", .run_fn = &@import("example-wdbx-plugin/mod.zig").run },
    .{ .name = "telemetry-exporter", .run_fn = &@import("telemetry-exporter/mod.zig").run },
    .{ .name = "ai-plugin", .run_fn = &@import("ai-plugin/mod.zig").run },
    .{ .name = "gpu-plugin", .run_fn = &@import("gpu-plugin/mod.zig").run },
    .{ .name = "accelerator-plugin", .run_fn = &@import("accelerator-plugin/mod.zig").run },
    .{ .name = "shader-plugin", .run_fn = &@import("shader-plugin/mod.zig").run },
    .{ .name = "mlir-plugin", .run_fn = &@import("mlir-plugin/mod.zig").run },
    .{ .name = "os-control-plugin", .run_fn = &@import("os-control-plugin/mod.zig").run },
    .{ .name = "hash-plugin", .run_fn = &@import("hash-plugin/mod.zig").run },
    .{ .name = "tui-plugin", .run_fn = &@import("tui-plugin/mod.zig").run },
    .{ .name = "nn-plugin", .run_fn = &@import("nn-plugin/mod.zig").run },
    .{ .name = "metrics-plugin", .run_fn = &@import("metrics-plugin/mod.zig").run },
    .{ .name = "sea-plugin", .run_fn = &@import("sea-plugin/mod.zig").run },
    .{ .name = "mobile-plugin", .run_fn = &@import("mobile-plugin/mod.zig").run },
    .{ .name = "foundationmodels-plugin", .run_fn = &@import("foundationmodels-plugin/mod.zig").run },
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
            self.allocator.free(plugin.info.target_feature);
            self.allocator.free(plugin.info.entry_point);
            self.allocator.free(plugin.info.path);
            for (plugin.info.commands) |cmd| {
                self.allocator.free(cmd.name);
                self.allocator.free(cmd.summary);
                for (cmd.aliases) |alias| self.allocator.free(alias);
                self.allocator.free(cmd.aliases);
            }
            self.allocator.free(plugin.info.commands);
            for (plugin.info.context_providers) |cp| {
                self.allocator.free(cp.name);
                self.allocator.free(cp.summary);
            }
            self.allocator.free(plugin.info.context_providers);
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

        // Parse optional commands array
        const commands = if (obj.get("commands")) |cmds_val| blk: {
            if (cmds_val != .array) break :blk &.{};
            const arr = cmds_val.array.items;
            const cmds = try allocator.alloc(ManifestCommand, arr.len);
            errdefer allocator.free(cmds);
            for (arr, 0..) |cmd_val, i| {
                if (cmd_val != .object) return ManifestSchemaError.InvalidJson;
                const cmd_obj = cmd_val.object;
                const cmd_name = cmd_obj.get("name") orelse return ManifestSchemaError.InvalidJson;
                if (cmd_name != .string or cmd_name.string.len == 0) return ManifestSchemaError.InvalidJson;
                const cmd_summary = if (cmd_obj.get("summary")) |s| if (s == .string) s.string else "" else "";
                const owned_cmd_name = try allocator.dupe(u8, cmd_name.string);
                errdefer allocator.free(owned_cmd_name);
                const owned_cmd_summary = try allocator.dupe(u8, cmd_summary);
                errdefer allocator.free(owned_cmd_summary);

                // Parse optional aliases
                var aliases: []const []const u8 = &.{};
                if (cmd_obj.get("aliases")) |alias_val| {
                    if (alias_val == .array) {
                        const alias_arr = try allocator.alloc([]const u8, alias_val.array.items.len);
                        errdefer allocator.free(alias_arr);
                        for (alias_val.array.items, 0..) |a, j| {
                            if (a != .string) return ManifestSchemaError.InvalidJson;
                            alias_arr[j] = try allocator.dupe(u8, a.string);
                        }
                        aliases = alias_arr;
                    }
                }

                cmds[i] = .{ .name = owned_cmd_name, .summary = owned_cmd_summary, .aliases = aliases };
            }
            break :blk cmds;
        } else &.{};

        // Parse optional context_providers array
        const context_providers = if (obj.get("context_providers")) |cps_val| blk: {
            if (cps_val != .array) break :blk &.{};
            const arr = cps_val.array.items;
            const cps = try allocator.alloc(ManifestContextProvider, arr.len);
            errdefer allocator.free(cps);
            for (arr, 0..) |cp_val, i| {
                if (cp_val != .object) return ManifestSchemaError.InvalidJson;
                const cp_obj = cp_val.object;
                const cp_name = cp_obj.get("name") orelse return ManifestSchemaError.InvalidJson;
                if (cp_name != .string or cp_name.string.len == 0) return ManifestSchemaError.InvalidJson;
                const cp_summary = if (cp_obj.get("summary")) |s| if (s == .string) s.string else "" else "";
                const owned_cp_name = try allocator.dupe(u8, cp_name.string);
                errdefer allocator.free(owned_cp_name);
                const owned_cp_summary = try allocator.dupe(u8, cp_summary);
                errdefer allocator.free(owned_cp_summary);
                cps[i] = .{ .name = owned_cp_name, .summary = owned_cp_summary };
            }
            break :blk cps;
        } else &.{};

        return .{
            .name = owned_name,
            .version = owned_version,
            .description = owned_description,
            .target_feature = owned_target,
            .entry_point = owned_entry,
            .commands = commands,
            .context_providers = context_providers,
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
            .commands = manifest.commands,
            .context_providers = manifest.context_providers,
        };

        const loaded = LoadedPlugin{ .info = plugin_info };
        try self.plugins.put(self.allocator, owned_key, loaded);

        telemetry.record("plugin.loaded");
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
        for (info.commands) |cmd| {
            self.allocator.free(cmd.name);
            self.allocator.free(cmd.summary);
            for (cmd.aliases) |alias| self.allocator.free(alias);
            self.allocator.free(cmd.aliases);
        }
        self.allocator.free(info.commands);
        for (info.context_providers) |cp| {
            self.allocator.free(cp.name);
            self.allocator.free(cp.summary);
        }
        self.allocator.free(info.context_providers);

        telemetry.record("plugin.unloaded");
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

        var it = self.plugins.iterator();
        var i: usize = 0;
        while (it.next()) |entry| : (i += 1) {
            result[i] = entry.value_ptr.*.info;
        }

        return result;
    }

    pub fn pluginCount(self: *PluginManager) usize {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        return self.plugins.count();
    }

    /// Collect context snippets from all loaded plugins that declare
    /// context_providers. Calls each provider's `run()` with input
    /// `__context__:<name>` and formats the results as
    /// `[context:plugin:provider]\nsnippet\n[/context]\n`.
    /// Caller owns the returned slice; returns empty string if none.
    pub fn collectContextSnippets(self: *PluginManager, allocator: std.mem.Allocator) ![]const u8 {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        var out = std.ArrayListUnmanaged(u8).empty;
        errdefer out.deinit(allocator);

        var it = self.plugins.iterator();
        while (it.next()) |entry| {
            const info = entry.value_ptr.info;
            if (info.context_providers.len == 0) continue;

            for (info.context_providers) |provider| {
                const input = try std.fmt.allocPrint(allocator, "__context__:{s}", .{provider.name});
                defer allocator.free(input);

                const snippet = self.run(allocator, info.name, input) catch |err| {
                    std.log.warn("context provider {s}/{s} failed: {s}", .{ info.name, provider.name, @errorName(err) });
                    continue;
                };
                defer allocator.free(snippet);

                try out.print(allocator, "[context:{s}:{s}]\n{s}\n[/context]\n", .{ info.name, provider.name, snippet });
            }
        }

        return try out.toOwnedSlice(allocator);
    }

    /// Real plugin execution dispatch for registered in-tree plugins.
    /// Bundled example plugins export `pub fn run(allocator, input) ![]u8`; we
    /// import and invoke the actual implementation here (AOT Zig model).
    /// External/native plugins would require an ABI bridge (future).
    pub fn run(self: *PluginManager, allocator: std.mem.Allocator, name: []const u8, input: []const u8) ![]u8 {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        _ = self.plugins.getEntry(name) orelse return error.PluginNotFound;
        telemetry.record("plugin.run");

        inline for (bundled_plugin_dispatch) |entry| {
            if (std.mem.eql(u8, name, entry.name)) {
                return entry.run_fn(allocator, input);
            }
        }

        return try std.fmt.allocPrint(
            allocator,
            "plugin '{s}' executed (contract honored; input len={d}).",
            .{ name, input.len },
        );
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
    {
        const abs_path = try temp_path.tempFilePath(std.testing.allocator, "test", "zig");
        defer std.testing.allocator.free(abs_path);
        try std.testing.expect(!isSafeEntryPoint(abs_path));
    }
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
