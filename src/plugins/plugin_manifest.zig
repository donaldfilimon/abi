const std = @import("std");
const plugin_validator = @import("../foundation/plugin_validator.zig");
const temp_path = @import("../foundation/temp_path.zig");

pub const isSafeEntryPoint = plugin_validator.isSafeEntryPoint;

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

test {
    std.testing.refAllDecls(@This());
}

test "plugin manager validates correct manifest" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    const parsed = try validatePlugin(std.testing.allocator, manifest_json);
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
    const parsed = try validatePlugin(std.testing.allocator, manifest_json);
    defer parsed.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("ai", parsed.target_feature);
    try std.testing.expectEqualStrings("nested/mod.zig", parsed.entry_point);
}

test "plugin manager rejects invalid manifest shapes" {
    try std.testing.expectError(ManifestSchemaError.InvalidJson, validatePlugin(std.testing.allocator, "not json"));
    try std.testing.expectError(ManifestSchemaError.InvalidJson, validatePlugin(std.testing.allocator, "[]"));
    try std.testing.expectError(ManifestSchemaError.MissingName, validatePlugin(std.testing.allocator,
        \\{"name": 123, "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.InvalidVersion, validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingDescription, validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "", "target_feature": "ai", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingTargetFeature, validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "", "entry_point": "mod.zig"}
    ));
    try std.testing.expectError(ManifestSchemaError.MissingEntryPoint, validatePlugin(std.testing.allocator,
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": ""}
    ));
}

test "plugin manager rejects manifest missing description" {
    const manifest_json =
        \\{"name": "test-plugin", "version": "1.0.0", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingDescription, validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing name" {
    const manifest_json =
        \\{"version": "1.0.0", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingName, validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing version" {
    const manifest_json =
        \\{"name": "test", "description": "A test", "target_feature": "ai", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingVersion, validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing entry_point" {
    const manifest_json =
        \\{"name": "test", "version": "1.0.0", "description": "A test", "target_feature": "ai"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingEntryPoint, validatePlugin(std.testing.allocator, manifest_json));
}

test "plugin manager rejects manifest missing target_feature" {
    const manifest_json =
        \\{"name": "test", "version": "1.0.0", "description": "A test", "entry_point": "mod.zig"}
    ;
    try std.testing.expectError(ManifestSchemaError.MissingTargetFeature, validatePlugin(std.testing.allocator, manifest_json));
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
    try std.testing.expectError(ManifestSchemaError.InvalidEntryPoint, validatePlugin(std.testing.allocator, manifest_json));
}
