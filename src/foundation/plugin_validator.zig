const std = @import("std");

pub fn validatePluginStructure(allocator: std.mem.Allocator, plugin_path: []const u8) !bool {
    const io = std.Options.debug_io;
    var dir = try std.Io.Dir.openDirAbsolute(io, plugin_path, .{});
    defer dir.close(io);

    _ = dir.statFile(io, "mod.zig", .{}) catch return false;
    _ = dir.statFile(io, "stub.zig", .{}) catch return false;

    const manifest_json = std.Io.Dir.readFileAlloc(dir, io, "abi-plugin.json", allocator, .limited(64 * 1024)) catch return false;
    defer allocator.free(manifest_json);

    const manifest = std.json.parseFromSlice(std.json.Value, allocator, manifest_json, .{}) catch return false;
    defer manifest.deinit();

    const obj = switch (manifest.value) {
        .object => |object| object,
        else => return false,
    };

    const name = jsonStringField(obj, "name") orelse return false;
    const version = jsonStringField(obj, "version") orelse return false;
    const entry_point = jsonStringField(obj, "entry_point") orelse jsonStringField(obj, "entryPoint") orelse return false;
    const target_feature = jsonStringField(obj, "target_feature") orelse jsonStringField(obj, "targetFeature") orelse return false;

    if (name.len == 0 or version.len == 0 or entry_point.len == 0 or target_feature.len == 0) return false;
    _ = dir.statFile(io, entry_point, .{}) catch return false;

    return true;
}

test {
    std.testing.refAllDecls(@This());
}

test "validatePluginStructure rejects missing directory" {
    try std.testing.expectError(error.FileNotFound, validatePluginStructure(std.testing.allocator, "/tmp/nonexistent_plugin_dir_abc123"));
}

fn jsonStringField(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}
