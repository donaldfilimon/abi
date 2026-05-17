const std = @import("std");

pub fn validatePluginStructure(plugin_path: []const u8) !bool {
    var dir = try std.fs.openDirAbsolute(plugin_path, .{});
    defer dir.close();

    _ = dir.statFile("mod.zig") catch return false;
    _ = dir.statFile("stub.zig") catch return false;

    var manifest_file = dir.openFile("abi-plugin.json", .{}) catch return false;
    defer manifest_file.close();

    const allocator = std.heap.page_allocator;
    const manifest_json = manifest_file.readToEndAlloc(allocator, 64 * 1024) catch return false;
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
    _ = dir.statFile(entry_point) catch return false;

    return true;
}

fn jsonStringField(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}
