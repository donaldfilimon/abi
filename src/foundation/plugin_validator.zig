const std = @import("std");

pub fn validatePluginStructure(plugin_path: []const u8) !bool {
    var dir = try std.fs.openDirAbsolute(plugin_path, .{});
    defer dir.close();

    _ = dir.statFile("mod.zig") catch return false;
    _ = dir.statFile("stub.zig") catch return false;
    _ = dir.statFile("abi-plugin.json") catch return false;

    return true;
}
