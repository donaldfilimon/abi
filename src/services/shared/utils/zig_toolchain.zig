const std = @import("std");
const builtin = @import("builtin");

pub fn zigBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zig.exe" else "zig";
}

pub fn zlsBinaryName() []const u8 {
    return if (builtin.os.tag == .windows) "zls.exe" else "zls";
}

pub fn allocZvmMasterZigPathFromHome(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "master", zigBinaryName() });
}

pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator) !?[]u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
    const home = std.mem.span(home_ptr);
    return try allocZvmMasterZigPathFromHome(allocator, home);
}

pub fn resolveExistingZvmMasterZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const path = try allocZvmMasterZigPath(allocator) orelse return null;
    if (!fileExistsAbsolute(io, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

pub fn resolveExistingPreferredZigPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    start_path: []const u8,
) !?[]u8 {
    _ = start_path;
    return resolveExistingZvmMasterZigPath(allocator, io);
}

fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

test {
    std.testing.refAllDecls(@This());
}
