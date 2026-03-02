const std = @import("std");

pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator) !?[]u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
    const home = std.mem.span(home_ptr);
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "master", "zig" });
}

pub fn resolveExistingZvmMasterZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    const path = try allocZvmMasterZigPath(allocator) orelse return null;

    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch {
        allocator.free(path);
        return null;
    };
    file.close(io);

    return path;
}
