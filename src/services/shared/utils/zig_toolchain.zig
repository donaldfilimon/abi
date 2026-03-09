const std = @import("std");

pub fn allocZvmMasterZigPath(allocator: std.mem.Allocator) !?[]u8 {
    const home_ptr = std.c.getenv("HOME") orelse std.c.getenv("USERPROFILE") orelse return null;
    const home = std.mem.span(home_ptr);
    return try std.fs.path.join(allocator, &.{ home, ".zvm", "master", "zig" });
}

pub fn resolveExistingZvmMasterZigPath(allocator: std.mem.Allocator, io: std.Io) !?[]u8 {
    // First, check repo-local CEL toolchain
    const cwd = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd);

    // Look for .cel/bin/zig in the current workspace
    const cel_path = try std.fs.path.join(allocator, &.{ cwd, ".cel", "bin", "zig" });
    const cel_file = std.Io.Dir.openFileAbsolute(io, cel_path, .{}) catch null;
    if (cel_file != null) {
        cel_file.?.close(io);
        return cel_path;
    }
    allocator.free(cel_path);

    // Fall back to ZVM master
    const path = try allocZvmMasterZigPath(allocator) orelse return null;

    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch {
        allocator.free(path);
        return null;
    };
    file.close(io);

    return path;
}
