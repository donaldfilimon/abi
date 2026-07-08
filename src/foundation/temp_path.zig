const std = @import("std");
const builtin = @import("builtin");
const env = @import("env.zig");

/// Returns the platform-appropriate temporary directory path.
/// Uses $TMPDIR (Unix), then $TEMP/$TMP (cross-platform), then fallback.
/// Caller owns the returned memory.
pub fn getTempDir(allocator: std.mem.Allocator) ![]const u8 {
    if (builtin.target.os.tag != .windows) {
        if (env.get("TMPDIR")) |val| return try allocator.dupe(u8, val);
    }
    if (env.get("TEMP")) |val| return try allocator.dupe(u8, val);
    if (env.get("TMP")) |val| return try allocator.dupe(u8, val);
    return switch (builtin.target.os.tag) {
        .windows => try allocator.dupe(u8, "\\Windows\\Temp"),
        else => try allocator.dupe(u8, "/tmp"),
    };
}

/// Creates a unique temp file path in the platform temp directory.
/// Format: <tempDir>/<prefix>_<pid>.<extension>
/// Caller owns the returned memory.
pub fn tempFilePath(allocator: std.mem.Allocator, prefix: []const u8, extension: []const u8) ![]u8 {
    const dir = try getTempDir(allocator);
    defer allocator.free(dir);
    return std.fmt.allocPrint(allocator, "{s}/{s}_{d}.{s}", .{ dir, prefix, std.c.getpid(), extension });
}

test {
    std.testing.refAllDecls(@This());
}

test "getTempDir returns a directory that exists" {
    const dir = try getTempDir(std.testing.allocator);
    defer std.testing.allocator.free(dir);
    try std.testing.expect(dir.len > 0);
    _ = try std.Io.Dir.statFile(.cwd(), std.testing.io, dir, .{});
}

test "tempFilePath generates a valid path" {
    const path = try tempFilePath(std.testing.allocator, "test_prefix", "txt");
    defer std.testing.allocator.free(path);
    try std.testing.expect(path.len > 0);
    try std.testing.expect(std.mem.endsWith(u8, path, ".txt"));
}
