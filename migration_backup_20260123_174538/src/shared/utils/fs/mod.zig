const std = @import("std");

pub fn join(allocator: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    if (parts.len == 0) return allocator.alloc(u8, 0);
    var total: usize = 0;
    for (parts) |part| {
        total += part.len;
    }
    total += parts.len - 1;

    const output = try allocator.alloc(u8, total);
    var index: usize = 0;
    for (parts, 0..) |part, i| {
        std.mem.copyForwards(u8, output[index..][0..part.len], part);
        index += part.len;
        if (i + 1 < parts.len) {
            output[index] = std.fs.path.sep;
            index += 1;
        }
    }
    return output;
}

pub fn hasExtension(path: []const u8, extension: []const u8) bool {
    if (extension.len == 0) return false;
    return std.mem.endsWith(u8, path, extension);
}

pub fn basename(path: []const u8) []const u8 {
    return std.fs.path.basename(path);
}

pub const PathValidationError = error{
    InvalidPath,
    PathTraversalAttempt,
    AbsolutePathRejected,
    InvalidCharacter,
};

pub fn isSafeBackupPath(path: []const u8) bool {
    const filename = std.fs.path.basename(path);

    if (filename.len == 0) return false;
    if (filename.len != path.len) return false;

    for (filename) |char| {
        if (char == 0) return false;
    }

    if (std.mem.indexOfScalar(u8, filename, ':') != null) return false;
    if (std.mem.startsWith(u8, filename, "\\\\")) return false;

    if (std.mem.indexOf(u8, filename, "..") != null) return false;

    return true;
}

pub fn normalizeBackupPath(allocator: std.mem.Allocator, user_path: []const u8) ![]u8 {
    if (!isSafeBackupPath(user_path)) return PathValidationError.InvalidPath;

    const filename = std.fs.path.basename(user_path);

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Create backups directory if it doesn't exist
    std.Io.Dir.cwd().createDir(io, "backups", .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    // Return the path under backups/ directory
    return try std.fs.path.join(allocator, &.{ "backups", filename });
}

test "fs helpers" {
    const allocator = std.testing.allocator;
    const expected = try std.fmt.allocPrint(
        allocator,
        "a{c}b{c}c",
        .{ std.fs.path.sep, std.fs.path.sep },
    );
    defer allocator.free(expected);

    const joined = try join(allocator, &.{ "a", "b", "c" });
    defer allocator.free(joined);
    try std.testing.expectEqualStrings(expected, joined);
    try std.testing.expect(hasExtension("file.zig", ".zig"));
    const path = try std.fmt.allocPrint(allocator, "dir{c}file.zig", .{std.fs.path.sep});
    defer allocator.free(path);
    try std.testing.expectEqualStrings("file.zig", basename(path));
}

test "path validation rejects traversal attempts" {
    try std.testing.expect(!isSafeBackupPath("../etc/passwd"));
    try std.testing.expect(!isSafeBackupPath("..\\..\\windows\\system32\\config"));
    try std.testing.expect(!isSafeBackupPath("/etc/passwd"));
    try std.testing.expect(!isSafeBackupPath("C:\\windows\\system32"));
    try std.testing.expect(!isSafeBackupPath("test/../file.bin"));
    try std.testing.expect(!isSafeBackupPath(""));

    try std.testing.expect(isSafeBackupPath("test.bin"));
    try std.testing.expect(isSafeBackupPath("database.db"));
    try std.testing.expect(isSafeBackupPath("backup_2024.db"));
}

test "path validation rejects absolute paths" {
    try std.testing.expect(!isSafeBackupPath("/var/data/db.bin"));
    try std.testing.expect(!isSafeBackupPath("/usr/local/backup"));
    try std.testing.expect(!isSafeBackupPath("C:\\Program Files\\backup.db"));
}

test "normalizeBackupPath rejects invalid paths" {
    try std.testing.expectError(PathValidationError.InvalidPath, normalizeBackupPath(std.testing.allocator, "../etc/passwd"));
    try std.testing.expectError(PathValidationError.InvalidPath, normalizeBackupPath(std.testing.allocator, "/absolute/path"));
}
