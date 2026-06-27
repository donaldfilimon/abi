const std = @import("std");

pub const stats = @import("stats.zig");
pub const reader = @import("reader.zig");
pub const writer = @import("writer.zig");
pub const filestream = @import("filestream.zig");

// Convenience accessors: re-export top-level names so existing callers work unchanged.
pub const IOStats = stats.IOStats;
pub const IOStatsSnapshot = stats.IOStatsSnapshot;
pub const BufferedReader = reader.BufferedReader;
pub const BufferedWriter = writer.BufferedWriter;
pub const FileStream = filestream.FileStream;

pub const DEFAULT_BUFFER_SIZE = 4096;
pub const MAX_PATH_LEN = std.Io.Dir.max_path_bytes;

fn defaultIo() std.Io {
    return std.Options.debug_io;
}

pub fn asyncReadFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io_context = defaultIo();
    const file = try std.Io.Dir.openFileAbsolute(io_context, path, .{});
    defer file.close(io_context);

    const stat = try file.stat(io_context);
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(buf);

    const bytes_read = try file.readPositionalAll(io_context, buf, 0);
    if (bytes_read < stat.size) {
        allocator.free(buf);
        return error.UnexpectedEOF;
    }

    return buf;
}

pub fn asyncWriteFile(path: []const u8, data: []const u8) !void {
    const io_context = defaultIo();
    const file = try std.Io.Dir.createFileAbsolute(io_context, path, .{ .truncate = true });
    defer file.close(io_context);

    try file.writeStreamingAll(io_context, data);
}

pub fn asyncAppendFile(path: []const u8, data: []const u8) !void {
    const io_context = defaultIo();
    const file = std.Io.Dir.openFileAbsolute(io_context, path, .{ .mode = .read_write }) catch |err| switch (err) {
        error.FileNotFound => try std.Io.Dir.createFileAbsolute(io_context, path, .{ .read = true }),
        else => |e| return e,
    };
    defer file.close(io_context);

    const stat = try file.stat(io_context);
    try file.writePositionalAll(io_context, data, stat.size);
}

pub fn resolvePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.Io.Dir.path.isAbsolute(path)) {
        return try allocator.dupe(u8, path);
    }

    var buf: [MAX_PATH_LEN]u8 = undefined;
    const len = try std.Io.Dir.realPathFile(.cwd(), defaultIo(), path, &buf);
    return try allocator.dupe(u8, buf[0..len]);
}

pub fn ensureDir(path: []const u8) !void {
    try std.Io.Dir.createDirPath(.cwd(), defaultIo(), path);
}

pub fn fileExists(path: []const u8) bool {
    std.Io.Dir.accessAbsolute(defaultIo(), path, .{ .read = true }) catch return false;
    return true;
}

pub fn dirExists(path: []const u8) bool {
    const stat = std.Io.Dir.statFile(.cwd(), defaultIo(), path, .{}) catch return false;
    return stat.kind == .directory;
}

fn deleteFileForTest(path: []const u8) void {
    std.Io.Dir.deleteFileAbsolute(defaultIo(), path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.log.warn("test cleanup failed: {s}", .{@errorName(err)}),
    };
}

fn deleteTreeForTest(path: []const u8) void {
    std.Io.Dir.deleteTree(.cwd(), defaultIo(), path) catch |err| std.log.warn("test cleanup failed: {s}", .{@errorName(err)});
}

test {
    std.testing.refAllDecls(@This());
}

test "asyncWriteFile and asyncReadFile roundtrip" {
    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_io_async_test_{d}.txt", .{std.c.getpid()});
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    const content = "hello async io world";
    try asyncWriteFile(test_path, content);

    const read_content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(read_content);

    try std.testing.expectEqualStrings(content, read_content);
}

test "asyncAppendFile" {
    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_io_append_test_{d}.txt", .{std.c.getpid()});
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    try asyncWriteFile(test_path, "line1\n");
    try asyncAppendFile(test_path, "line2\n");

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("line1\nline2\n", content);
}

test "asyncAppendFile creates missing file" {
    const test_path = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_io_append_create_test_{d}.txt", .{std.c.getpid()});
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    try asyncAppendFile(test_path, "created\n");

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("created\n", content);
}

test "fileExists and dirExists" {
    try std.testing.expect(fileExists("/tmp"));
    try std.testing.expect(dirExists("/tmp"));
    try std.testing.expect(!fileExists("/tmp/nonexistent_file_12345"));
    try std.testing.expect(!dirExists("/tmp/nonexistent_dir_12345"));
}

test "ensureDir creates nested directories" {
    const test_root = try std.fmt.allocPrint(std.testing.allocator, "/tmp/abi_io_test_nested_{d}", .{std.c.getpid()});
    defer std.testing.allocator.free(test_root);
    defer deleteTreeForTest(test_root);

    const test_dir = try std.fs.path.join(std.testing.allocator, &.{ test_root, "a/b/c" });
    defer std.testing.allocator.free(test_dir);

    try ensureDir(test_dir);
    try std.testing.expect(dirExists(test_dir));
}

test "resolvePath absolute path" {
    const abs = "/tmp/test";
    const resolved = try resolvePath(std.testing.allocator, abs);
    defer std.testing.allocator.free(resolved);

    try std.testing.expectEqualStrings(abs, resolved);
}
