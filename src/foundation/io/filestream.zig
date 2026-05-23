const std = @import("std");
const io_stats = @import("stats.zig");

pub const IOStats = io_stats.IOStats;

pub const FileStream = struct {
    file: ?std.Io.File = null,
    path: []const u8,
    mode: Mode,
    offset: u64 = 0,
    stats: ?*IOStats = null,

    pub const Mode = enum {
        read,
        write,
        append,
        read_write,
    };

    pub fn open(path: []const u8, mode: Mode) !FileStream {
        const io_context = defaultIo();
        var initial_offset: u64 = 0;
        const file = switch (mode) {
            .read => try std.Io.Dir.openFileAbsolute(io_context, path, .{}),
            .write => try std.Io.Dir.createFileAbsolute(io_context, path, .{ .truncate = true }),
            .append => blk: {
                const opened = std.Io.Dir.openFileAbsolute(io_context, path, .{ .mode = .read_write }) catch |err| switch (err) {
                    error.FileNotFound => try std.Io.Dir.createFileAbsolute(io_context, path, .{ .read = true }),
                    else => |e| return e,
                };
                initial_offset = (try opened.stat(io_context)).size;
                break :blk opened;
            },
            .read_write => try std.Io.Dir.createFileAbsolute(io_context, path, .{ .read = true }),
        };
        return FileStream{
            .file = file,
            .path = path,
            .mode = mode,
            .offset = initial_offset,
        };
    }

    pub fn deinit(self: *FileStream) void {
        if (self.file) |f| {
            f.close(defaultIo());
            self.file = null;
        }
    }

    pub fn read(self: *FileStream, buf: []u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.readPositional(defaultIo(), &.{buf}, self.offset);
        self.offset += n;
        if (self.stats) |s| {
            s.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn readAll(self: *FileStream, buf: []u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.readPositionalAll(defaultIo(), buf, self.offset);
        self.offset += n;
        if (self.stats) |s| {
            s.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn write(self: *FileStream, data: []const u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.writePositional(defaultIo(), &.{data}, self.offset);
        self.offset += n;
        if (self.stats) |s| {
            s.recordWrite(@intCast(n));
        }
        return n;
    }

    pub fn writeAll(self: *FileStream, data: []const u8) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.writePositionalAll(defaultIo(), data, self.offset);
        self.offset += data.len;
        if (self.stats) |s| {
            s.recordWrite(@intCast(data.len));
        }
    }

    pub fn seekTo(self: *FileStream, offset: u64) !void {
        _ = self.file orelse return error.FileNotOpen;
        self.offset = offset;
    }

    pub fn seekBy(self: *FileStream, offset: i64) !void {
        _ = self.file orelse return error.FileNotOpen;
        if (offset < 0) {
            self.offset -= @intCast(-offset);
        } else {
            self.offset += @intCast(offset);
        }
    }

    pub fn getPos(self: *FileStream) !u64 {
        _ = self.file orelse return error.FileNotOpen;
        return self.offset;
    }

    pub fn getEndPos(self: *FileStream) !u64 {
        const f = self.file orelse return error.FileNotOpen;
        return (try f.stat(defaultIo())).size;
    }

    pub fn stat(self: *FileStream) !std.Io.File.Stat {
        const f = self.file orelse return error.FileNotOpen;
        return try f.stat(defaultIo());
    }

    pub fn sync(self: *FileStream) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.sync(defaultIo());
    }

    pub fn isOpen(self: *FileStream) bool {
        return self.file != null;
    }
};

fn defaultIo() std.Io {
    return std.Options.debug_io;
}

test {
    std.testing.refAllDecls(@This());
}

test "FileStream read and seek" {
    const test_path = "/tmp/abi_io_filestream_test.txt";
    defer std.Io.Dir.deleteFileAbsolute(defaultIo(), test_path) catch |err| std.log.warn("test cleanup failed: {s}", .{@errorName(err)});

    {
        const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
        defer file.close(defaultIo());
        try file.writeStreamingAll(defaultIo(), "0123456789");
    }

    var stream = try FileStream.open(test_path, .read);
    defer stream.deinit();

    var buf: [10]u8 = undefined;
    const n = try stream.readAll(&buf);
    try std.testing.expectEqual(@as(usize, 10), n);
    try std.testing.expectEqualStrings("0123456789", buf[0..n]);

    try stream.seekTo(5);
    const pos = try stream.getPos();
    try std.testing.expectEqual(@as(u64, 5), pos);

    const n2 = try stream.read(buf[0..3]);
    try std.testing.expectEqual(@as(usize, 3), n2);
    try std.testing.expectEqualStrings("567", buf[0..3]);
}

test "FileStream write mode" {
    const test_path = "/tmp/abi_io_filestream_write_test.txt";
    defer std.Io.Dir.deleteFileAbsolute(defaultIo(), test_path) catch |err| std.log.warn("test cleanup failed: {s}", .{@errorName(err)});

    {
        var stream = try FileStream.open(test_path, .write);
        defer stream.deinit();

        try stream.writeAll("stream data");
        try stream.sync();
    }

    const f = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer f.close(defaultIo());
    const stat = try f.stat(defaultIo());
    const content = try std.testing.allocator.alloc(u8, @intCast(stat.size));
    defer std.testing.allocator.free(content);
    _ = try f.readPositionalAll(defaultIo(), content, 0);

    try std.testing.expectEqualStrings("stream data", content);
}

test "FileStream with stats" {
    const test_path = "/tmp/abi_io_filestream_stats_test.txt";
    defer std.Io.Dir.deleteFileAbsolute(defaultIo(), test_path) catch |err| std.log.warn("test cleanup failed: {s}", .{@errorName(err)});

    var iostats = IOStats{};

    {
        var stream = try FileStream.open(test_path, .write);
        defer stream.deinit();
        stream.stats = &iostats;

        try stream.writeAll("stats test");
    }

    {
        var stream = try FileStream.open(test_path, .read);
        defer stream.deinit();
        stream.stats = &iostats;

        var buf: [32]u8 = undefined;
        _ = try stream.readAll(&buf);
    }

    const snap = iostats.snapshot();
    try std.testing.expect(snap.bytes_written > 0);
    try std.testing.expect(snap.bytes_read > 0);
    try std.testing.expect(snap.write_ops > 0);
    try std.testing.expect(snap.read_ops > 0);
}
