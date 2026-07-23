const std = @import("std");
const io_stats = @import("stats.zig");
const temp_path = @import("../temp_path.zig");

pub const IOStats = io_stats.IOStats;

pub const BufferedReader = struct {
    file: std.Io.File,
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8) = std.ArrayListUnmanaged(u8).empty,
    pos: usize = 0,
    end: usize = 0,
    offset: u64 = 0,
    stats: ?*IOStats = null,

    pub fn init(file: std.Io.File, buffer_size: usize) !BufferedReader {
        return initWithAllocator(std.heap.page_allocator, file, buffer_size);
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator, file: std.Io.File, buffer_size: usize) !BufferedReader {
        var reader = BufferedReader{
            .file = file,
            .allocator = allocator,
            .pos = 0,
            .end = 0,
        };
        try reader.buffer.resize(allocator, buffer_size);
        return reader;
    }

    pub fn deinit(self: *BufferedReader, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.buffer.deinit(self.allocator);
    }

    pub fn fill(self: *BufferedReader) !usize {
        self.pos = 0;
        const bytes_read = try self.file.readPositionalAll(defaultIo(), self.buffer.items, self.offset);
        self.offset += bytes_read;
        self.end = bytes_read;
        if (self.stats) |s| {
            s.recordRead(@intCast(bytes_read));
        }
        return bytes_read;
    }

    pub fn read(self: *BufferedReader, out: []u8) !usize {
        if (self.pos >= self.end) {
            const filled = try self.fill();
            if (filled == 0) return 0;
        }
        const avail = self.end - self.pos;
        const to_copy = @min(avail, out.len);
        @memcpy(out[0..to_copy], self.buffer.items[self.pos .. self.pos + to_copy]);
        self.pos += to_copy;
        return to_copy;
    }

    pub fn readAll(self: *BufferedReader, out: []u8) !usize {
        var total: usize = 0;
        while (total < out.len) {
            const n = try self.read(out[total..]);
            if (n == 0) break;
            total += n;
        }
        return total;
    }

    pub fn readByte(self: *BufferedReader) !u8 {
        if (self.pos >= self.end) {
            _ = try self.fill();
            if (self.pos >= self.end) return error.EndOfStream;
        }
        const b = self.buffer.items[self.pos];
        self.pos += 1;
        return b;
    }

    pub fn readUntilDelimiter(self: *BufferedReader, allocator: std.mem.Allocator, delimiter: u8) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);
        while (true) {
            const b = try self.readByte();
            if (b == delimiter) break;
            try result.append(allocator, b);
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn isEmpty(self: *BufferedReader) bool {
        return self.pos >= self.end;
    }

    pub fn available(self: *BufferedReader) usize {
        return self.end - self.pos;
    }
};

fn defaultIo() std.Io {
    return std.Options.debug_io;
}

fn testPath(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return try temp_path.tempFilePath(allocator, name, "txt");
}

fn deleteFileForTest(path: []const u8) void {
    std.Io.Dir.deleteFileAbsolute(defaultIo(), path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.log.warn("test cleanup failed: {s}", .{@errorName(err)}),
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "BufferedReader basic read" {
    const test_path = try testPath(std.testing.allocator, "abi_io_buffered_read_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    {
        const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
        defer file.close(defaultIo());
        try file.writeStreamingAll(defaultIo(), "hello buffered reader");
    }

    const file = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
    defer reader.deinit(std.testing.allocator);

    var buf: [64]u8 = undefined;
    const n = try reader.readAll(&buf);
    try std.testing.expectEqualStrings("hello buffered reader", buf[0..n]);
}

test "BufferedReader readByte" {
    const test_path = try testPath(std.testing.allocator, "abi_io_readbyte_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    {
        const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
        defer file.close(defaultIo());
        try file.writeStreamingAll(defaultIo(), "AB");
    }

    const file = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
    defer reader.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u8, 'A'), try reader.readByte());
    try std.testing.expectEqual(@as(u8, 'B'), try reader.readByte());
}

test "BufferedReader with stats" {
    const test_path = try testPath(std.testing.allocator, "abi_io_bufreader_stats_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    {
        const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
        defer file.close(defaultIo());
        try file.writeStreamingAll(defaultIo(), "stats buffered read");
    }

    var iostats = IOStats{};

    const file = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
    defer reader.deinit(std.testing.allocator);
    reader.stats = &iostats;

    var buf: [64]u8 = undefined;
    _ = try reader.readAll(&buf);

    const snap = iostats.snapshot();
    try std.testing.expect(snap.bytes_read > 0);
    try std.testing.expect(snap.read_ops > 0);
}
