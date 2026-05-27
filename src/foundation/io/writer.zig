const std = @import("std");
const io_stats = @import("stats.zig");

pub const IOStats = io_stats.IOStats;

pub const BufferedWriter = struct {
    file: std.Io.File,
    allocator: std.mem.Allocator = std.heap.page_allocator,
    buffer: std.ArrayListUnmanaged(u8) = std.ArrayListUnmanaged(u8).empty,
    pos: usize = 0,
    offset: u64 = 0,
    stats: ?*IOStats = null,

    pub fn init(file: std.Io.File, buffer_size: usize) !BufferedWriter {
        return initWithAllocator(std.heap.page_allocator, file, buffer_size);
    }

    pub fn initWithAllocator(allocator: std.mem.Allocator, file: std.Io.File, buffer_size: usize) !BufferedWriter {
        var writer = BufferedWriter{
            .file = file,
            .allocator = allocator,
            .pos = 0,
        };
        try writer.buffer.resize(allocator, buffer_size);
        return writer;
    }

    pub fn deinit(self: *BufferedWriter, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.buffer.deinit(self.allocator);
    }

    pub fn flush(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writePositionalAll(defaultIo(), self.buffer.items[0..self.pos], self.offset);
            self.offset += self.pos;
            if (self.stats) |s| {
                s.recordWrite(@intCast(self.pos));
            }
            self.pos = 0;
        }
    }

    pub fn write(self: *BufferedWriter, data: []const u8) !usize {
        var remaining = data;
        while (remaining.len > 0) {
            const available = self.buffer.items.len - self.pos;
            if (available == 0) {
                try self.flush();
                continue;
            }
            const to_copy = @min(available, remaining.len);
            @memcpy(self.buffer.items[self.pos .. self.pos + to_copy], remaining[0..to_copy]);
            self.pos += to_copy;
            remaining = remaining[to_copy..];
        }
        return data.len;
    }

    pub fn writeAll(self: *BufferedWriter, data: []const u8) !void {
        _ = try self.write(data);
    }

    pub fn writeByte(self: *BufferedWriter, b: u8) !void {
        if (self.pos >= self.buffer.items.len) {
            try self.flush();
        }
        self.buffer.items[self.pos] = b;
        self.pos += 1;
    }

    pub fn writeByteNTimes(self: *BufferedWriter, b: u8, n: usize) !void {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            try self.writeByte(b);
        }
    }
};

fn defaultIo() std.Io {
    return std.Options.debug_io;
}

fn testPath(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "/tmp/{s}_{d}.txt", .{ name, std.c.getpid() });
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

test "BufferedWriter basic write" {
    const test_path = try testPath(std.testing.allocator, "abi_io_buffered_write_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
    defer writer.deinit(std.testing.allocator);

    try writer.writeAll("hello buffered writer");
    try writer.flush();

    const f = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer f.close(defaultIo());
    const stat = try f.stat(defaultIo());
    const read_content = try std.testing.allocator.alloc(u8, @intCast(stat.size));
    defer std.testing.allocator.free(read_content);
    _ = try f.readPositionalAll(defaultIo(), read_content, 0);

    try std.testing.expectEqualStrings("hello buffered writer", read_content);
}

test "BufferedWriter writeByte" {
    const test_path = try testPath(std.testing.allocator, "abi_io_writebyte_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
    defer writer.deinit(std.testing.allocator);

    try writer.writeByte('X');
    try writer.writeByte('Y');
    try writer.writeByte('Z');
    try writer.flush();

    const f = try std.Io.Dir.openFileAbsolute(defaultIo(), test_path, .{});
    defer f.close(defaultIo());
    const stat = try f.stat(defaultIo());
    const read_content = try std.testing.allocator.alloc(u8, @intCast(stat.size));
    defer std.testing.allocator.free(read_content);
    _ = try f.readPositionalAll(defaultIo(), read_content, 0);

    try std.testing.expectEqualStrings("XYZ", read_content);
}

test "BufferedWriter with stats" {
    const test_path = try testPath(std.testing.allocator, "abi_io_bufwriter_stats_test");
    defer std.testing.allocator.free(test_path);
    defer deleteFileForTest(test_path);

    var iostats = IOStats{};

    const file = try std.Io.Dir.createFileAbsolute(defaultIo(), test_path, .{ .truncate = true });
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
    defer writer.deinit(std.testing.allocator);
    writer.stats = &iostats;

    try writer.writeAll("stats buffered write");
    try writer.flush();

    const snap = iostats.snapshot();
    try std.testing.expect(snap.bytes_written > 0);
    try std.testing.expect(snap.write_ops > 0);
}
