const std = @import("std");
const sync = @import("sync.zig");
const time = @import("time.zig");

pub const DEFAULT_BUFFER_SIZE = 4096;
pub const MAX_PATH_LEN = std.fs.max_path_bytes;

pub const IOStats = struct {
    bytes_read: u64 = 0,
    bytes_written: u64 = 0,
    read_ops: u64 = 0,
    write_ops: u64 = 0,
    errors: u64 = 0,
    last_activity_ms: i64 = 0,
    lock: sync.SpinLock = sync.SpinLock{},

    pub fn recordRead(self: *IOStats, bytes: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_read += bytes;
        self.read_ops += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn recordWrite(self: *IOStats, bytes: u64) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_written += bytes;
        self.write_ops += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn recordError(self: *IOStats) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.errors += 1;
        self.last_activity_ms = time.unixMs();
    }

    pub fn snapshot(self: *IOStats) IOStatsSnapshot {
        self.lock.lock();
        defer self.lock.unlock();
        return IOStatsSnapshot{
            .bytes_read = self.bytes_read,
            .bytes_written = self.bytes_written,
            .read_ops = self.read_ops,
            .write_ops = self.write_ops,
            .errors = self.errors,
            .last_activity_ms = self.last_activity_ms,
        };
    }

    pub fn reset(self: *IOStats) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.bytes_read = 0;
        self.bytes_written = 0;
        self.read_ops = 0;
        self.write_ops = 0;
        self.errors = 0;
        self.last_activity_ms = 0;
    }
};

pub const IOStatsSnapshot = struct {
    bytes_read: u64,
    bytes_written: u64,
    read_ops: u64,
    write_ops: u64,
    errors: u64,
    last_activity_ms: i64,
};

pub const BufferedReader = struct {
    file: std.fs.File,
    buffer: std.ArrayListUnmanaged(u8) = std.ArrayListUnmanaged(u8).empty,
    pos: usize = 0,
    end: usize = 0,
    stats: ?*IOStats = null,

    pub fn init(file: std.fs.File, buffer_size: usize) !BufferedReader {
        var reader = BufferedReader{
            .file = file,
            .pos = 0,
            .end = 0,
        };
        try reader.buffer.resize(file.allocator, buffer_size);
        return reader;
    }

    pub fn deinit(self: *BufferedReader, allocator: std.mem.Allocator) void {
        self.buffer.deinit(allocator);
    }

    pub fn fill(self: *BufferedReader) !usize {
        self.pos = 0;
        const bytes_read = try self.file.readAll(self.buffer.items);
        self.end = bytes_read;
        if (self.stats) |stats| {
            stats.recordRead(@intCast(bytes_read));
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

pub const BufferedWriter = struct {
    file: std.fs.File,
    buffer: std.ArrayListUnmanaged(u8) = std.ArrayListUnmanaged(u8).empty,
    pos: usize = 0,
    stats: ?*IOStats = null,

    pub fn init(file: std.fs.File, buffer_size: usize) !BufferedWriter {
        var writer = BufferedWriter{
            .file = file,
            .pos = 0,
        };
        try writer.buffer.resize(file.allocator, buffer_size);
        return writer;
    }

    pub fn deinit(self: *BufferedWriter, allocator: std.mem.Allocator) void {
        self.buffer.deinit(allocator);
    }

    pub fn flush(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer.items[0..self.pos]);
            if (self.stats) |stats| {
                stats.recordWrite(@intCast(self.pos));
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

pub const FileStream = struct {
    file: ?std.fs.File = null,
    path: []const u8,
    mode: Mode,
    stats: ?*IOStats = null,

    pub const Mode = enum {
        read,
        write,
        append,
        read_write,
    };

    pub fn open(path: []const u8, mode: Mode) !FileStream {
        const file = switch (mode) {
            .read => try std.fs.openFileAbsolute(path, .{}),
            .write => try std.fs.createFileAbsolute(path, .{ .truncate = true }),
            .append => try std.fs.createFileAbsolute(path, .{ .append = true, .read = true }),
            .read_write => try std.fs.createFileAbsolute(path, .{ .read = true }),
        };
        return FileStream{
            .file = file,
            .path = path,
            .mode = mode,
        };
    }

    pub fn deinit(self: *FileStream) void {
        if (self.file) |f| {
            f.close();
            self.file = null;
        }
    }

    pub fn read(self: *FileStream, buf: []u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.read(buf);
        if (self.stats) |stats| {
            stats.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn readAll(self: *FileStream, buf: []u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.readAll(buf);
        if (self.stats) |stats| {
            stats.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn write(self: *FileStream, data: []const u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.write(data);
        if (self.stats) |stats| {
            stats.recordWrite(@intCast(n));
        }
        return n;
    }

    pub fn writeAll(self: *FileStream, data: []const u8) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.writeAll(data);
        if (self.stats) |stats| {
            stats.recordWrite(@intCast(data.len));
        }
    }

    pub fn seekTo(self: *FileStream, offset: u64) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.seekTo(offset);
    }

    pub fn seekBy(self: *FileStream, offset: i64) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.seekBy(offset);
    }

    pub fn getPos(self: *FileStream) !u64 {
        const f = self.file orelse return error.FileNotOpen;
        return try f.getPos();
    }

    pub fn getEndPos(self: *FileStream) !u64 {
        const f = self.file orelse return error.FileNotOpen;
        return try f.getEndPos();
    }

    pub fn stat(self: *FileStream) !std.fs.File.Stat {
        const f = self.file orelse return error.FileNotOpen;
        return try f.stat();
    }

    pub fn sync(self: *FileStream) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.sync();
    }

    pub fn isOpen(self: *FileStream) bool {
        return self.file != null;
    }
};

pub fn asyncReadFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const stat = try file.stat();
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(buf);

    const bytes_read = try file.readAll(buf);
    if (bytes_read < stat.size) {
        allocator.free(buf);
        return error.UnexpectedEOF;
    }

    return buf;
}

pub fn asyncWriteFile(path: []const u8, data: []const u8) !void {
    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer file.close();

    try file.writeAll(data);
}

pub fn asyncAppendFile(path: []const u8, data: []const u8) !void {
    const file = try std.fs.createFileAbsolute(path, .{ .append = true });
    defer file.close();

    try file.writeAll(data);
}

pub fn resolvePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        return try allocator.dupe(u8, path);
    }

    var buf: [MAX_PATH_LEN]u8 = undefined;
    const cwd = std.fs.cwd();
    const resolved = try cwd.realpath(path, &buf);
    return try allocator.dupe(u8, resolved);
}

pub fn ensureDir(path: []const u8) !void {
    if (dirExists(path)) return;

    var iter = std.mem.splitScalar(u8, path, std.fs.path.sep);
    var built = std.ArrayListUnmanaged(u8).empty;
    defer built.deinit(std.heap.page_allocator);

    var first = true;
    while (iter.next()) |component| {
        if (component.len == 0) {
            if (first) {
                try built.append(std.heap.page_allocator, std.fs.path.sep);
                first = false;
            }
            continue;
        }
        first = false;
        if (built.items.len > 0 and built.items[built.items.len - 1] != std.fs.path.sep) {
            try built.append(std.heap.page_allocator, std.fs.path.sep);
        }
        try built.appendSlice(std.heap.page_allocator, component);

        const dir_path = built.items;
        std.fs.makeDirAbsolute(dir_path) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
    }
}

pub fn fileExists(path: []const u8) bool {
    std.fs.accessAbsolute(path, .{ .mode = .read_only }) catch return false;
    return true;
}

pub fn dirExists(path: []const u8) bool {
    const stat = std.fs.cwd().statFile(path) catch return false;
    return stat.kind == .directory;
}

test {
    std.testing.refAllDecls(@This());
}

test "IOStats record and snapshot" {
    var stats = IOStats{};
    stats.recordRead(100);
    stats.recordRead(200);
    stats.recordWrite(50);
    stats.recordError();

    const snap = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 300), snap.bytes_read);
    try std.testing.expectEqual(@as(u64, 50), snap.bytes_written);
    try std.testing.expectEqual(@as(u64, 2), snap.read_ops);
    try std.testing.expectEqual(@as(u64, 1), snap.write_ops);
    try std.testing.expectEqual(@as(u64, 1), snap.errors);
    try std.testing.expect(snap.last_activity_ms > 0);
}

test "IOStats reset" {
    var stats = IOStats{};
    stats.recordRead(100);
    stats.reset();

    const snap = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snap.bytes_read);
    try std.testing.expectEqual(@as(u64, 0), snap.bytes_written);
    try std.testing.expectEqual(@as(u64, 0), snap.read_ops);
}

test "asyncWriteFile and asyncReadFile roundtrip" {
    const test_path = "/tmp/abi_io_async_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const content = "hello async io world";
    try asyncWriteFile(test_path, content);

    const read_content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(read_content);

    try std.testing.expectEqualStrings(content, read_content);
}

test "asyncAppendFile" {
    const test_path = "/tmp/abi_io_append_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    try asyncWriteFile(test_path, "line1\n");
    try asyncAppendFile(test_path, "line2\n");

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("line1\nline2\n", content);
}

test "fileExists and dirExists" {
    try std.testing.expect(fileExists("/tmp"));
    try std.testing.expect(dirExists("/tmp"));
    try std.testing.expect(!fileExists("/tmp/nonexistent_file_12345"));
    try std.testing.expect(!dirExists("/tmp/nonexistent_dir_12345"));
}

test "ensureDir creates nested directories" {
    const test_dir = "/tmp/abi_io_test_nested/a/b/c";
    defer {
        std.fs.deleteTreeAbsolute("/tmp/abi_io_test_nested") catch {};
    }

    try ensureDir(test_dir);
    try std.testing.expect(dirExists(test_dir));
}

test "resolvePath absolute path" {
    const abs = "/tmp/test";
    const resolved = try resolvePath(std.testing.allocator, abs);
    defer std.testing.allocator.free(resolved);

    try std.testing.expectEqualStrings(abs, resolved);
}

test "BufferedReader basic read" {
    const test_path = "/tmp/abi_io_buffered_read_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    try asyncWriteFile(test_path, "hello buffered reader");

    const file = try std.fs.openFileAbsolute(test_path, .{});
    defer file.close();

    var reader = try BufferedReader.init(file, 64);
    defer reader.deinit(std.testing.allocator);

    var buf: [64]u8 = undefined;
    const n = try reader.readAll(&buf);
    try std.testing.expectEqualStrings("hello buffered reader", buf[0..n]);
}

test "BufferedReader readByte" {
    const test_path = "/tmp/abi_io_readbyte_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    try asyncWriteFile(test_path, "AB");

    const file = try std.fs.openFileAbsolute(test_path, .{});
    defer file.close();

    var reader = try BufferedReader.init(file, 64);
    defer reader.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u8, 'A'), try reader.readByte());
    try std.testing.expectEqual(@as(u8, 'B'), try reader.readByte());
}

test "BufferedWriter basic write" {
    const test_path = "/tmp/abi_io_buffered_write_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const file = try std.fs.createFileAbsolute(test_path, .{ .truncate = true });
    defer file.close();

    var writer = try BufferedWriter.init(file, 64);
    defer writer.deinit(std.testing.allocator);

    try writer.writeAll("hello buffered writer");
    try writer.flush();

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("hello buffered writer", content);
}

test "BufferedWriter writeByte" {
    const test_path = "/tmp/abi_io_writebyte_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const file = try std.fs.createFileAbsolute(test_path, .{ .truncate = true });
    defer file.close();

    var writer = try BufferedWriter.init(file, 64);
    defer writer.deinit(std.testing.allocator);

    try writer.writeByte('X');
    try writer.writeByte('Y');
    try writer.writeByte('Z');
    try writer.flush();

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("XYZ", content);
}

test "FileStream read and seek" {
    const test_path = "/tmp/abi_io_filestream_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    try asyncWriteFile(test_path, "0123456789");

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
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    {
        var stream = try FileStream.open(test_path, .write);
        defer stream.deinit();

        try stream.writeAll("stream data");
        try stream.sync();
    }

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("stream data", content);
}

test "FileStream with stats" {
    const test_path = "/tmp/abi_io_filestream_stats_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    var stats = IOStats{};

    {
        var stream = try FileStream.open(test_path, .write);
        defer stream.deinit();
        stream.stats = &stats;

        try stream.writeAll("stats test");
    }

    {
        var stream = try FileStream.open(test_path, .read);
        defer stream.deinit();
        stream.stats = &stats;

        var buf: [32]u8 = undefined;
        _ = try stream.readAll(&buf);
    }

    const snap = stats.snapshot();
    try std.testing.expect(snap.bytes_written > 0);
    try std.testing.expect(snap.bytes_read > 0);
    try std.testing.expect(snap.write_ops > 0);
    try std.testing.expect(snap.read_ops > 0);
}

test "BufferedReader with stats" {
    const test_path = "/tmp/abi_io_bufreader_stats_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    try asyncWriteFile(test_path, "stats buffered read");

    var stats = IOStats{};

    const file = try std.fs.openFileAbsolute(test_path, .{});
    defer file.close();

    var reader = try BufferedReader.init(file, 64);
    defer reader.deinit(std.testing.allocator);
    reader.stats = &stats;

    var buf: [64]u8 = undefined;
    _ = try reader.readAll(&buf);

    const snap = stats.snapshot();
    try std.testing.expect(snap.bytes_read > 0);
    try std.testing.expect(snap.read_ops > 0);
}

test "BufferedWriter with stats" {
    const test_path = "/tmp/abi_io_bufwriter_stats_test.txt";
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    var stats = IOStats{};

    const file = try std.fs.createFileAbsolute(test_path, .{ .truncate = true });
    defer file.close();

    var writer = try BufferedWriter.init(file, 64);
    defer writer.deinit(std.testing.allocator);
    writer.stats = &stats;

    try writer.writeAll("stats buffered write");
    try writer.flush();

    const snap = stats.snapshot();
    try std.testing.expect(snap.bytes_written > 0);
    try std.testing.expect(snap.write_ops > 0);
}
