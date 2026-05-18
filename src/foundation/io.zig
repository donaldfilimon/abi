const std = @import("std");
const sync = @import("sync.zig");
const time = @import("time.zig");

pub const DEFAULT_BUFFER_SIZE = 4096;
pub const MAX_PATH_LEN = std.Io.Dir.max_path_bytes;

fn defaultIo() std.Io {
    return std.Options.debug_io;
}

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
    file: std.Io.File,
    allocator: std.mem.Allocator = std.heap.page_allocator,
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
        if (self.stats) |stats| {
            stats.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn readAll(self: *FileStream, buf: []u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.readPositionalAll(defaultIo(), buf, self.offset);
        self.offset += n;
        if (self.stats) |stats| {
            stats.recordRead(@intCast(n));
        }
        return n;
    }

    pub fn write(self: *FileStream, data: []const u8) !usize {
        const f = self.file orelse return error.FileNotOpen;
        const n = try f.writePositional(defaultIo(), &.{data}, self.offset);
        self.offset += n;
        if (self.stats) |stats| {
            stats.recordWrite(@intCast(n));
        }
        return n;
    }

    pub fn writeAll(self: *FileStream, data: []const u8) !void {
        const f = self.file orelse return error.FileNotOpen;
        try f.writePositionalAll(defaultIo(), data, self.offset);
        self.offset += data.len;
        if (self.stats) |stats| {
            stats.recordWrite(@intCast(data.len));
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
    std.Io.Dir.deleteFileAbsolute(defaultIo(), path) catch {};
}

fn deleteTreeForTest(path: []const u8) void {
    std.Io.Dir.deleteTree(.cwd(), defaultIo(), path) catch {};
}

fn openFileForRead(path: []const u8) !std.Io.File {
    return try std.Io.Dir.openFileAbsolute(defaultIo(), path, .{});
}

fn createFileForWrite(path: []const u8) !std.Io.File {
    return try std.Io.Dir.createFileAbsolute(defaultIo(), path, .{ .truncate = true });
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
    defer deleteFileForTest(test_path);

    const content = "hello async io world";
    try asyncWriteFile(test_path, content);

    const read_content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(read_content);

    try std.testing.expectEqualStrings(content, read_content);
}

test "asyncAppendFile" {
    const test_path = "/tmp/abi_io_append_test.txt";
    defer deleteFileForTest(test_path);

    try asyncWriteFile(test_path, "line1\n");
    try asyncAppendFile(test_path, "line2\n");

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("line1\nline2\n", content);
}

test "asyncAppendFile creates missing file" {
    const test_path = "/tmp/abi_io_append_create_test.txt";
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
    const test_dir = "/tmp/abi_io_test_nested/a/b/c";
    defer {
        deleteTreeForTest("/tmp/abi_io_test_nested");
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
    defer deleteFileForTest(test_path);

    try asyncWriteFile(test_path, "hello buffered reader");

    const file = try openFileForRead(test_path);
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
    defer reader.deinit(std.testing.allocator);

    var buf: [64]u8 = undefined;
    const n = try reader.readAll(&buf);
    try std.testing.expectEqualStrings("hello buffered reader", buf[0..n]);
}

test "BufferedReader readByte" {
    const test_path = "/tmp/abi_io_readbyte_test.txt";
    defer deleteFileForTest(test_path);

    try asyncWriteFile(test_path, "AB");

    const file = try openFileForRead(test_path);
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
    defer reader.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u8, 'A'), try reader.readByte());
    try std.testing.expectEqual(@as(u8, 'B'), try reader.readByte());
}

test "BufferedWriter basic write" {
    const test_path = "/tmp/abi_io_buffered_write_test.txt";
    defer deleteFileForTest(test_path);

    const file = try createFileForWrite(test_path);
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
    defer writer.deinit(std.testing.allocator);

    try writer.writeAll("hello buffered writer");
    try writer.flush();

    const content = try asyncReadFile(std.testing.allocator, test_path);
    defer std.testing.allocator.free(content);

    try std.testing.expectEqualStrings("hello buffered writer", content);
}

test "BufferedWriter writeByte" {
    const test_path = "/tmp/abi_io_writebyte_test.txt";
    defer deleteFileForTest(test_path);

    const file = try createFileForWrite(test_path);
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
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
    defer deleteFileForTest(test_path);

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
    defer deleteFileForTest(test_path);

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
    defer deleteFileForTest(test_path);

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
    defer deleteFileForTest(test_path);

    try asyncWriteFile(test_path, "stats buffered read");

    var stats = IOStats{};

    const file = try openFileForRead(test_path);
    defer file.close(defaultIo());

    var reader = try BufferedReader.initWithAllocator(std.testing.allocator, file, 64);
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
    defer deleteFileForTest(test_path);

    var stats = IOStats{};

    const file = try createFileForWrite(test_path);
    defer file.close(defaultIo());

    var writer = try BufferedWriter.initWithAllocator(std.testing.allocator, file, 64);
    defer writer.deinit(std.testing.allocator);
    writer.stats = &stats;

    try writer.writeAll("stats buffered write");
    try writer.flush();

    const snap = stats.snapshot();
    try std.testing.expect(snap.bytes_written > 0);
    try std.testing.expect(snap.write_ops > 0);
}
