//! Write-Ahead Log (WAL) with file I/O for crash recovery.
//!
//! Binary format per entry:
//!   4 bytes: payload length (u32, little-endian)
//!   8 bytes: sequence number (u64, little-endian)
//!   1 byte:  entry type enum
//!   N bytes: payload data
//!   4 bytes: CRC32 checksum over [length .. payload]

const std = @import("std");
const Crc32 = @import("integrity.zig").Crc32;
const Io = std.Io;
const File = Io.File;
const Dir = Io.Dir;

pub const HEADER_SIZE: usize = 4 + 8 + 1;
pub const CHECKSUM_SIZE: usize = 4;
pub const FRAME_OVERHEAD: usize = HEADER_SIZE + CHECKSUM_SIZE;

pub const WalEntry = struct {
    seq: u64,
    entry_type: WalEntryType,
    data: []const u8,
    checksum: u32,
};

pub const WalEntryType = enum(u8) {
    insert = 0x01,
    update = 0x02,
    delete = 0x03,
    checkpoint = 0x10,
    commit = 0xFF,
};

fn computeFrameChecksum(length: u32, seq: u64, entry_type: WalEntryType, data: []const u8) u32 {
    var crc = Crc32{};
    crc.update(&std.mem.toBytes(std.mem.nativeTo(u32, length, .little)));
    crc.update(&std.mem.toBytes(std.mem.nativeTo(u64, seq, .little)));
    crc.update(&.{@intFromEnum(entry_type)});
    crc.update(data);
    return crc.finalize();
}

fn initIo(allocator: std.mem.Allocator) Io.Threaded {
    return Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

fn serializeFrame(allocator: std.mem.Allocator, seq: u64, entry_type: WalEntryType, data: []const u8) ![]u8 {
    const length: u32 = @intCast(data.len);
    const checksum = computeFrameChecksum(length, seq, entry_type, data);
    const total = HEADER_SIZE + data.len + CHECKSUM_SIZE;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);

    @memcpy(buf[0..4], &std.mem.toBytes(std.mem.nativeTo(u32, length, .little)));
    @memcpy(buf[4..12], &std.mem.toBytes(std.mem.nativeTo(u64, seq, .little)));
    buf[12] = @intFromEnum(entry_type);
    if (data.len > 0) {
        @memcpy(buf[HEADER_SIZE .. HEADER_SIZE + data.len], data);
    }
    @memcpy(buf[HEADER_SIZE + data.len ..][0..4], &std.mem.toBytes(std.mem.nativeTo(u32, checksum, .little)));
    return buf;
}

// ============================================================================
// WalWriter
// ============================================================================

pub const WalWriter = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,
    seq: u64,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !WalWriter {
        var io_backend = initIo(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Create the file if it doesn't exist (non-truncating open)
        const file = Dir.cwd().createFile(io, path, .{ .truncate = false, .read = true }) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return error.Unexpected,
        };
        const file_len = file.length(io) catch 0;
        var highest: u64 = 0;
        if (file_len > 0) {
            highest = recoverHighestSeq(allocator, file, io, file_len);
        }
        file.close(io);

        return .{
            .allocator = allocator,
            .file_path = path,
            .seq = highest,
        };
    }

    fn recoverHighestSeq(allocator: std.mem.Allocator, file: File, io: Io, file_len: u64) u64 {
        // Read the full file and parse frames in memory
        const data = allocator.alloc(u8, @intCast(file_len)) catch return 0;
        defer allocator.free(data);
        const read = file.readPositionalAll(io, data, 0) catch return 0;
        if (read < FRAME_OVERHEAD) return 0;
        const buf = data[0..read];

        var pos: usize = 0;
        var highest: u64 = 0;
        while (pos + FRAME_OVERHEAD <= buf.len) {
            const length = std.mem.readInt(u32, buf[pos..][0..4], .little);
            const seq = std.mem.readInt(u64, buf[pos + 4 ..][0..8], .little);
            const frame_size = HEADER_SIZE + @as(usize, length) + CHECKSUM_SIZE;
            if (pos + frame_size > buf.len) break;
            if (seq > highest) highest = seq;
            pos += frame_size;
        }
        return highest;
    }

    pub fn append(self: *WalWriter, entry_type: WalEntryType, data: []const u8) !u64 {
        self.seq += 1;

        const frame = try serializeFrame(self.allocator, self.seq, entry_type, data);
        defer self.allocator.free(frame);

        var io_backend = initIo(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Open for append: create non-truncating, then write at end
        const file = Dir.cwd().createFile(io, self.file_path, .{ .truncate = false }) catch
            return error.Unexpected;
        defer file.close(io);

        const file_len = file.length(io) catch return error.Unexpected;
        file.writePositionalAll(io, frame, file_len) catch return error.Unexpected;
        file.sync(io) catch {};

        return self.seq;
    }

    pub fn checkpoint(self: *WalWriter) !void {
        _ = try self.append(.checkpoint, &.{});
    }

    pub fn truncate(self: *WalWriter) !void {
        var io_backend = initIo(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = Dir.cwd().createFile(io, self.file_path, .{ .truncate = false }) catch
            return error.Unexpected;
        defer file.close(io);
        file.setLength(io, 0) catch return error.Unexpected;
    }

    pub fn deinit(self: *WalWriter) void {
        _ = self;
    }
};

// ============================================================================
// WalReader
// ============================================================================

pub const WalReader = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !WalReader {
        var io_backend = initIo(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        // Verify the file exists
        Dir.cwd().access(io, path, .{}) catch return error.FileNotFound;

        return .{
            .allocator = allocator,
            .file_path = path,
        };
    }

    pub fn replay(self: *WalReader) ![]WalEntry {
        var io_backend = initIo(self.allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = Dir.cwd().openFile(io, self.file_path, .{}) catch
            return error.OpenFailed;
        defer file.close(io);

        const file_len = file.length(io) catch return error.Unexpected;
        if (file_len == 0) {
            const empty = try self.allocator.alloc(WalEntry, 0);
            return empty;
        }

        const data = try self.allocator.alloc(u8, @intCast(file_len));
        defer self.allocator.free(data);
        const read = file.readPositionalAll(io, data, 0) catch return error.Unexpected;
        const buf = data[0..read];

        var entries = std.ArrayListUnmanaged(WalEntry).empty;
        errdefer {
            for (entries.items) |entry| {
                if (entry.data.len > 0) self.allocator.free(entry.data);
            }
            entries.deinit(self.allocator);
        }

        var pos: usize = 0;
        while (pos + FRAME_OVERHEAD <= buf.len) {
            const length = std.mem.readInt(u32, buf[pos..][0..4], .little);
            const seq = std.mem.readInt(u64, buf[pos + 4 ..][0..8], .little);
            const entry_type_byte = buf[pos + 12];

            const entry_type: WalEntryType = switch (entry_type_byte) {
                0x01 => .insert,
                0x02 => .update,
                0x03 => .delete,
                0x10 => .checkpoint,
                0xFF => .commit,
                else => break,
            };

            const frame_size = HEADER_SIZE + @as(usize, length) + CHECKSUM_SIZE;
            if (pos + frame_size > buf.len) break;

            const payload_start = pos + HEADER_SIZE;
            const payload_end = payload_start + length;
            const payload = buf[payload_start..payload_end];

            const stored_checksum = std.mem.readInt(u32, buf[payload_end..][0..4], .little);
            const computed = computeFrameChecksum(length, seq, entry_type, payload);
            if (stored_checksum != computed) {
                return error.InvalidChecksum;
            }

            // Copy payload to owned memory
            const owned_data: []u8 = if (length > 0) blk: {
                const d = try self.allocator.alloc(u8, length);
                @memcpy(d, payload);
                break :blk d;
            } else @constCast(&[_]u8{});

            try entries.append(self.allocator, .{
                .seq = seq,
                .entry_type = entry_type,
                .data = owned_data,
                .checksum = stored_checksum,
            });

            pos += frame_size;
        }

        return entries.toOwnedSlice(self.allocator);
    }

    pub fn deinit(self: *WalReader) void {
        _ = self;
    }

    pub fn freeEntries(allocator: std.mem.Allocator, entries: []WalEntry) void {
        for (entries) |entry| {
            if (entry.data.len > 0) allocator.free(entry.data);
        }
        allocator.free(entries);
    }
};

// ============================================================================
// Tests
// ============================================================================

fn getTestPath(buf: *[128]u8, name: []const u8) ![]u8 {
    return std.fmt.bufPrint(buf, "/tmp/abi_wal_test_{d}_{s}.wal", .{ @import("../../../foundation/mod.zig").time.unixMs(), name });
}

fn deleteTestFile(path: []const u8) void {
    const allocator = std.testing.allocator;
    var io_backend = initIo(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    Dir.cwd().deleteFile(io, path) catch {};
}

test "wal write and replay entries" {
    const allocator = std.testing.allocator;
    var path_buf: [128]u8 = undefined;
    const path = try getTestPath(&path_buf, "roundtrip");
    defer deleteTestFile(path);
    deleteTestFile(path);

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();

        const seq1 = try writer.append(.insert, "hello");
        try std.testing.expectEqual(@as(u64, 1), seq1);

        const seq2 = try writer.append(.update, "world");
        try std.testing.expectEqual(@as(u64, 2), seq2);

        const seq3 = try writer.append(.delete, "gone");
        try std.testing.expectEqual(@as(u64, 3), seq3);
    }

    {
        var reader = try WalReader.init(allocator, path);
        defer reader.deinit();

        const entries = try reader.replay();
        defer WalReader.freeEntries(allocator, entries);

        try std.testing.expectEqual(@as(usize, 3), entries.len);

        try std.testing.expectEqual(@as(u64, 1), entries[0].seq);
        try std.testing.expectEqual(WalEntryType.insert, entries[0].entry_type);
        try std.testing.expectEqualStrings("hello", entries[0].data);

        try std.testing.expectEqual(@as(u64, 2), entries[1].seq);
        try std.testing.expectEqual(WalEntryType.update, entries[1].entry_type);
        try std.testing.expectEqualStrings("world", entries[1].data);

        try std.testing.expectEqual(@as(u64, 3), entries[2].seq);
        try std.testing.expectEqual(WalEntryType.delete, entries[2].entry_type);
        try std.testing.expectEqualStrings("gone", entries[2].data);
    }
}

test "wal checkpoint and truncate" {
    const allocator = std.testing.allocator;
    var path_buf: [128]u8 = undefined;
    const path = try getTestPath(&path_buf, "checkpoint");
    defer deleteTestFile(path);
    deleteTestFile(path);

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();

        _ = try writer.append(.insert, "data1");
        _ = try writer.append(.insert, "data2");
        try writer.checkpoint();
        try writer.truncate();

        _ = try writer.append(.insert, "data3");
    }

    {
        var reader = try WalReader.init(allocator, path);
        defer reader.deinit();

        const entries = try reader.replay();
        defer WalReader.freeEntries(allocator, entries);

        try std.testing.expectEqual(@as(usize, 1), entries.len);
        try std.testing.expectEqualStrings("data3", entries[0].data);
    }
}

test "wal crc detects corruption" {
    const allocator = std.testing.allocator;
    var path_buf: [128]u8 = undefined;
    const path = try getTestPath(&path_buf, "corrupt");
    defer deleteTestFile(path);
    deleteTestFile(path);

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();
        _ = try writer.append(.insert, "important data");
    }

    // Corrupt a byte in the payload area
    {
        var io_backend = initIo(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = try Dir.cwd().createFile(io, path, .{ .truncate = false });
        defer file.close(io);
        file.writePositionalAll(io, &.{0xFF}, HEADER_SIZE + 2) catch unreachable;
    }

    {
        var reader = try WalReader.init(allocator, path);
        defer reader.deinit();

        const result = reader.replay();
        try std.testing.expectError(error.InvalidChecksum, result);
    }
}

test "wal empty payload entries" {
    const allocator = std.testing.allocator;
    var path_buf: [128]u8 = undefined;
    const path = try getTestPath(&path_buf, "empty");
    defer deleteTestFile(path);
    deleteTestFile(path);

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();
        try writer.checkpoint();
        _ = try writer.append(.commit, &.{});
    }

    {
        var reader = try WalReader.init(allocator, path);
        defer reader.deinit();

        const entries = try reader.replay();
        defer WalReader.freeEntries(allocator, entries);

        try std.testing.expectEqual(@as(usize, 2), entries.len);
        try std.testing.expectEqual(WalEntryType.checkpoint, entries[0].entry_type);
        try std.testing.expectEqual(WalEntryType.commit, entries[1].entry_type);
    }
}

test "wal resume appending to existing file" {
    const allocator = std.testing.allocator;
    var path_buf: [128]u8 = undefined;
    const path = try getTestPath(&path_buf, "resume");
    defer deleteTestFile(path);
    deleteTestFile(path);

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();
        _ = try writer.append(.insert, "first");
        _ = try writer.append(.insert, "second");
    }

    {
        var writer = try WalWriter.init(allocator, path);
        defer writer.deinit();
        const seq = try writer.append(.insert, "third");
        try std.testing.expectEqual(@as(u64, 3), seq);
    }

    {
        var reader = try WalReader.init(allocator, path);
        defer reader.deinit();

        const entries = try reader.replay();
        defer WalReader.freeEntries(allocator, entries);

        try std.testing.expectEqual(@as(usize, 3), entries.len);
        try std.testing.expectEqualStrings("first", entries[0].data);
        try std.testing.expectEqualStrings("second", entries[1].data);
        try std.testing.expectEqualStrings("third", entries[2].data);
    }
}
