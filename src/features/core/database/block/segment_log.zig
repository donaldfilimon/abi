//! Append-only segments.
//!
//! Provides durable, append-only block storage using POSIX file I/O.
//! Each block is written as a length-prefixed (u32 LE) encoded frame.

const std = @import("std");
const core = @import("../../mod.zig");
const block = @import("block.zig");
const codec = @import("codec.zig");

pub const SegmentLog = struct {
    allocator: std.mem.Allocator,
    /// Path to the segment file on disk.
    path: []const u8,
    current_offset: u64,

    pub fn init(allocator: std.mem.Allocator, path: []const u8, initial_size: u64) !SegmentLog {
        return SegmentLog{
            .allocator = allocator,
            .path = path,
            .current_offset = initial_size,
        };
    }

    pub fn deinit(self: *SegmentLog) void {
        _ = self;
        // Path is caller-owned; nothing to free.
    }

    /// Append a block to the segment file. Returns the byte offset at which
    /// the frame was written.
    ///
    /// On-disk frame format:
    ///   [u32 LE length][encoded block bytes ...]
    pub fn append(self: *SegmentLog, b: block.StoredBlock) !u64 {
        const encoded = try codec.encodeBlock(self.allocator, b);
        defer self.allocator.free(encoded);

        const frame_offset = self.current_offset;

        const file = try std.fs.openFileAbsolute(self.path, .{ .mode = .write_only });
        defer file.close();

        // Seek to the write position.
        try file.seekTo(self.current_offset);

        // Write the length prefix (u32 LE).
        const len: u32 = @intCast(encoded.len);
        var len_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &len_bytes, len, .little);
        try writeAll(file, &len_bytes);

        // Write the encoded block data.
        try writeAll(file, encoded);

        self.current_offset += @as(u64, 4) + @as(u64, encoded.len);

        return frame_offset;
    }

    /// Read a block back from the segment file at the given byte offset.
    pub fn readAt(self: *SegmentLog, offset: u64) !block.StoredBlock {
        const file = try std.fs.openFileAbsolute(self.path, .{ .mode = .read_only });
        defer file.close();

        // Seek to the frame offset.
        try file.seekTo(offset);

        // Read the u32 LE length prefix.
        var len_bytes: [4]u8 = undefined;
        try readAll(file, &len_bytes);
        const len = std.mem.readInt(u32, &len_bytes, .little);

        // Read the encoded block data.
        const data = try self.allocator.alloc(u8, len);
        errdefer self.allocator.free(data);
        try readAll(file, data);
        defer self.allocator.free(data);

        // Decode and return — the caller owns the decoded payload memory.
        return codec.decodeBlock(self.allocator, data);
    }

    // -- helpers --

    fn writeAll(file: std.fs.File, buf: []const u8) !void {
        try file.writer().writeAll(buf);
    }

    fn readAll(file: std.fs.File, buf: []u8) !void {
        try file.reader().readNoEof(buf);
    }
};

test "SegmentLog append and read back" {
    const allocator = std.testing.allocator;

    // Use a temp file path.
    const tmp_path = "/tmp/abi_segment_log_test.seg";

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Clean up any leftover file from a previous run.
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    var log = try SegmentLog.init(allocator, tmp_path, 0);
    defer log.deinit();

    // Build a minimal StoredBlock.
    const payload = "hello segment";
    const b = block.StoredBlock{
        .header = .{
            .id = .{ .id = @as([32]u8, @splat(0xAB)) },
            .kind = @enumFromInt(0),
            .version = 1,
            .content_hash = @as([32]u8, @splat(0)),
            .timestamp = .{ .counter = 42 },
            .size = @intCast(payload.len),
            .flags = 0,
            .compression_marker = 0,
        },
        .payload = payload,
    };

    const offset0 = try log.append(b);
    try std.testing.expectEqual(@as(u64, 0), offset0);

    // Append a second block.
    const payload2 = "world";
    const b2 = block.StoredBlock{
        .header = .{
            .id = .{ .id = @as([32]u8, @splat(0xCD)) },
            .kind = @enumFromInt(0),
            .version = 2,
            .content_hash = @as([32]u8, @splat(0)),
            .timestamp = .{ .counter = 99 },
            .size = @intCast(payload2.len),
            .flags = 0,
            .compression_marker = 0,
        },
        .payload = payload2,
    };

    const offset1 = try log.append(b2);
    try std.testing.expect(offset1 > offset0);

    // Read back first block.
    var rb = try log.readAt(offset0);
    defer allocator.free(rb.payload);
    try std.testing.expectEqualSlices(u8, payload, rb.payload);
    try std.testing.expectEqual(@as(u32, 1), rb.header.version);

    // Read back second block.
    var rb2 = try log.readAt(offset1);
    defer allocator.free(rb2.payload);
    try std.testing.expectEqualSlices(u8, payload2, rb2.payload);
    try std.testing.expectEqual(@as(u32, 2), rb2.header.version);

    // Cleanup.
}
