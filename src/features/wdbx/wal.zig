//! Write-ahead log for the WDBX Store (Storage Layer).
//!
//! Each mutation is appended as a CRC32-framed, JSON-encoded record so the log
//! is append-only, deterministic, and verifiable. Replay reconstructs a Store
//! by handing the verified record bodies to `persistence.deserialize`, reusing
//! the exact same apply path as snapshot restore. A truncated tail or a flipped
//! byte is rejected with `error.WalCorruption` rather than silently replaying
//! partial state.
//!
//! Frame format (one record per line, after the header line):
//!   <crc32-hex8> <minified-json>\n
//! where crc32 is computed over the JSON bytes only.

const std = @import("std");
const wdbx_mod = @import("mod.zig");
const persistence = @import("persistence.zig");

pub const WAL_HEADER = "# ABI-WDBX-WAL v1";

pub const WalError = error{
    InvalidHeader,
    WalCorruption,
    MissingFrame,
};

fn crc32Hex(bytes: []const u8) [8]u8 {
    const sum = std.hash.crc.Crc32.hash(bytes);
    return std.fmt.bytesToHex(std.mem.toBytes(std.mem.nativeToBig(u32, sum)), .lower);
}

/// Append one already-encoded JSON record to the WAL at `path`, creating the
/// log (with header) on first write. The record is framed with a CRC32 of its
/// bytes so replay can detect corruption.
pub fn appendRecord(io: std.Io, allocator: std.mem.Allocator, path: []const u8, json: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const existing = cwd.readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    defer if (existing) |e| allocator.free(e);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    if (existing) |e| {
        try out.writer.writeAll(e);
    } else {
        try out.writer.writeAll(WAL_HEADER);
        try out.writer.writeAll("\n");
    }
    const crc = crc32Hex(json);
    try out.writer.writeAll(&crc);
    try out.writer.writeAll(" ");
    try out.writer.writeAll(json);
    try out.writer.writeAll("\n");

    try cwd.writeFile(io, .{ .sub_path = path, .data = out.written() });
}

/// Append a key/value mutation record.
pub fn appendKv(io: std.Io, allocator: std.mem.Allocator, path: []const u8, key: []const u8, value: []const u8) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("kv");
    try w.objectField("key");
    try w.write(key);
    try w.objectField("value");
    try w.write(value);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append a conversation-block mutation record (with explicit timestamp so the
/// replayed SHA-256 chain hash reproduces the original exactly).
pub fn appendBlock(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8, timestamp_ms: i64) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("block");
    try w.objectField("profile");
    try w.write(profile);
    try w.objectField("query_id");
    try w.write(query_id);
    try w.objectField("response_id");
    try w.write(response_id);
    try w.objectField("metadata");
    try w.write(metadata);
    try w.objectField("timestamp_ms");
    try w.write(timestamp_ms);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append a temporal timestamp record. Replay routes this through the snapshot
/// parser, so WAL and JSONL checkpoint restore keep the same validation path.
pub fn appendTemporalNode(io: std.Io, allocator: std.mem.Allocator, path: []const u8, id: u32, timestamp_ms: i64) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("temporal_node");
    try w.objectField("id");
    try w.write(id);
    try w.objectField("timestamp_ms");
    try w.write(timestamp_ms);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Append one causal edge record for temporal/causal hybrid ranking recovery.
pub fn appendTemporalEdge(io: std.Io, allocator: std.mem.Allocator, path: []const u8, cause: u32, effect: u32) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    var w = std.json.Stringify{ .writer = &buf.writer, .options = .{ .whitespace = .minified } };
    try w.beginObject();
    try w.objectField("type");
    try w.write("temporal_edge");
    try w.objectField("cause");
    try w.write(cause);
    try w.objectField("effect");
    try w.write(effect);
    try w.endObject();
    try appendRecord(io, allocator, path, buf.written());
}

/// Verify every frame's CRC32 and return the number of records, without
/// building a Store. Use for a fast integrity check (`wdbx db verify`).
pub fn verify(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !usize {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);
    var count: usize = 0;
    var it = frameIterator(content);
    while (try it.next()) |_| count += 1;
    return count;
}

/// Replay the WAL into a fresh Store. Reuses `persistence.deserialize` so the
/// apply semantics (id/timestamp/hash fidelity, range checks) are identical to
/// snapshot restore.
pub fn replay(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !wdbx_mod.Store {
    const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(content);

    var body: std.Io.Writer.Allocating = .init(allocator);
    defer body.deinit();
    try body.writer.writeAll(persistence.HEADER);
    try body.writer.writeAll("\n");

    var it = frameIterator(content);
    while (try it.next()) |json| {
        try body.writer.writeAll(json);
        try body.writer.writeAll("\n");
    }

    return persistence.deserialize(allocator, body.written());
}

const FrameIterator = struct {
    lines: std.mem.SplitIterator(u8, .scalar),
    started: bool = false,

    /// Returns the next verified JSON payload, or null at end. Header must be
    /// the first line; any frame whose CRC does not match its payload, or that
    /// lacks the `<crc> <json>` shape, fails with `error.WalCorruption`.
    fn next(self: *FrameIterator) !?[]const u8 {
        if (!self.started) {
            const header = self.lines.next() orelse return error.InvalidHeader;
            if (!std.mem.eql(u8, std.mem.trim(u8, header, " \t\r"), WAL_HEADER)) return error.InvalidHeader;
            self.started = true;
        }
        while (self.lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0) continue;
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.WalCorruption;
            const crc = line[0..sp];
            const json = line[sp + 1 ..];
            if (json.len == 0) return error.WalCorruption;
            const expect = crc32Hex(json);
            if (!std.mem.eql(u8, crc, &expect)) return error.WalCorruption;
            return json;
        }
        return null;
    }
};

fn frameIterator(content: []const u8) FrameIterator {
    return .{ .lines = std.mem.splitScalar(u8, content, '\n') };
}

fn deleteTestFileIfExists(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.debug.print("failed to delete test file '{s}': {s}\n", .{ path, @errorName(err) }),
    };
}

test "wal: append and replay reconstruct kv + block state" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-replay.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "agent:abbey", "trained");
    try appendKv(std.testing.io, allocator, path, "agent:aviva", "creative");
    try appendBlock(std.testing.io, allocator, path, "abbey", 1, 2, "{\"turn\":1}", 1000);
    try appendBlock(std.testing.io, allocator, path, "aviva", 3, 4, "{\"turn\":2}", 2000);
    try appendTemporalNode(std.testing.io, allocator, path, 1, 1000);
    try appendTemporalNode(std.testing.io, allocator, path, 2, 2000);
    try appendTemporalEdge(std.testing.io, allocator, path, 1, 2);

    try std.testing.expectEqual(@as(usize, 7), try verify(std.testing.io, allocator, path));

    var store = try replay(std.testing.io, allocator, path);
    defer store.deinit();

    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expectEqualStrings("trained", store.get("agent:abbey").?);
    try std.testing.expectEqual(@as(usize, 2), store.blockCount());
    try std.testing.expect(store.verifyBlocks());
    const last = store.lastBlock().?;
    try std.testing.expectEqual(@as(i64, 2000), last.timestamp_ms);
    try std.testing.expectEqual(@as(usize, 2), store.temporalNodeCount());
    try std.testing.expectEqual(@as(usize, 1), store.temporalEdgeCount());
    try std.testing.expectEqual(@as(?i64, 1000), store.temporalTimestamp(1));
}

test "wal: detects a flipped byte in a framed record" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-corrupt.wal";
    defer deleteTestFileIfExists(path);
    deleteTestFileIfExists(path);

    try appendKv(std.testing.io, allocator, path, "k1", "v1");

    // Flip a byte inside the JSON payload; CRC must reject it on replay.
    const content = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024));
    defer allocator.free(content);
    // Target the payload key "k1" (the header version string also contains
    // "v1", so flip a byte that only exists inside a framed record).
    const idx = std.mem.indexOf(u8, content, "k1").?;
    content[idx] = 'X';
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = content });

    try std.testing.expectError(error.WalCorruption, replay(std.testing.io, allocator, path));
    try std.testing.expectError(error.WalCorruption, verify(std.testing.io, allocator, path));
}

test "wal: rejects a missing or wrong header" {
    const allocator = std.testing.allocator;
    const path = "zig-out/wdbx-wal-badheader.wal";
    defer deleteTestFileIfExists(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = "garbage\n{\"type\":\"kv\"}\n" });
    try std.testing.expectError(error.InvalidHeader, replay(std.testing.io, allocator, path));
}

test {
    std.testing.refAllDecls(@This());
}
