//! Binary RPC Protocol for ABI Distributed Engine
//!
//! Wire format for inter-node communication. All multi-byte fields are
//! little-endian. Each frame starts with a fixed 23-byte header:
//!
//!   [4 magic "ABIR"][2 version][1 msg_type][4 payload_len][8 request_id][4 crc32]
//!
//! The CRC32 covers the payload bytes only (IEEE polynomial).

const std = @import("std");

/// Magic bytes identifying an ABI RPC frame.
pub const MAGIC: [4]u8 = .{ 'A', 'B', 'I', 'R' };

/// Current protocol version.
pub const VERSION: u16 = 1;

/// Header size in bytes.
pub const HEADER_SIZE: usize = 23;

/// Message type identifiers.
pub const MessageType = enum(u8) {
    // Heartbeat
    heartbeat = 0x01,
    heartbeat_ack = 0x02,

    // Raft consensus (0x10-0x15)
    request_vote = 0x10,
    request_vote_response = 0x11,
    append_entries = 0x12,
    append_entries_response = 0x13,
    install_snapshot = 0x14,
    install_snapshot_response = 0x15,

    // Block replication (0x20-0x25)
    block_replicate = 0x20,
    block_replicate_ack = 0x21,
    block_request = 0x22,
    block_response = 0x23,
    block_sync_request = 0x24,
    block_sync_response = 0x25,

    // Anti-entropy (0x30-0x31)
    version_vector_exchange = 0x30,
    version_vector_response = 0x31,
};

/// Fixed-size frame header (23 bytes).
pub const RpcHeader = struct {
    magic: [4]u8,
    version: u16,
    msg_type: MessageType,
    payload_len: u32,
    request_id: u64,
    crc32: u32,

    /// Validate header magic and CRC against payload.
    pub fn validate(self: *const RpcHeader, payload: []const u8) bool {
        if (!std.mem.eql(u8, &self.magic, &MAGIC)) return false;
        if (payload.len != self.payload_len) return false;
        const computed = std.hash.crc.Crc32IsoHdlc.hash(payload);
        return computed == self.crc32;
    }

    /// Serialize header to bytes.
    pub fn serialize(self: *const RpcHeader, buf: *[HEADER_SIZE]u8) void {
        @memcpy(buf[0..4], &self.magic);
        std.mem.writeInt(u16, buf[4..6], self.version, .little);
        buf[6] = @intFromEnum(self.msg_type);
        std.mem.writeInt(u32, buf[7..11], self.payload_len, .little);
        std.mem.writeInt(u64, buf[11..19], self.request_id, .little);
        std.mem.writeInt(u32, buf[19..23], self.crc32, .little);
    }

    /// Deserialize header from bytes.
    pub fn deserialize(buf: *const [HEADER_SIZE]u8) RpcHeader {
        return .{
            .magic = buf[0..4].*,
            .version = std.mem.readInt(u16, buf[4..6], .little),
            .msg_type = @enumFromInt(buf[6]),
            .payload_len = std.mem.readInt(u32, buf[7..11], .little),
            .request_id = std.mem.readInt(u64, buf[11..19], .little),
            .crc32 = std.mem.readInt(u32, buf[19..23], .little),
        };
    }
};

/// Frame a message: header + payload → contiguous byte slice.
pub fn frameMessage(
    allocator: std.mem.Allocator,
    msg_type: MessageType,
    request_id: u64,
    payload: []const u8,
) ![]u8 {
    const frame = try allocator.alloc(u8, HEADER_SIZE + payload.len);
    errdefer allocator.free(frame);

    const crc = std.hash.crc.Crc32IsoHdlc.hash(payload);

    const header = RpcHeader{
        .magic = MAGIC,
        .version = VERSION,
        .msg_type = msg_type,
        .payload_len = @intCast(payload.len),
        .request_id = request_id,
        .crc32 = crc,
    };

    header.serialize(frame[0..HEADER_SIZE]);
    if (payload.len > 0) {
        @memcpy(frame[HEADER_SIZE..], payload);
    }

    return frame;
}

/// Parse result from parseFrame.
pub const ParsedFrame = struct {
    header: RpcHeader,
    payload: []const u8,
    /// Total bytes consumed (header + payload).
    consumed: usize,
};

/// Maximum frame size (header + payload): 64 MiB.
pub const MAX_FRAME_SIZE: usize = 64 << 20;

/// Parse a frame from a buffer. Returns null if insufficient data.
pub fn parseFrame(buf: []const u8) ?ParsedFrame {
    if (buf.len < HEADER_SIZE) return null;

    const header = RpcHeader.deserialize(buf[0..HEADER_SIZE]);
    const total = HEADER_SIZE + header.payload_len;

    if (total > MAX_FRAME_SIZE) return null;
    if (buf.len < total) return null;

    const payload = buf[HEADER_SIZE..total];

    return .{
        .header = header,
        .payload = payload,
        .consumed = total,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// Raft Message Serialization
// ═══════════════════════════════════════════════════════════════════════

/// Serialize a RequestVote message to bytes.
pub fn serializeRequestVote(allocator: std.mem.Allocator, req: anytype) ![]u8 {
    // Layout: [8 term][2 id_len][N candidate_id][8 last_log_index][8 last_log_term][1 is_pre_vote]
    const id = req.candidate_id;
    const size = 8 + 2 + id.len + 8 + 8 + 1;
    const buf = try allocator.alloc(u8, size);
    var offset: usize = 0;

    std.mem.writeInt(u64, buf[offset..][0..8], req.term, .little);
    offset += 8;
    std.mem.writeInt(u16, buf[offset..][0..2], @intCast(id.len), .little);
    offset += 2;
    @memcpy(buf[offset .. offset + id.len], id);
    offset += id.len;
    std.mem.writeInt(u64, buf[offset..][0..8], req.last_log_index, .little);
    offset += 8;
    std.mem.writeInt(u64, buf[offset..][0..8], req.last_log_term, .little);
    offset += 8;
    buf[offset] = if (req.is_pre_vote) 1 else 0;

    return buf;
}

/// Deserialize a RequestVote message from bytes.
pub fn deserializeRequestVote(buf: []const u8) !struct {
    term: u64,
    candidate_id: []const u8,
    last_log_index: u64,
    last_log_term: u64,
    is_pre_vote: bool,
} {
    if (buf.len < 27) return error.InvalidMessage;
    var offset: usize = 0;

    const term = std.mem.readInt(u64, buf[offset..][0..8], .little);
    offset += 8;
    const id_len = std.mem.readInt(u16, buf[offset..][0..2], .little);
    offset += 2;
    if (buf.len < offset + id_len + 17) return error.InvalidMessage;
    const candidate_id = buf[offset .. offset + id_len];
    offset += id_len;
    const last_log_index = std.mem.readInt(u64, buf[offset..][0..8], .little);
    offset += 8;
    const last_log_term = std.mem.readInt(u64, buf[offset..][0..8], .little);
    offset += 8;
    const is_pre_vote = buf[offset] != 0;

    return .{
        .term = term,
        .candidate_id = candidate_id,
        .last_log_index = last_log_index,
        .last_log_term = last_log_term,
        .is_pre_vote = is_pre_vote,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// Version Vector Serialization
// ═══════════════════════════════════════════════════════════════════════

/// Serialize a version vector (list of node_id:timestamp pairs).
/// Layout: [4 entry_count] [per entry: [2 id_len][N id][8 timestamp]]
pub fn serializeVersionVector(
    allocator: std.mem.Allocator,
    entries: []const VectorEntry,
) ![]u8 {
    var size: usize = 4;
    for (entries) |e| {
        size += 2 + e.node_id.len + 8;
    }

    const buf = try allocator.alloc(u8, size);
    var offset: usize = 0;

    std.mem.writeInt(u32, buf[offset..][0..4], @intCast(entries.len), .little);
    offset += 4;

    for (entries) |e| {
        std.mem.writeInt(u16, buf[offset..][0..2], @intCast(e.node_id.len), .little);
        offset += 2;
        @memcpy(buf[offset .. offset + e.node_id.len], e.node_id);
        offset += e.node_id.len;
        std.mem.writeInt(i64, buf[offset..][0..8], e.timestamp, .little);
        offset += 8;
    }

    return buf;
}

/// Deserialized version vector entry.
pub const VectorEntry = struct {
    node_id: []const u8,
    timestamp: i64,
};

/// Deserialize version vector entries (slices point into buf).
pub fn deserializeVersionVector(buf: []const u8) !struct { entries: []const VectorEntry, raw: []const u8 } {
    _ = buf;
    // Returns empty for now — full implementation would parse the binary format.
    // Callers should use the round-trip test pattern.
    return .{ .entries = &.{}, .raw = &.{} };
}

/// Deserialize version vector into pre-allocated slice.
pub fn deserializeVersionVectorInto(buf: []const u8, out: []VectorEntry) !u32 {
    if (buf.len < 4) return error.InvalidMessage;

    const count = std.mem.readInt(u32, buf[0..4], .little);
    var offset: usize = 4;
    var i: u32 = 0;

    while (i < count and i < out.len) : (i += 1) {
        if (buf.len < offset + 2) return error.InvalidMessage;
        const id_len = std.mem.readInt(u16, buf[offset..][0..2], .little);
        offset += 2;
        if (buf.len < offset + id_len + 8) return error.InvalidMessage;
        out[i] = .{
            .node_id = buf[offset .. offset + id_len],
            .timestamp = std.mem.readInt(i64, buf[offset + id_len ..][0..8], .little),
        };
        offset += id_len + 8;
    }

    return i;
}

// ═══════════════════════════════════════════════════════════════════════
// Block Serialization
// ═══════════════════════════════════════════════════════════════════════

/// Serialize a block header for replication.
/// Layout: [8 block_hash][8 timestamp][8 commit_timestamp][4 data_len][N data]
pub fn serializeBlock(
    allocator: std.mem.Allocator,
    block_hash: u64,
    timestamp: i64,
    commit_timestamp: i64,
    data: []const u8,
) ![]u8 {
    const size = 8 + 8 + 8 + 4 + data.len;
    const buf = try allocator.alloc(u8, size);

    std.mem.writeInt(u64, buf[0..8], block_hash, .little);
    std.mem.writeInt(i64, buf[8..16], timestamp, .little);
    std.mem.writeInt(i64, buf[16..24], commit_timestamp, .little);
    std.mem.writeInt(u32, buf[24..28], @intCast(data.len), .little);
    if (data.len > 0) {
        @memcpy(buf[28..], data);
    }

    return buf;
}

/// Deserialized block header.
pub const BlockHeader = struct {
    block_hash: u64,
    timestamp: i64,
    commit_timestamp: i64,
    data: []const u8,
};

/// Deserialize a block from bytes.
pub fn deserializeBlock(buf: []const u8) !BlockHeader {
    if (buf.len < 28) return error.InvalidMessage;

    const data_len = std.mem.readInt(u32, buf[24..28], .little);
    if (buf.len < 28 + data_len) return error.InvalidMessage;

    return .{
        .block_hash = std.mem.readInt(u64, buf[0..8], .little),
        .timestamp = std.mem.readInt(i64, buf[8..16], .little),
        .commit_timestamp = std.mem.readInt(i64, buf[16..24], .little),
        .data = buf[28 .. 28 + data_len],
    };
}

// Error type
pub const RpcError = error{
    InvalidMessage,
    InvalidMagic,
    CrcMismatch,
    UnsupportedVersion,
    BufferTooSmall,
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "frame round-trip" {
    const allocator = std.testing.allocator;
    const payload = "hello world";

    const frame = try frameMessage(allocator, .heartbeat, 42, payload);
    defer allocator.free(frame);

    const parsed = parseFrame(frame) orelse return error.TestUnexpectedResult;
    try std.testing.expect(parsed.header.validate(parsed.payload));
    try std.testing.expectEqual(MessageType.heartbeat, parsed.header.msg_type);
    try std.testing.expectEqual(@as(u64, 42), parsed.header.request_id);
    try std.testing.expectEqualStrings("hello world", parsed.payload);
}

test "CRC validation: corrupt payload byte" {
    const allocator = std.testing.allocator;
    const payload = "test data";

    const frame = try frameMessage(allocator, .request_vote, 1, payload);
    defer allocator.free(frame);

    // Corrupt one byte in payload
    frame[HEADER_SIZE + 2] ^= 0xFF;

    const parsed = parseFrame(frame) orelse return error.TestUnexpectedResult;
    try std.testing.expect(!parsed.header.validate(parsed.payload));
}

test "multi-frame parsing: concatenated frames" {
    const allocator = std.testing.allocator;

    const frame1 = try frameMessage(allocator, .heartbeat, 1, "first");
    defer allocator.free(frame1);
    const frame2 = try frameMessage(allocator, .append_entries, 2, "second");
    defer allocator.free(frame2);

    // Concatenate
    const combined = try allocator.alloc(u8, frame1.len + frame2.len);
    defer allocator.free(combined);
    @memcpy(combined[0..frame1.len], frame1);
    @memcpy(combined[frame1.len..], frame2);

    // Parse first
    const p1 = parseFrame(combined) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(MessageType.heartbeat, p1.header.msg_type);
    try std.testing.expectEqualStrings("first", p1.payload);

    // Parse second
    const p2 = parseFrame(combined[p1.consumed..]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(MessageType.append_entries, p2.header.msg_type);
    try std.testing.expectEqualStrings("second", p2.payload);
}

test "RequestVote round-trip" {
    const allocator = std.testing.allocator;
    const req = .{
        .term = @as(u64, 5),
        .candidate_id = @as([]const u8, "node-42"),
        .last_log_index = @as(u64, 100),
        .last_log_term = @as(u64, 4),
        .is_pre_vote = true,
    };

    const buf = try serializeRequestVote(allocator, req);
    defer allocator.free(buf);

    const parsed = try deserializeRequestVote(buf);
    try std.testing.expectEqual(@as(u64, 5), parsed.term);
    try std.testing.expectEqualStrings("node-42", parsed.candidate_id);
    try std.testing.expectEqual(@as(u64, 100), parsed.last_log_index);
    try std.testing.expectEqual(@as(u64, 4), parsed.last_log_term);
    try std.testing.expect(parsed.is_pre_vote);
}

test "block round-trip" {
    const allocator = std.testing.allocator;
    const data = "block content data";

    const buf = try serializeBlock(allocator, 0xDEADBEEF, 1000, 1001, data);
    defer allocator.free(buf);

    const block = try deserializeBlock(buf);
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF), block.block_hash);
    try std.testing.expectEqual(@as(i64, 1000), block.timestamp);
    try std.testing.expectEqual(@as(i64, 1001), block.commit_timestamp);
    try std.testing.expectEqualStrings("block content data", block.data);
}

test "VersionVector round-trip with 3 entries" {
    const allocator = std.testing.allocator;

    const entries = [_]VectorEntry{
        .{ .node_id = "node-a", .timestamp = 100 },
        .{ .node_id = "node-b", .timestamp = 200 },
        .{ .node_id = "node-c", .timestamp = 300 },
    };

    const buf = try serializeVersionVector(allocator, &entries);
    defer allocator.free(buf);

    var out: [8]VectorEntry = undefined;
    const count = try deserializeVersionVectorInto(buf, &out);
    try std.testing.expectEqual(@as(u32, 3), count);
    try std.testing.expectEqualStrings("node-a", out[0].node_id);
    try std.testing.expectEqual(@as(i64, 100), out[0].timestamp);
    try std.testing.expectEqualStrings("node-b", out[1].node_id);
    try std.testing.expectEqual(@as(i64, 200), out[1].timestamp);
    try std.testing.expectEqualStrings("node-c", out[2].node_id);
    try std.testing.expectEqual(@as(i64, 300), out[2].timestamp);
}

test "empty payload frame" {
    const allocator = std.testing.allocator;
    const frame = try frameMessage(allocator, .heartbeat_ack, 99, "");
    defer allocator.free(frame);

    try std.testing.expectEqual(HEADER_SIZE, frame.len);

    const parsed = parseFrame(frame) orelse return error.TestUnexpectedResult;
    try std.testing.expect(parsed.header.validate(parsed.payload));
    try std.testing.expectEqual(@as(u32, 0), parsed.header.payload_len);
}

test "insufficient data returns null" {
    const short: [10]u8 = undefined;
    try std.testing.expect(parseFrame(&short) == null);
}

test {
    std.testing.refAllDecls(@This());
}
