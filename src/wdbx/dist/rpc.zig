//! Skeleton binary RPC codec for node-to-node block replication.
//!
//! Fixed-format messages: 1-byte type, 4-byte payload length (little-endian),
//! then payload. Encode/decode only; no I/O.

const std = @import("std");

pub const MessageType = enum(u8) {
    heartbeat = 0,
    block_sync_request = 1,
    block_sync_response = 2,
    block_chunk = 3,
};

pub const Header = struct {
    msg_type: MessageType,
    payload_len: u32,

    pub const encoded_len = 5;

    pub fn encode(self: Header, out: []u8) !void {
        if (out.len < encoded_len) return error.BufferTooSmall;
        out[0] = @intFromEnum(self.msg_type);
        std.mem.writeInt(u32, out[1..5], self.payload_len, .little);
    }

    pub fn decode(bytes: []const u8) !Header {
        if (bytes.len < encoded_len) return error.BufferTooSmall;
        const msg_type = @as(MessageType, @enumFromInt(bytes[0]));
        const payload_len = std.mem.readInt(u32, bytes[1..5], .little);
        return .{ .msg_type = msg_type, .payload_len = payload_len };
    }
};

/// Heartbeat payload: node_id (4 bytes), load_factor (4 bytes), timestamp (8 bytes).
pub const HeartbeatPayload = struct {
    node_id: u32,
    load_factor: f32,
    timestamp: i64,

    pub const encoded_len = 16;

    pub fn encode(self: HeartbeatPayload, out: []u8) !void {
        if (out.len < encoded_len) return error.BufferTooSmall;
        std.mem.writeInt(u32, out[0..4], self.node_id, .little);
        std.mem.writeInt(u32, out[4..8], @bitCast(self.load_factor), .little);
        std.mem.writeInt(i64, out[8..16], self.timestamp, .little);
    }

    pub fn decode(bytes: []const u8) !HeartbeatPayload {
        if (bytes.len < encoded_len) return error.BufferTooSmall;
        return .{
            .node_id = std.mem.readInt(u32, bytes[0..4], .little),
            .load_factor = @bitCast(std.mem.readInt(u32, bytes[4..8], .little)),
            .timestamp = std.mem.readInt(i64, bytes[8..16], .little),
        };
    }
};

/// Block sync request: shard_id (4 bytes), block_id (32 bytes).
pub const BlockSyncRequest = struct {
    shard_id: u32,
    block_id: [32]u8,

    pub const encoded_len = 36;

    pub fn encode(self: BlockSyncRequest, out: []u8) !void {
        if (out.len < encoded_len) return error.BufferTooSmall;
        std.mem.writeInt(u32, out[0..4], self.shard_id, .little);
        @memcpy(out[4..36], &self.block_id);
    }

    pub fn decode(bytes: []const u8) !BlockSyncRequest {
        if (bytes.len < encoded_len) return error.BufferTooSmall;
        var block_id: [32]u8 = undefined;
        @memcpy(&block_id, bytes[4..36]);
        return .{
            .shard_id = std.mem.readInt(u32, bytes[0..4], .little),
            .block_id = block_id,
        };
    }
};

/// Block sync response: status (1 byte), total_byte_len (4 bytes). status: 0 = ok, 1 = not_found.
pub const BlockSyncResponse = struct {
    status: BlockSyncStatus,
    total_byte_len: u32,

    pub const encoded_len = 5;

    pub const BlockSyncStatus = enum(u8) {
        ok = 0,
        not_found = 1,
    };

    pub fn encode(self: BlockSyncResponse, out: []u8) !void {
        if (out.len < encoded_len) return error.BufferTooSmall;
        out[0] = @intFromEnum(self.status);
        std.mem.writeInt(u32, out[1..5], self.total_byte_len, .little);
    }

    pub fn decode(bytes: []const u8) !BlockSyncResponse {
        if (bytes.len < encoded_len) return error.BufferTooSmall;
        const status = @as(BlockSyncStatus, @enumFromInt(bytes[0]));
        return .{
            .status = status,
            .total_byte_len = std.mem.readInt(u32, bytes[1..5], .little),
        };
    }
};

/// Block chunk: chunk_index (4 bytes), chunk_len (4 bytes), then data. Payload = fixed 8 bytes + data.
pub const BlockChunk = struct {
    chunk_index: u32,
    /// Length of the data slice following the 8-byte header in the payload.
    chunk_len: u32,

    pub const fixed_len = 8;

    pub fn encode(self: BlockChunk, data: []const u8, out: []u8) !void {
        if (data.len != self.chunk_len) return error.LengthMismatch;
        if (out.len < fixed_len + data.len) return error.BufferTooSmall;
        std.mem.writeInt(u32, out[0..4], self.chunk_index, .little);
        std.mem.writeInt(u32, out[4..8], self.chunk_len, .little);
        @memcpy(out[8..][0..data.len], data);
    }

    /// Decode fixed header; caller uses payload[fixed_len..][0..chunk_len] for data.
    pub fn decodeHeader(payload: []const u8) !BlockChunk {
        if (payload.len < fixed_len) return error.BufferTooSmall;
        const chunk_index = std.mem.readInt(u32, payload[0..4], .little);
        const chunk_len = std.mem.readInt(u32, payload[4..8], .little);
        return .{ .chunk_index = chunk_index, .chunk_len = chunk_len };
    }
};

/// Encode a full message: header + payload.
pub fn encodeMessage(msg_type: MessageType, payload: []const u8, out: []u8) !usize {
    const need = Header.encoded_len + payload.len;
    if (out.len < need) return error.BufferTooSmall;
    const h = Header{ .msg_type = msg_type, .payload_len = @intCast(payload.len) };
    try h.encode(out[0..Header.encoded_len]);
    @memcpy(out[Header.encoded_len..need], payload);
    return need;
}

/// Decode header from the start of a buffer; payload is out[Header.encoded_len..][0..header.payload_len].
pub fn decodeHeader(bytes: []const u8) !Header {
    return Header.decode(bytes);
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "Header encode/decode roundtrip" {
    const h = Header{ .msg_type = .block_sync_request, .payload_len = 36 };
    var buf: [Header.encoded_len]u8 = undefined;
    try h.encode(&buf);
    const decoded = try Header.decode(&buf);
    try std.testing.expectEqual(h.msg_type, decoded.msg_type);
    try std.testing.expectEqual(h.payload_len, decoded.payload_len);
}

test "HeartbeatPayload roundtrip" {
    const p = HeartbeatPayload{
        .node_id = 1,
        .load_factor = 0.75,
        .timestamp = 1000000,
    };
    var buf: [HeartbeatPayload.encoded_len]u8 = undefined;
    try p.encode(&buf);
    const decoded = try HeartbeatPayload.decode(&buf);
    try std.testing.expectEqual(p.node_id, decoded.node_id);
    try std.testing.expect(p.load_factor == decoded.load_factor);
    try std.testing.expectEqual(p.timestamp, decoded.timestamp);
}

test "BlockSyncRequest roundtrip" {
    var block_id: [32]u8 = undefined;
    @memset(&block_id, 0);
    block_id[0] = 0xab;
    const req = BlockSyncRequest{ .shard_id = 2, .block_id = block_id };
    var buf: [BlockSyncRequest.encoded_len]u8 = undefined;
    try req.encode(&buf);
    const decoded = try BlockSyncRequest.decode(&buf);
    try std.testing.expectEqual(req.shard_id, decoded.shard_id);
    try std.testing.expect(std.mem.eql(u8, &req.block_id, &decoded.block_id));
}

test "encodeMessage and decodeHeader" {
    var payload: [HeartbeatPayload.encoded_len]u8 = undefined;
    const hp = HeartbeatPayload{
        .node_id = 3,
        .load_factor = 0.0,
        .timestamp = 99,
    };
    try hp.encode(&payload);

    var out: [128]u8 = undefined;
    const n = try encodeMessage(.heartbeat, &payload, &out);
    try std.testing.expect(n == Header.encoded_len + payload.len);

    const h = try decodeHeader(out[0..]);
    try std.testing.expectEqual(MessageType.heartbeat, h.msg_type);
    try std.testing.expectEqual(@as(u32, HeartbeatPayload.encoded_len), h.payload_len);

    const decoded_payload = try HeartbeatPayload.decode(out[Header.encoded_len..][0..h.payload_len]);
    try std.testing.expectEqual(@as(u32, 3), decoded_payload.node_id);
    try std.testing.expectEqual(@as(i64, 99), decoded_payload.timestamp);
}

test "BlockSyncResponse roundtrip" {
    const r = BlockSyncResponse{ .status = .ok, .total_byte_len = 4096 };
    var buf: [BlockSyncResponse.encoded_len]u8 = undefined;
    try r.encode(&buf);
    const decoded = try BlockSyncResponse.decode(&buf);
    try std.testing.expectEqual(BlockSyncResponse.BlockSyncStatus.ok, decoded.status);
    try std.testing.expectEqual(@as(u32, 4096), decoded.total_byte_len);

    const r2 = BlockSyncResponse{ .status = .not_found, .total_byte_len = 0 };
    try r2.encode(&buf);
    const decoded2 = try BlockSyncResponse.decode(&buf);
    try std.testing.expectEqual(BlockSyncResponse.BlockSyncStatus.not_found, decoded2.status);
    try std.testing.expectEqual(@as(u32, 0), decoded2.total_byte_len);
}

test "BlockChunk encode and decodeHeader" {
    const data: [4]u8 = .{ 0xde, 0xad, 0xbe, 0xef };
    var out: [BlockChunk.fixed_len + 4]u8 = undefined;
    const chunk = BlockChunk{ .chunk_index = 1, .chunk_len = 4 };
    try chunk.encode(&data, &out);
    const dec = try BlockChunk.decodeHeader(&out);
    try std.testing.expectEqual(@as(u32, 1), dec.chunk_index);
    try std.testing.expectEqual(@as(u32, 4), dec.chunk_len);
    try std.testing.expect(std.mem.eql(u8, out[BlockChunk.fixed_len..][0..4], &data));
}

test {
    std.testing.refAllDecls(@This());
}
