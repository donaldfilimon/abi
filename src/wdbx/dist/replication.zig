//! In-process block sync path using dist.rpc codec.
//!
//! Request side: encode BlockSyncRequest, then decode a response stream (BlockSyncResponse + zero or more BlockChunk).
//! No network I/O; callers pass in-memory buffers. Use for tests and as the core of a future TCP transport.

const std = @import("std");
const rpc = @import("rpc.zig");

/// Optional trace callback for sync steps (e.g. "request shard=0", "response status=ok", "chunk 0 len=256").
pub const TraceFn = *const fn (msg: []const u8) void;

/// Result of running the requester side of a block sync.
pub const SyncResult = struct {
    status: rpc.BlockSyncResponse.BlockSyncStatus,
    total_byte_len: u32,
    chunks_received: u32,
    bytes_received: u32,
};

/// Run the requester path: parse response_bytes as a stream of rpc messages (one BlockSyncResponse, then zero or more BlockChunk).
/// Append chunk payloads to sink. Call trace with log messages when trace is non-null.
pub fn runRequesterPath(
    allocator: std.mem.Allocator,
    request: rpc.BlockSyncRequest,
    response_bytes: []const u8,
    sink: *std.ArrayListUnmanaged(u8),
    trace: ?TraceFn,
) !SyncResult {
    var cursor: usize = 0;
    var result = SyncResult{
        .status = .not_found,
        .total_byte_len = 0,
        .chunks_received = 0,
        .bytes_received = 0,
    };

    if (trace) |t| {
        t("sync request");
    }
    _ = request;

    if (cursor + rpc.Header.encoded_len > response_bytes.len) {
        if (trace) |t| t("response truncated");
        return result;
    }
    const h = try rpc.decodeHeader(response_bytes[cursor..]);
    cursor += rpc.Header.encoded_len;
    if (h.msg_type != .block_sync_response) {
        if (trace) |t| t("expected block_sync_response");
        return result;
    }
    if (cursor + h.payload_len > response_bytes.len) {
        if (trace) |t| t("response payload truncated");
        return result;
    }
    const resp = try rpc.BlockSyncResponse.decode(response_bytes[cursor..][0..h.payload_len]);
    cursor += h.payload_len;
    result.status = resp.status;
    result.total_byte_len = resp.total_byte_len;
    if (trace) |t| {
        var buf: [64]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "response status={s} total={d}", .{
            @tagName(resp.status),
            resp.total_byte_len,
        }) catch "response ok";
        t(msg);
    }
    if (resp.status != .ok) return result;

    const MAX_PAYLOAD = 16 << 20; // 16 MiB
    const MAX_CHUNKS = 10_000;
    while (cursor + rpc.Header.encoded_len <= response_bytes.len) {
        if (result.chunks_received >= MAX_CHUNKS) break;
        const ch = try rpc.decodeHeader(response_bytes[cursor..]);
        cursor += rpc.Header.encoded_len;
        if (ch.msg_type != .block_chunk) break;
        if (ch.payload_len > MAX_PAYLOAD) break;
        if (cursor + ch.payload_len > response_bytes.len) break;
        const chunk = try rpc.BlockChunk.decodeHeader(response_bytes[cursor..][0..ch.payload_len]);
        // Overflow-safe bounds check
        if (chunk.chunk_len > std.math.maxInt(u32) - rpc.BlockChunk.fixed_len) break;
        if (ch.payload_len < rpc.BlockChunk.fixed_len + chunk.chunk_len) break;
        if (cursor + rpc.BlockChunk.fixed_len + chunk.chunk_len > response_bytes.len) break;
        const data = response_bytes[cursor + rpc.BlockChunk.fixed_len ..][0..chunk.chunk_len];
        try sink.appendSlice(allocator, data);
        result.chunks_received += 1;
        result.bytes_received += chunk.chunk_len;
        if (trace) |t| {
            var buf: [48]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "chunk {d} len={d}", .{ chunk.chunk_index, chunk.chunk_len }) catch "chunk";
            t(msg);
        }
        cursor += ch.payload_len;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "runRequesterPath: not_found" {
    const allocator = std.testing.allocator;
    var sink = std.ArrayListUnmanaged(u8){};
    defer sink.deinit(allocator);

    var buf: [128]u8 = undefined;
    const resp = rpc.BlockSyncResponse{ .status = .not_found, .total_byte_len = 0 };
    var payload: [rpc.BlockSyncResponse.encoded_len]u8 = undefined;
    try resp.encode(&payload);
    const n = try rpc.encodeMessage(.block_sync_response, &payload, &buf);
    const req = rpc.BlockSyncRequest{ .shard_id = 0, .block_id = [_]u8{0} ** 32 };
    const result = try runRequesterPath(allocator, req, buf[0..n], &sink, null);
    try std.testing.expect(result.status == .not_found);
    try std.testing.expect(result.chunks_received == 0);
    try std.testing.expect(sink.items.len == 0);
}

test "runRequesterPath: ok with one chunk" {
    const allocator = std.testing.allocator;
    var sink = std.ArrayListUnmanaged(u8){};
    defer sink.deinit(allocator);

    const payload_data = [_]u8{ 0xde, 0xad, 0xbe, 0xef };
    var stream = std.ArrayListUnmanaged(u8){};
    defer stream.deinit(allocator);
    var tmp: [256]u8 = undefined;
    const resp = rpc.BlockSyncResponse{ .status = .ok, .total_byte_len = 4 };
    var resp_payload: [rpc.BlockSyncResponse.encoded_len]u8 = undefined;
    try resp.encode(&resp_payload);
    const n1 = try rpc.encodeMessage(.block_sync_response, &resp_payload, &tmp);
    try stream.appendSlice(allocator, tmp[0..n1]);
    const chunk = rpc.BlockChunk{ .chunk_index = 0, .chunk_len = 4 };
    var chunk_payload: [rpc.BlockChunk.fixed_len + 4]u8 = undefined;
    try chunk.encode(&payload_data, &chunk_payload);
    const n2 = try rpc.encodeMessage(.block_chunk, &chunk_payload, &tmp);
    try stream.appendSlice(allocator, tmp[0..n2]);

    const req = rpc.BlockSyncRequest{ .shard_id = 1, .block_id = [_]u8{0} ** 32 };
    const result = try runRequesterPath(allocator, req, stream.items, &sink, null);
    try std.testing.expect(result.status == .ok);
    try std.testing.expect(result.total_byte_len == 4);
    try std.testing.expect(result.chunks_received == 1);
    try std.testing.expect(result.bytes_received == 4);
    try std.testing.expect(sink.items.len == 4);
    try std.testing.expect(std.mem.eql(u8, sink.items, &payload_data));
}

test {
    std.testing.refAllDecls(@This());
}
