//! RPC Serialization
//!
//! Request/response serialization helpers for Raft consensus and database operations.

const std = @import("std");
const protocol = @import("protocol.zig");

const TransportError = protocol.TransportError;

/// RPC request/response serialization helpers.
pub const RpcSerializer = struct {
    /// Serialize a Raft RequestVote request.
    pub fn serializeVoteRequest(
        allocator: std.mem.Allocator,
        term: u64,
        candidate_id: []const u8,
        last_log_index: u64,
        last_log_term: u64,
    ) ![]u8 {
        const header_size = 8 + 4 + 8 + 8; // term + id_len + last_log_index + last_log_term
        const total_size = header_size + candidate_id.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], term, .little);
        offset += 8;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(candidate_id.len), .little);
        offset += 4;
        std.mem.writeInt(u64, buffer[offset..][0..8], last_log_index, .little);
        offset += 8;
        std.mem.writeInt(u64, buffer[offset..][0..8], last_log_term, .little);
        offset += 8;
        @memcpy(buffer[offset..][0..candidate_id.len], candidate_id);

        return buffer;
    }

    /// Deserialize a Raft RequestVote request.
    pub fn deserializeVoteRequest(data: []const u8) !struct {
        term: u64,
        candidate_id: []const u8,
        last_log_index: u64,
        last_log_term: u64,
    } {
        if (data.len < 28) return TransportError.InvalidMessage;

        var offset: usize = 0;
        const term = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;
        const id_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        const last_log_index = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;
        const last_log_term = std.mem.readInt(u64, data[offset..][0..8], .little);
        offset += 8;

        if (offset + id_len > data.len) return TransportError.InvalidMessage;
        const candidate_id = data[offset..][0..id_len];

        return .{
            .term = term,
            .candidate_id = candidate_id,
            .last_log_index = last_log_index,
            .last_log_term = last_log_term,
        };
    }

    /// Serialize a Raft RequestVote response.
    pub fn serializeVoteResponse(
        allocator: std.mem.Allocator,
        term: u64,
        vote_granted: bool,
        voter_id: []const u8,
    ) ![]u8 {
        const header_size = 8 + 1 + 4; // term + vote_granted + id_len
        const total_size = header_size + voter_id.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], term, .little);
        offset += 8;
        buffer[offset] = if (vote_granted) 1 else 0;
        offset += 1;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(voter_id.len), .little);
        offset += 4;
        @memcpy(buffer[offset..][0..voter_id.len], voter_id);

        return buffer;
    }

    /// Serialize a database search request.
    pub fn serializeSearchRequest(
        allocator: std.mem.Allocator,
        query_vector: []const f32,
        top_k: u32,
        shard_id: u32,
    ) ![]u8 {
        const header_size = 4 + 4 + 4; // vector_len + top_k + shard_id
        const vector_bytes = query_vector.len * @sizeOf(f32);
        const total_size = header_size + vector_bytes;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(query_vector.len), .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], top_k, .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], shard_id, .little);
        offset += 4;

        const vector_slice = std.mem.sliceAsBytes(query_vector);
        @memcpy(buffer[offset..][0..vector_bytes], vector_slice);

        return buffer;
    }

    /// Serialize a database insert request.
    pub fn serializeInsertRequest(
        allocator: std.mem.Allocator,
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8,
    ) ![]u8 {
        const meta = metadata orelse "";
        const header_size = 8 + 4 + 4; // id + vector_len + meta_len
        const vector_bytes = vector.len * @sizeOf(f32);
        const total_size = header_size + vector_bytes + meta.len;

        var buffer = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        std.mem.writeInt(u64, buffer[offset..][0..8], id, .little);
        offset += 8;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(vector.len), .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(meta.len), .little);
        offset += 4;

        const vector_slice = std.mem.sliceAsBytes(vector);
        @memcpy(buffer[offset..][0..vector_bytes], vector_slice);
        offset += vector_bytes;

        if (meta.len > 0) {
            @memcpy(buffer[offset..][0..meta.len], meta);
        }

        return buffer;
    }
};

test "rpc serialization vote request" {
    const allocator = std.testing.allocator;

    const data = try RpcSerializer.serializeVoteRequest(allocator, 5, "node-1", 10, 3);
    defer allocator.free(data);

    const parsed = try RpcSerializer.deserializeVoteRequest(data);
    try std.testing.expectEqual(@as(u64, 5), parsed.term);
    try std.testing.expectEqualStrings("node-1", parsed.candidate_id);
    try std.testing.expectEqual(@as(u64, 10), parsed.last_log_index);
    try std.testing.expectEqual(@as(u64, 3), parsed.last_log_term);
}
