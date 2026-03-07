const std = @import("std");

pub const MAGIC: [4]u8 = .{ 'A', 'B', 'I', 'R' };
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 23;

pub const MessageType = enum(u8) {
    heartbeat = 0x01,
    heartbeat_ack = 0x02,
    request_vote = 0x10,
    request_vote_response = 0x11,
    append_entries = 0x12,
    append_entries_response = 0x13,
    install_snapshot = 0x14,
    install_snapshot_response = 0x15,
    block_replicate = 0x20,
    block_replicate_ack = 0x21,
    block_request = 0x22,
    block_response = 0x23,
    block_sync_request = 0x24,
    block_sync_response = 0x25,
    version_vector_exchange = 0x30,
    version_vector_response = 0x31,
};

pub const RpcHeader = struct {
    magic: [4]u8,
    version: u16,
    msg_type: MessageType,
    payload_len: u32,
    request_id: u64,
    crc32: u32,

    pub fn validate(_: *const RpcHeader, _: []const u8) bool {
        return false;
    }

    pub fn serialize(_: *const RpcHeader, _: *[HEADER_SIZE]u8) void {}

    pub fn deserialize(_: *const [HEADER_SIZE]u8) RpcHeader {
        return .{
            .magic = MAGIC,
            .version = VERSION,
            .msg_type = .heartbeat,
            .payload_len = 0,
            .request_id = 0,
            .crc32 = 0,
        };
    }
};

pub const ParsedFrame = struct {
    header: RpcHeader,
    payload: []const u8,
    consumed: usize,
};

pub const VectorEntry = struct {
    node_id: []const u8,
    timestamp: i64,
};

pub const BlockHeader = struct {
    block_hash: u64,
    timestamp: i64,
    commit_timestamp: i64,
    data: []const u8,
};

pub const RpcError = error{
    InvalidMessage,
    InvalidMagic,
    CrcMismatch,
    UnsupportedVersion,
    BufferTooSmall,
};

pub fn frameMessage(_: std.mem.Allocator, _: MessageType, _: u64, _: []const u8) error{FeatureDisabled}![]u8 {
    return error.FeatureDisabled;
}

pub fn parseFrame(_: []const u8) ?ParsedFrame {
    return null;
}

pub fn serializeRequestVote(_: std.mem.Allocator, _: anytype) error{FeatureDisabled}![]u8 {
    return error.FeatureDisabled;
}

pub fn deserializeRequestVote(_: []const u8) error{InvalidMessage}!struct {
    term: u64,
    candidate_id: []const u8,
    last_log_index: u64,
    last_log_term: u64,
    is_pre_vote: bool,
} {
    return error.InvalidMessage;
}

pub fn serializeVersionVector(
    _: std.mem.Allocator,
    _: []const VectorEntry,
) error{FeatureDisabled}![]u8 {
    return error.FeatureDisabled;
}

pub fn deserializeVersionVector(_: []const u8) !struct { entries: []const VectorEntry, raw: []const u8 } {
    return .{ .entries = &.{}, .raw = &.{} };
}

pub fn deserializeVersionVectorInto(_: []const u8, _: []VectorEntry) error{InvalidMessage}!u32 {
    return error.InvalidMessage;
}

pub fn serializeBlock(
    _: std.mem.Allocator,
    _: u64,
    _: i64,
    _: i64,
    _: []const u8,
) error{FeatureDisabled}![]u8 {
    return error.FeatureDisabled;
}

pub fn deserializeBlock(_: []const u8) error{InvalidMessage}!BlockHeader {
    return error.InvalidMessage;
}

test {
    std.testing.refAllDecls(@This());
}
