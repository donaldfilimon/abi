//! Canonical entity identifiers.

const std = @import("std");

pub const BlockId = struct {
    id: [32]u8,
};

pub const ShardId = struct {
    id: u32,
};

pub const NodeId = struct {
    id: u32,
};

pub const TraceId = struct {
    id: u64,
};
