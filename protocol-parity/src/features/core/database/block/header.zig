//! Versioned block headers.

const std = @import("std");
const core = @import("../../mod.zig");

pub const BlockHeader = struct {
    id: core.ids.BlockId,
    kind: core.types.BlockKind,
    version: u32,
    content_hash: [32]u8,
    timestamp: core.time.LogicalClock,
    size: u32,
    flags: u16,
    compression_marker: u8,
};
