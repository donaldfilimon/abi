//! Produces compact packets for the inference engine.

const std = @import("std");

pub const ContextPacket = struct {
    // TODO: emit ContextPacket
};

pub const ContextAssembler = struct {
    // TODO: gather candidates, trim to token/byte budget, preserve important lineage, optionally summarize oversized groups
};
