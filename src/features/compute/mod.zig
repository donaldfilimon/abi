//! Omni-Compute Module
//!
//! Provides the distributed mesh networking, multi-GPU orchestration,
//! and tensor sharing protocols.

pub const mesh = @import("mesh");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
