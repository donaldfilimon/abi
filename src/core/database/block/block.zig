//! StoredBlock and payload views.

const std = @import("std");
const header = @import("header.zig");

pub const StoredBlock = struct {
    header: header.BlockHeader,
    payload: []const u8,
};
