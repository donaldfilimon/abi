//! Binary encode and decode logic.

const std = @import("std");
const block = @import("block.zig");

pub fn encodeBlock(allocator: std.mem.Allocator, b: block.StoredBlock) ![]u8 {
    _ = allocator;
    _ = b;
    unreachable; // TODO
}

pub fn decodeBlock(allocator: std.mem.Allocator, data: []const u8) !block.StoredBlock {
    _ = allocator;
    _ = data;
    unreachable; // TODO
}
