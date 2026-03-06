//! Public BlockStore API.

const std = @import("std");
const block = @import("block.zig");
const core = @import("../core/mod.zig");

pub const BlockStore = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BlockStore {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BlockStore) void {
        _ = self;
    }

    pub fn put(self: *BlockStore, b: block.StoredBlock) !void {
        _ = self;
        _ = b;
        unreachable; // TODO
    }

    pub fn get(self: *BlockStore, id: core.ids.BlockId) !?block.StoredBlock {
        _ = self;
        _ = id;
        unreachable; // TODO
    }
};
