//! Time-range or ordered retrieval and direct lookup.

const std = @import("std");
const core = @import("../core");

pub const BTreeIndex = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BTreeIndex {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BTreeIndex) void {
        _ = self;
    }
};

pub const HashIndex = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(core.ids.BlockId, u64),

    pub fn init(allocator: std.mem.Allocator) HashIndex {
        return .{
            .allocator = allocator,
            .map = .empty,
        };
    }

    pub fn deinit(self: *HashIndex) void {
        self.map.deinit(self.allocator);
    }

    pub fn put(self: *HashIndex, id: core.ids.BlockId, offset: u64) !void {
        try self.map.put(self.allocator, id, offset);
    }

    pub fn get(self: *const HashIndex, id: core.ids.BlockId) ?u64 {
        return self.map.get(id);
    }
};
