//! Public BlockStore API.

const std = @import("std");
const block = @import("block.zig");
const core = @import("../core/mod.zig");

pub const BlockStore = struct {
    allocator: std.mem.Allocator,
    blocks: std.AutoHashMapUnmanaged([32]u8, block.StoredBlock) = .empty,

    pub fn init(allocator: std.mem.Allocator) BlockStore {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BlockStore) void {
        var iter = self.blocks.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.payload);
        }
        self.blocks.deinit(self.allocator);
    }

    pub fn put(self: *BlockStore, b: block.StoredBlock) !void {
        const duped_payload = try self.allocator.dupe(u8, b.payload);
        errdefer self.allocator.free(duped_payload);

        const gop = try self.blocks.getOrPut(self.allocator, b.header.id.id);
        if (gop.found_existing) {
            self.allocator.free(gop.value_ptr.payload);
        }
        gop.value_ptr.* = b;
        gop.value_ptr.payload = duped_payload;
    }

    pub fn get(self: *BlockStore, id: core.ids.BlockId) !?block.StoredBlock {
        if (self.blocks.get(id.id)) |b| {
            const duped_payload = try self.allocator.dupe(u8, b.payload);
            return block.StoredBlock{
                .header = b.header,
                .payload = duped_payload,
            };
        }
        return null;
    }
};
