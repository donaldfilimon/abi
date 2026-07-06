const std = @import("std");
const testing = std.testing;

pub const PoolAllocator = struct {
    backing_allocator: std.mem.Allocator,
    block_size: usize,
    free_list: ?*Node = null,
    chunks: std.ArrayListUnmanaged([]align(@alignOf(Node)) u8) = .empty,

    const Node = struct {
        next: ?*Node,
    };

    pub fn init(allocator: std.mem.Allocator, block_size: usize) PoolAllocator {
        return .{
            .backing_allocator = allocator,
            .block_size = std.mem.alignForward(usize, @max(block_size, @sizeOf(Node)), @alignOf(Node)),
            .chunks = .empty,
        };
    }

    pub fn deinit(self: *PoolAllocator) void {
        for (self.chunks.items) |chunk| {
            self.backing_allocator.free(chunk);
        }
        self.chunks.deinit(self.backing_allocator);
    }

    pub fn alloc(self: *PoolAllocator) ![]u8 {
        if (self.free_list) |node| {
            self.free_list = node.next;
            const ptr = @as([*]u8, @ptrCast(node));
            return ptr[0..self.block_size];
        }

        // Allocate a new chunk
        const chunk_blocks = 64;
        const chunk_size = self.block_size * chunk_blocks;
        const chunk = try self.backing_allocator.alignedAlloc(u8, .fromByteUnits(@alignOf(Node)), chunk_size);
        errdefer self.backing_allocator.free(chunk);

        try self.chunks.append(self.backing_allocator, chunk);

        // Add blocks (except the first one) to the free list
        var i: usize = 1;
        while (i < chunk_blocks) : (i += 1) {
            const block_ptr = chunk.ptr + (i * self.block_size);
            const node = @as(*Node, @ptrCast(@alignCast(block_ptr)));
            node.next = self.free_list;
            self.free_list = node;
        }

        return chunk[0..self.block_size];
    }

    pub fn free(self: *PoolAllocator, block: []u8) void {
        const node = @as(*Node, @ptrCast(@alignCast(block.ptr)));
        node.next = self.free_list;
        self.free_list = node;
    }
};

test "PoolAllocator: single block allocation and deallocation" {
    var pool = PoolAllocator.init(testing.allocator, 16);
    defer pool.deinit();

    const block = try pool.alloc();
    try testing.expect(block.len == 16);
    pool.free(block);
}

test "PoolAllocator: multiple allocations and reuse" {
    var pool = PoolAllocator.init(testing.allocator, 32);
    defer pool.deinit();

    var blocks: [100][]u8 = undefined;
    for (&blocks) |*b| {
        b.* = try pool.alloc();
        try testing.expect(b.len == 32);
    }

    // Free some and reuse
    pool.free(blocks[0]);
    pool.free(blocks[1]);

    const b1 = try pool.alloc();
    const b2 = try pool.alloc();

    try testing.expect(b1.ptr == blocks[1].ptr or b1.ptr == blocks[0].ptr);
    try testing.expect(b2.ptr == blocks[1].ptr or b2.ptr == blocks[0].ptr);
}

test {
    testing.refAllDecls(@This());
}
