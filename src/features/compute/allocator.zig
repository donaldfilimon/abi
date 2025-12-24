const std = @import("std");

pub const FixedBlockPool = struct {
    allocator: std.mem.Allocator,
    block_size: usize,
    capacity: usize,
    storage: []u8,
    free_list: std.ArrayList(usize),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, block_size: usize, capacity: usize) !FixedBlockPool {
        const total_size = block_size * capacity;
        const storage = try allocator.alloc(u8, total_size);
        var free_list = try std.ArrayList(usize).initCapacity(allocator, capacity);
        for (0..capacity) |i| {
            try free_list.append(i);
        }
        return .{
            .allocator = allocator,
            .block_size = block_size,
            .capacity = capacity,
            .storage = storage,
            .free_list = free_list,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *FixedBlockPool) void {
        self.free_list.deinit();
        self.allocator.free(self.storage);
    }

    pub fn alloc(self: *FixedBlockPool) ?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.free_list.items.len == 0) return null;
        const idx = self.free_list.pop();
        const start = idx * self.block_size;
        return self.storage[start .. start + self.block_size];
    }

    pub fn free(self: *FixedBlockPool, slice: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const base = @intFromPtr(self.storage.ptr);
        const addr = @intFromPtr(slice.ptr);
        if (addr < base or addr >= base + self.storage.len) return;
        const offset = addr - base;
        const idx = offset / self.block_size;
        if (idx < self.capacity) {
            self.free_list.append(idx) catch {};
        }
    }
};

pub const ArenaPool = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) ArenaPool {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn allocator(self: *ArenaPool) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn deinit(self: *ArenaPool) void {
        self.arena.deinit();
    }
};

test "fixed block pool" {
    var pool = try FixedBlockPool.init(std.testing.allocator, 32, 4);
    defer pool.deinit();

    const block = pool.alloc() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 32), block.len);
    pool.free(block);
}
