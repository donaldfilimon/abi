//! Memory management utilities for the ABI framework.
//!
//! Provides stable allocators for long-lived data, worker arenas for scratch
//! allocations, and fixed-size memory pools for performance-critical scenarios.

const std = @import("std");

pub const StableAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{}) = .{},

    pub fn allocator(self: *StableAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    pub fn deinit(self: *StableAllocator) void {
        _ = self.gpa.deinit();
    }
};

pub const WorkerArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) WorkerArena {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn allocator(self: *WorkerArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *WorkerArena) void {
        self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *WorkerArena) void {
        self.arena.deinit();
    }
};

pub const ScratchArena = WorkerArena;

pub const PoolError = error{
    InvalidPool,
    InvalidBlock,
};

pub const FixedPool = struct {
    allocator: std.mem.Allocator,
    block_size: usize,
    storage: []u8,
    free_offsets: []usize,
    free_count: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        block_size: usize,
        block_count: usize,
    ) !FixedPool {
        if (block_size == 0 or block_count == 0) return PoolError.InvalidPool;

        const total_size = block_size * block_count;
        const storage = try allocator.alloc(u8, total_size);
        errdefer allocator.free(storage);

        const free_offsets = try allocator.alloc(usize, block_count);
        errdefer allocator.free(free_offsets);

        var i: usize = 0;
        while (i < block_count) : (i += 1) {
            free_offsets[i] = i * block_size;
        }

        return .{
            .allocator = allocator,
            .block_size = block_size,
            .storage = storage,
            .free_offsets = free_offsets,
            .free_count = block_count,
        };
    }

    pub fn deinit(self: *FixedPool) void {
        self.allocator.free(self.storage);
        self.allocator.free(self.free_offsets);
        self.* = undefined;
    }

    pub fn alloc(self: *FixedPool) ?[]u8 {
        if (self.free_count == 0) return null;
        self.free_count -= 1;
        const offset = self.free_offsets[self.free_count];
        return self.storage[offset .. offset + self.block_size];
    }

    pub fn free(self: *FixedPool, block: []u8) PoolError!void {
        if (block.len != self.block_size) return PoolError.InvalidBlock;

        const base = @intFromPtr(self.storage.ptr);
        const addr = @intFromPtr(block.ptr);
        if (addr < base) return PoolError.InvalidBlock;
        const offset = addr - base;
        if (offset % self.block_size != 0) return PoolError.InvalidBlock;
        if (offset + self.block_size > self.storage.len) {
            return PoolError.InvalidBlock;
        }
        if (self.free_count >= self.free_offsets.len) return PoolError.InvalidBlock;

        self.free_offsets[self.free_count] = offset;
        self.free_count += 1;
    }

    pub fn available(self: *const FixedPool) usize {
        return self.free_count;
    }

    pub fn capacity(self: *const FixedPool) usize {
        return self.free_offsets.len;
    }
};

test "stable allocator allocates" {
    var stable = StableAllocator{};
    defer stable.deinit();
    const allocator = stable.allocator();
    const buffer = try allocator.alloc(u8, 16);
    allocator.free(buffer);
}

test "worker arena reset" {
    var arena = WorkerArena.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const first = try allocator.alloc(u8, 8);
    _ = first;
    arena.reset();
    const second = try allocator.alloc(u8, 8);
    try std.testing.expectEqual(@as(usize, 8), second.len);
}

test "fixed pool allocates and reuses blocks" {
    var pool = try FixedPool.init(std.testing.allocator, 8, 2);
    defer pool.deinit();

    const a = pool.alloc() orelse return error.TestUnexpectedResult;
    const b = pool.alloc() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), pool.available());
    try std.testing.expect(pool.alloc() == null);

    try pool.free(a);
    try std.testing.expectEqual(@as(usize, 1), pool.available());

    const c = pool.alloc() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), pool.available());
    _ = b;
    _ = c;
}

test "fixed pool rejects invalid blocks" {
    var pool = try FixedPool.init(std.testing.allocator, 16, 4);
    defer pool.deinit();

    const valid = pool.alloc() orelse return error.TestUnexpectedResult;
    defer _ = pool.free(valid);

    var invalid = [_]u8{0} ** 16;
    try std.testing.expectError(PoolError.InvalidBlock, pool.free(&invalid));

    var wrong_size = [_]u8{0} ** 8;
    try std.testing.expectError(PoolError.InvalidBlock, pool.free(&wrong_size));
}
