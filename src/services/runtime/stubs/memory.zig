const std = @import("std");
const types = @import("types.zig");

pub const MemoryPool = struct {
    pub fn init(_: std.mem.Allocator, _: MemoryPoolConfig) types.Error!MemoryPool {
        return error.RuntimeDisabled;
    }

    pub fn deinit(_: *MemoryPool) void {}

    pub fn alloc(_: *MemoryPool, _: usize) types.Error![]u8 {
        return error.RuntimeDisabled;
    }

    pub fn free(_: *MemoryPool, _: []u8) void {}

    pub fn getStats(_: *const MemoryPool) MemoryPoolStats {
        return .{};
    }
};

pub const MemoryPoolConfig = struct {
    block_size: usize = 4096,
    initial_blocks: usize = 16,
};

pub const MemoryPoolStats = struct {
    allocated_bytes: usize = 0,
    free_bytes: usize = 0,
    block_count: usize = 0,
};

pub const ArenaAllocator = struct {
    pub fn init(_: std.mem.Allocator) ArenaAllocator {
        return .{};
    }

    pub fn deinit(_: *ArenaAllocator) void {}

    pub fn allocator(_: *ArenaAllocator) std.mem.Allocator {
        return std.heap.page_allocator;
    }

    pub fn reset(_: *ArenaAllocator) void {}
};
