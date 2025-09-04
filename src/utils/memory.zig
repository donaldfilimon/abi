//! Memory utilities for WDBX-AI utils
//!
//! Provides memory management utilities for the utils module.

const std = @import("std");
const core = @import("../core/mod.zig");

// Re-export core memory functionality
pub const MemoryPool = core.allocators.PoolAllocator;
pub const TrackedArena = core.allocators.TrackedArenaAllocator;
pub const MemoryStats = core.memory.MemoryStats;

/// Simple memory pool for fixed-size allocations
pub const SimpleMemoryPool = struct {
    allocator: std.mem.Allocator,
    pool: std.ArrayList([]u8),
    item_size: usize,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, item_size: usize) Self {
        return Self{
            .allocator = allocator,
            .pool = std.ArrayList([]u8).init(allocator),
            .item_size = item_size,
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.pool.items) |item| {
            self.allocator.free(item);
        }
        self.pool.deinit();
    }
    
    pub fn acquire(self: *Self) ![]u8 {
        if (self.pool.items.len > 0) {
            return self.pool.pop();
        }
        return try self.allocator.alloc(u8, self.item_size);
    }
    
    pub fn release(self: *Self, item: []u8) !void {
        if (item.len == self.item_size) {
            try self.pool.append(item);
        } else {
            self.allocator.free(item);
        }
    }
};

test "simple memory pool" {
    const testing = std.testing;
    
    var pool = SimpleMemoryPool.init(testing.allocator, 64);
    defer pool.deinit();
    
    const item1 = try pool.acquire();
    const item2 = try pool.acquire();
    
    try pool.release(item1);
    try pool.release(item2);
}