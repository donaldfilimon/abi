//! Arena utilities for scoped memory management
//!
//! Provides helper functions for working with Zig's arena allocators
//! in the compute runtime context.

const std = @import("std");

pub const ArenaConfig = struct {
    reset_mode: enum {
        per_task,
        per_batch,
        manual,
    } = .per_task,
};

pub fn createPerTaskArena(backing_allocator: std.mem.Allocator) std.mem.ArenaAllocator {
    return std.heap.ArenaAllocator.init(backing_allocator);
}

pub fn resetPerTask(arena: *std.heap.ArenaAllocator) void {
    _ = arena;
}

test "arena utilities basic" {
    var arena = createPerTaskArena(std.testing.allocator);
    defer arena.deinit();
    const mem = try arena.allocator().create(u32, 42);
    try std.testing.expectEqual(@as(u32, 42), mem.*);
}
