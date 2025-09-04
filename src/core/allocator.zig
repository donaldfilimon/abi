//! Memory Allocation and Management Module
//!
//! This module provides memory allocation utilities and allocator types
//! for the WDBX-AI system.

const std = @import("std");

/// Memory allocation utilities
pub const allocatorUtils = struct {
    /// Get the default page allocator
    pub const page = std.heap.page_allocator;

    /// Get the default general purpose allocator
    pub const general = std.heap.GeneralPurposeAllocator(.{}){};

    /// Get the default arena allocator
    pub fn arena() std.heap.ArenaAllocator {
        return std.heap.ArenaAllocator.init(page);
    }

    /// Get the default fixed buffer allocator
    pub fn fixedBuffer(buffer: []u8) std.heap.FixedBufferAllocator {
        return std.heap.FixedBufferAllocator.init(buffer);
    }

    /// Get the default C allocator
    pub const c = std.heap.c_allocator;
};

/// Re-export commonly used allocator types
pub const Allocator = std.mem.Allocator;
pub const ArenaAllocator = std.heap.ArenaAllocator;
pub const FixedBufferAllocator = std.heap.FixedBufferAllocator;
pub const GeneralPurposeAllocator = std.heap.GeneralPurposeAllocator;

/// Random number generation utilities
pub const random = struct {
    /// Get a random integer in the range [min, max]
    pub fn int(comptime T: type, min: T, max: T) T {
        const rng = std.crypto.random;
        return rng.intRangeAtMost(T, min, max);
    }

    /// Get a random float in the range [0, 1)
    pub fn float(comptime T: type) T {
        const rng = std.crypto.random;
        return rng.float(T);
    }

    /// Get a random float in the range [min, max)
    pub fn floatRange(comptime T: type, min: T, max: T) T {
        const rng = std.crypto.random;
        return rng.float(T) * (max - min) + min;
    }

    /// Get a random boolean with given probability
    pub fn boolean(probability: f32) bool {
        return float(f32) < probability;
    }

    /// Get a random element from a slice
    pub fn choice(comptime T: type, slice: []const T) T {
        const index = int(usize, 0, slice.len - 1);
        return slice[index];
    }

    /// Shuffle a slice in place
    pub fn shuffle(comptime T: type, slice: []T) void {
        const rng = std.crypto.random;
        rng.shuffle(T, slice);
    }
};

/// Initialize allocator module
pub fn init(allocator_instance: anytype) !void {
    _ = allocator_instance;
    // Nothing to initialize for now
}

/// Cleanup allocator module
pub fn deinit() void {
    // Nothing to cleanup for now
}

/// Get allocator type information
pub fn getType() []const u8 {
    return "Standard Zig Allocators";
}

test "Allocator utilities" {
    const testing = std.testing;
    
    // Test random number generation
    const random_int = random.int(u32, 1, 100);
    try testing.expect(random_int >= 1 and random_int <= 100);

    const random_float = random.float(f32);
    try testing.expect(random_float >= 0.0 and random_float < 1.0);

    // Test random choice
    const items = [_]u8{ 1, 2, 3, 4, 5 };
    const choice = random.choice(u8, &items);
    try testing.expect(choice >= 1 and choice <= 5);
}
