//! Bloom Filter - Probabilistic data structure for set membership
//!
//! This module provides bloom filter implementations for:
//! - Space-efficient set membership testing
//! - Configurable false positive rates
//! - Memory-efficient storage

const std = @import("std");

/// Bloom filter for efficient set membership testing
pub const BloomFilter = struct {
    const Self = @This();

    /// Bit array for filter data
    bits: std.DynamicBitSet,
    /// Number of hash functions
    hash_count: u32,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new bloom filter
    pub fn init(allocator: std.mem.Allocator, size: usize, hash_count: u32) !*Self {
        const filter = try allocator.create(Self);
        filter.* = Self{
            .bits = try std.DynamicBitSet.initEmpty(allocator, size),
            .hash_count = hash_count,
            .allocator = allocator,
        };
        return filter;
    }

    /// Deinitialize the filter
    pub fn deinit(self: *Self) void {
        self.bits.deinit();
        self.allocator.destroy(self);
    }

    /// Add an item to the filter
    pub fn add(self: *Self, data: []const u8) void {
        const hash = std.hash.Wyhash.hash(0, data);
        for (0..self.hash_count) |i| {
            const bit_index = (hash +% i) % self.bits.capacity();
            self.bits.set(bit_index);
        }
    }

    /// Check if an item might be in the filter
    pub fn contains(self: *Self, data: []const u8) bool {
        const hash = std.hash.Wyhash.hash(0, data);
        for (0..self.hash_count) |i| {
            const bit_index = (hash +% i) % self.bits.capacity();
            if (!self.bits.isSet(bit_index)) return false;
        }
        return true;
    }
};
