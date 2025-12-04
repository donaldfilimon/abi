//! Memory Pool - Efficient object pooling for memory reuse
//!
//! This module provides memory pool implementations for:
//! - Object reuse to reduce allocation overhead
//! - Memory-efficient storage
//! - Thread-safe variants available

const std = @import("std");

/// Generic memory pool for object reuse
pub fn MemoryPool(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Pool data
        pool: std.ArrayList(T),
        /// Available indices
        available: std.ArrayList(usize),
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new memory pool
        pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
            const pool = try allocator.create(Self);
            pool.* = Self{
                .pool = try std.ArrayList(T).initCapacity(allocator, initial_capacity),
                .available = try std.ArrayList(usize).initCapacity(allocator, initial_capacity),
                .allocator = allocator,
            };

            // Pre-allocate objects
            try pool.pool.ensureTotalCapacity(initial_capacity);
            for (0..initial_capacity) |i| {
                try pool.available.append(i);
            }

            return pool;
        }

        /// Deinitialize the pool
        pub fn deinit(self: *Self) void {
            self.pool.deinit();
            self.available.deinit();
            self.allocator.destroy(self);
        }

        /// Get an object from the pool
        pub fn get(self: *Self) ?*T {
            if (self.available.items.len == 0) return null;
            const index = self.available.pop();
            return &self.pool.items[index];
        }

        /// Return an object to the pool
        pub fn put(self: *Self, object: *T) void {
            const index = @intFromPtr(object) - @intFromPtr(&self.pool.items[0]);
            self.available.append(index) catch {};
        }
    };
}
