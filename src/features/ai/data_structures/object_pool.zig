//! Object Pool - Specialized object pooling
//!
//! This module provides specialized object pool implementations for:
//! - Type-safe object reuse
//! - Automatic cleanup
//! - Memory management optimization

const std = @import("std");

/// Generic object pool for type-safe object reuse
pub fn ObjectPool(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Pool data
        objects: std.ArrayList(T),
        /// Available objects
        available: std.ArrayList(usize),
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new object pool
        pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
            const pool = try allocator.create(Self);
            pool.* = Self{
                .objects = try std.ArrayList(T).initCapacity(allocator, initial_capacity),
                .available = try std.ArrayList(usize).initCapacity(allocator, initial_capacity),
                .allocator = allocator,
            };

            // Pre-allocate objects
            try pool.objects.ensureTotalCapacity(initial_capacity);
            for (0..initial_capacity) |i| {
                try pool.available.append(i);
            }

            return pool;
        }

        /// Deinitialize the pool
        pub fn deinit(self: *Self) void {
            self.objects.deinit();
            self.available.deinit();
            self.allocator.destroy(self);
        }

        /// Get an object from the pool
        pub fn acquire(self: *Self) ?*T {
            if (self.available.items.len == 0) return null;
            const index = self.available.pop();
            return &self.objects.items[index];
        }

        /// Return an object to the pool
        pub fn release(self: *Self, object: *T) void {
            const index = @intFromPtr(object) - @intFromPtr(&self.objects.items[0]);
            self.available.append(index) catch {};
        }
    };
}
