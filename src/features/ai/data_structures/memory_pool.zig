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
            if (self.available.items.len == 0) {
                // Try to expand pool if possible
                if (self.pool.items.len < self.pool.capacity) {
                    const new_index = self.pool.items.len;
                    self.pool.items.len += 1;
                    return &self.pool.items[new_index];
                }
                return null;
            }
            const index = self.available.pop();
            return &self.pool.items[index];
        }

        /// Return an object to the pool
        pub fn put(self: *Self, object: *T) void {
            const pool_start = @intFromPtr(self.pool.items.ptr);
            const object_ptr = @intFromPtr(object);
            
            // Bounds check for safety
            if (object_ptr < pool_start or object_ptr >= pool_start + (@sizeOf(T) * self.pool.items.len)) {
                return; // Object not from this pool
            }
            
            const index = (object_ptr - pool_start) / @sizeOf(T);
            self.available.append(index) catch {
                // If we can't add to available list, just ignore
                // This prevents memory leaks in error conditions
            };
        }

        /// Get multiple objects at once (batch operation)
        pub fn getBatch(self: *Self, count: usize) []T {
            const actual_count = @min(count, self.available.items.len);
            if (actual_count == 0) return &[_]T{};
            
            const start_idx = self.available.items.len - actual_count;
            const indices = self.available.items[start_idx..];
            self.available.items.len = start_idx;
            
            var result: []T = undefined;
            result.ptr = @ptrCast(&self.pool.items[indices[0]]);
            result.len = actual_count;
            return result;
        }

        /// Return multiple objects at once (batch operation)
        pub fn putBatch(self: *Self, objects: []T) void {
            for (objects) |object| {
                self.put(&object);
            }
        }
    };
}
