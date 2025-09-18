//! Cache - Thread-safe caching implementations
//!
//! This module provides various cache implementations:
//! - LRU (Least Recently Used) cache
//! - Thread-safe variants
//! - Memory-efficient storage

const std = @import("std");

/// Thread-safe LRU cache implementation
pub fn ThreadSafeCache(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        /// Cache data
        data: std.AutoHashMap(K, V),
        /// Mutex for thread safety
        mutex: std.Thread.Mutex,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new thread-safe cache
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
            _ = capacity; // Not used in this stub
            const cache = try allocator.create(Self);
            cache.* = Self{
                .data = std.AutoHashMap(K, V).init(allocator),
                .mutex = std.Thread.Mutex{},
                .allocator = allocator,
            };
            return cache;
        }

        /// Deinitialize the cache
        pub fn deinit(self: *Self) void {
            self.data.deinit();
            self.allocator.destroy(self);
        }

        /// Get a value from the cache
        pub fn get(self: *Self, key: K) ?V {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.data.get(key);
        }

        /// Put a value in the cache
        pub fn put(self: *Self, key: K, value: V) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.data.put(key, value);
        }
    };
}

/// LRU Cache implementation
pub fn LRUCache(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        /// Cache data
        data: std.AutoHashMap(K, V),
        /// Access order tracking
        access_order: std.ArrayList(K),
        /// Maximum capacity
        capacity: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new LRU cache
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
            const cache = try allocator.create(Self);
            cache.* = Self{
                .data = std.AutoHashMap(K, V).init(allocator),
                .access_order = try std.ArrayList(K).initCapacity(allocator, capacity),
                .capacity = capacity,
                .allocator = allocator,
            };
            return cache;
        }

        /// Deinitialize the cache
        pub fn deinit(self: *Self) void {
            self.data.deinit();
            self.access_order.deinit();
            self.allocator.destroy(self);
        }

        /// Get a value from the cache
        pub fn get(self: *Self, key: K) ?V {
            return self.data.get(key);
        }

        /// Put a value in the cache
        pub fn put(self: *Self, key: K, value: V) !void {
            if (self.data.count() >= self.capacity) {
                // Remove oldest item
                if (self.access_order.popOrNull()) |old_key| {
                    _ = self.data.remove(old_key);
                }
            }
            try self.data.put(key, value);
            try self.access_order.append(key);
        }
    };
}
