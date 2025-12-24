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
        /// Access order tracking
        access_order: std.ArrayList(K),
        /// Maximum capacity
        capacity: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new thread-safe cache
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
            const cache = try allocator.create(Self);
            cache.* = Self{
                .data = std.AutoHashMap(K, V).init(allocator),
                .mutex = std.Thread.Mutex{},
                .access_order = try std.ArrayList(K).initCapacity(allocator, capacity),
                .capacity = capacity,
                .allocator = allocator,
            };
            return cache;
        }

        /// Deinitialize the cache
        pub fn deinit(self: *Self) void {
            self.data.deinit();
            self.access_order.deinit(self.allocator);
            self.allocator.destroy(self);
        }

        /// Get a value from the cache
        pub fn get(self: *Self, key: K) ?V {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.data.get(key)) |value| {
                self.moveToEnd(key);
                return value;
            }
            return null;
        }

        /// Put a value in the cache
        pub fn put(self: *Self, key: K, value: V) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            // If capacity is 0, don't store anything
            if (self.capacity == 0) return;

            if (self.data.contains(key)) {
                try self.data.put(key, value);
                self.moveToEnd(key);
                return;
            }

            if (self.data.count() >= self.capacity) {
                if (self.access_order.items.len > 0) {
                    const old_key = self.access_order.items[0];
                    _ = self.data.remove(old_key);
                    // Shift elements left manually
                    for (1..self.access_order.items.len) |i| {
                        self.access_order.items[i - 1] = self.access_order.items[i];
                    }
                    _ = self.access_order.pop();
                }
            }
            try self.data.put(key, value);
            try self.access_order.append(self.allocator, key);
        }

        fn moveToEnd(self: *Self, key: K) void {
            for (self.access_order.items, 0..) |existing_key, i| {
                if (existing_key == key) {
                    _ = self.access_order.orderedRemove(i);
                    break;
                }
            }
            self.access_order.append(self.allocator, key) catch {};
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
            self.access_order.deinit(self.allocator);
            self.allocator.destroy(self);
        }

        /// Get a value from the cache and update its access order
        pub fn get(self: *Self, key: K) ?V {
            if (self.data.get(key)) |value| {
                // Move key to the end (most recently used)
                self.moveToEnd(key);
                return value;
            }
            return null;
        }

        /// Put a value in the cache with proper LRU handling
        pub fn put(self: *Self, key: K, value: V) !void {
            // If capacity is 0, don't store anything
            if (self.capacity == 0) return;

            if (self.data.contains(key)) {
                // Update existing value and move to end
                try self.data.put(key, value);
                self.moveToEnd(key);
                return;
            }

            if (self.data.count() >= self.capacity) {
                // Remove least recently used item (from the front)
                if (self.access_order.items.len > 0) {
                    const old_key = self.access_order.items[0];
                    _ = self.data.remove(old_key);
                    // Shift elements left manually
                    for (1..self.access_order.items.len) |i| {
                        self.access_order.items[i - 1] = self.access_order.items[i];
                    }
                    _ = self.access_order.pop();
                }
            }
            try self.data.put(key, value);
            try self.access_order.append(self.allocator, key);
        }

        /// Helper function to move a key to the end of the access order
        fn moveToEnd(self: *Self, key: K) void {
            // Remove key from current position and add to end
            for (self.access_order.items, 0..) |existing_key, i| {
                if (existing_key == key) {
                    _ = self.access_order.orderedRemove(i);
                    break;
                }
            }
            self.access_order.append(self.allocator, key) catch {};
        }
    };
}

test "LRU cache eviction" {
    const testing = std.testing;

    var cache = try LRUCache(u32, []const u8).init(testing.allocator, 3);
    defer cache.deinit();

    // Fill cache to capacity
    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    // Verify all items are present
    try testing.expectEqualStrings("one", cache.get(1).?);
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);

    // Add fourth item, should evict least recently used (key 1)
    try cache.put(4, "four");

    // Key 1 should be evicted
    try testing.expect(cache.get(1) == null);
    // Others should still be present
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);
    try testing.expectEqualStrings("four", cache.get(4).?);
}

test "LRU cache access order" {
    const testing = std.testing;

    var cache = try LRUCache(u32, []const u8).init(testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    // Access key 1, making it most recently used
    _ = cache.get(1);

    // Add fourth item, should evict key 2 (now least recently used)
    try cache.put(4, "four");

    // Key 2 should be evicted, key 1 should remain
    try testing.expect(cache.get(2) == null);
    try testing.expectEqualStrings("one", cache.get(1).?);
    try testing.expectEqualStrings("three", cache.get(3).?);
    try testing.expectEqualStrings("four", cache.get(4).?);
}

test "LRU cache update existing key" {
    const testing = std.testing;

    var cache = try LRUCache(u32, []const u8).init(testing.allocator, 3);
    defer cache.deinit();

    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    // Update existing key, should not change size or evict
    try cache.put(1, "updated_one");

    try testing.expectEqualStrings("updated_one", cache.get(1).?);
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);
}

test "Thread-safe LRU cache eviction" {
    const testing = std.testing;

    var cache = try ThreadSafeCache(u32, []const u8).init(testing.allocator, 3);
    defer cache.deinit();

    // Fill cache to capacity
    try cache.put(1, "one");
    try cache.put(2, "two");
    try cache.put(3, "three");

    // Verify all items are present
    try testing.expectEqualStrings("one", cache.get(1).?);
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);

    // Add fourth item, should evict least recently used (key 1)
    try cache.put(4, "four");

    // Key 1 should be evicted
    try testing.expect(cache.get(1) == null);
    // Others should still be present
    try testing.expectEqualStrings("two", cache.get(2).?);
    try testing.expectEqualStrings("three", cache.get(3).?);
    try testing.expectEqualStrings("four", cache.get(4).?);
}

test "Thread-safe LRU cache concurrent access" {
    const testing = std.testing;

    var cache = try ThreadSafeCache(u32, []const u8).init(testing.allocator, 10);
    defer cache.deinit();

    // Test concurrent puts and gets
    const TestData = struct {
        cache: *ThreadSafeCache(u32, []const u8),
        id: u32,

        fn run(self: *@This()) !void {
            for (0..100) |i| {
                const key = self.id * 100 + @as(u32, @intCast(i));
                const value = try std.fmt.allocPrint(testing.allocator, "value_{d}_{d}", .{ self.id, i });
                defer testing.allocator.free(value);

                try self.cache.put(key, value);
                _ = self.cache.get(key);
            }
        }
    };

    var test_data_1 = TestData{ .cache = cache, .id = 1 };
    var test_data_2 = TestData{ .cache = cache, .id = 2 };

    const thread1 = try std.Thread.spawn(.{}, TestData.run, .{&test_data_1});
    const thread2 = try std.Thread.spawn(.{}, TestData.run, .{&test_data_2});

    thread1.join();
    thread2.join();

    // Verify some values are still accessible
    if (cache.get(150)) |value| {
        try testing.expect(std.mem.eql(u8, value, "value_1_50"));
    }
}

test "Cache capacity zero" {
    const testing = std.testing;

    var cache = try LRUCache(u32, []const u8).init(testing.allocator, 0);
    defer cache.deinit();

    // Should not store any items
    try cache.put(1, "one");
    try testing.expect(cache.get(1) == null);
}

test "Cache large capacity" {
    const testing = std.testing;

    var cache = try LRUCache(u32, []const u8).init(testing.allocator, 1000);
    defer cache.deinit();

    // Add many items without eviction
    for (0..100) |i| {
        const value = try std.fmt.allocPrint(testing.allocator, "value_{d}", .{i});
        defer testing.allocator.free(value);
        try cache.put(@as(u32, @intCast(i)), value);
    }

    // All items should still be accessible
    for (0..100) |i| {
        const value = cache.get(@as(u32, @intCast(i)));
        try testing.expect(value != null);
    }
}
