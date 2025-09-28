//! Core Collections Module
//!
//! Modern collection utilities with proper Zig 0.16 patterns
//! Provides standardized initialization and memory management

const std = @import("std");

/// Standardized ArrayList wrapper with proper initialization
pub fn ArrayList(comptime T: type) type {
    return struct {
        const Self = @This();
        inner: std.ArrayList(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .inner = std.ArrayList(T).init(allocator) };
        }

        pub fn initCapacity(allocator: std.mem.Allocator, cap: usize) !Self {
            return .{ .inner = try std.ArrayList(T).initCapacity(allocator, cap) };
        }

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn append(self: *Self, allocator: std.mem.Allocator, item: T) !void {
            try self.inner.append(allocator, item);
        }

        pub fn appendSlice(self: *Self, allocator: std.mem.Allocator, slice: []const T) !void {
            try self.inner.appendSlice(allocator, slice);
        }

        pub fn insert(self: *Self, allocator: std.mem.Allocator, index: usize, item: T) !void {
            try self.inner.insert(allocator, index, item);
        }

        pub fn orderedRemove(self: *Self, index: usize) T {
            return self.inner.orderedRemove(index);
        }

        pub fn swapRemove(self: *Self, index: usize) T {
            return self.inner.swapRemove(index);
        }

        pub fn clearAndFree(self: *Self, allocator: std.mem.Allocator) void {
            self.inner.clearAndFree(allocator);
        }

        pub fn resize(self: *Self, allocator: std.mem.Allocator, new_len: usize) !void {
            try self.inner.resize(allocator, new_len);
        }

        pub fn shrinkAndFree(self: *Self, allocator: std.mem.Allocator, new_len: usize) void {
            self.inner.shrinkAndFree(allocator, new_len);
        }

        // Accessors
        pub fn items(self: *const Self) []const T {
            return self.inner.items;
        }

        pub fn itemsMut(self: *Self) []T {
            return self.inner.items;
        }

        pub fn len(self: *const Self) usize {
            return self.inner.items.len;
        }

        pub fn capacity(self: *const Self) usize {
            return self.inner.capacity;
        }
    };
}

/// Standardized StringHashMap wrapper with proper initialization
pub fn StringHashMap(comptime V: type) type {
    return struct {
        const Self = @This();
        inner: std.StringHashMap(V),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .inner = std.StringHashMap(V).init(allocator) };
        }

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn put(self: *Self, allocator: std.mem.Allocator, key: []const u8, value: V) !void {
            try self.inner.put(allocator, key, value);
        }

        pub fn putNoClobber(self: *Self, allocator: std.mem.Allocator, key: []const u8, value: V) !void {
            try self.inner.putNoClobber(allocator, key, value);
        }

        pub fn get(self: *const Self, key: []const u8) ?V {
            return self.inner.get(key);
        }

        pub fn getPtr(self: *Self, key: []const u8) ?*V {
            return self.inner.getPtr(key);
        }

        pub fn remove(self: *Self, key: []const u8) bool {
            return self.inner.remove(key);
        }

        pub fn contains(self: *const Self, key: []const u8) bool {
            return self.inner.contains(key);
        }

        pub fn count(self: *const Self) u32 {
            return self.inner.count();
        }

        pub fn iterator(self: *const Self) std.StringHashMap(V).Iterator {
            return self.inner.iterator();
        }

        pub fn keyIterator(self: *const Self) std.StringHashMap(V).KeyIterator {
            return self.inner.keyIterator();
        }

        pub fn valueIterator(self: *const Self) std.StringHashMap(V).ValueIterator {
            return self.inner.valueIterator();
        }

        pub fn clearAndFree(self: *Self, allocator: std.mem.Allocator) void {
            self.inner.clearAndFree(allocator);
        }
    };
}

/// Standardized AutoHashMap wrapper with proper initialization
pub fn AutoHashMap(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();
        inner: std.AutoHashMap(K, V),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .inner = std.AutoHashMap(K, V).init(allocator) };
        }

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn put(self: *Self, allocator: std.mem.Allocator, key: K, value: V) !void {
            try self.inner.put(allocator, key, value);
        }

        pub fn putNoClobber(self: *Self, allocator: std.mem.Allocator, key: K, value: V) !void {
            try self.inner.putNoClobber(allocator, key, value);
        }

        pub fn get(self: *const Self, key: K) ?V {
            return self.inner.get(key);
        }

        pub fn getPtr(self: *Self, key: K) ?*V {
            return self.inner.getPtr(key);
        }

        pub fn remove(self: *Self, key: K) bool {
            return self.inner.remove(key);
        }

        pub fn contains(self: *const Self, key: K) bool {
            return self.inner.contains(key);
        }

        pub fn count(self: *const Self) u32 {
            return self.inner.count();
        }

        pub fn iterator(self: *const Self) std.AutoHashMap(K, V).Iterator {
            return self.inner.iterator();
        }

        pub fn keyIterator(self: *const Self) std.AutoHashMap(K, V).KeyIterator {
            return self.inner.keyIterator();
        }

        pub fn valueIterator(self: *const Self) std.AutoHashMap(K, V).ValueIterator {
            return self.inner.valueIterator();
        }

        pub fn clearAndFree(self: *Self, allocator: std.mem.Allocator) void {
            self.inner.clearAndFree(allocator);
        }
    };
}

/// Memory-safe arena allocator wrapper
pub const ArenaAllocator = struct {
    inner: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) ArenaAllocator {
        return .{ .inner = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        self.inner.deinit();
    }

    pub fn reset(self: *ArenaAllocator, mode: std.heap.ArenaAllocator.ResetMode) void {
        self.inner.reset(mode);
    }

    pub fn allocator(self: *ArenaAllocator) std.mem.Allocator {
        return self.inner.allocator();
    }
};

/// Collection utilities for common patterns
pub const utils = struct {
    /// Safely append to ArrayList with proper error handling
    pub fn appendSafe(comptime T: type, list: *ArrayList(T), allocator: std.mem.Allocator, item: T) bool {
        list.append(allocator, item) catch return false;
        return true;
    }

    /// Safely put into StringHashMap with proper error handling
    pub fn putSafe(comptime V: type, map: *StringHashMap(V), allocator: std.mem.Allocator, key: []const u8, value: V) bool {
        map.put(allocator, key, value) catch return false;
        return true;
    }

    /// Create a deep copy of a string
    pub fn dupeString(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        return try allocator.dupe(u8, str);
    }

    /// Clean up StringHashMap with string keys and values
    pub fn cleanupStringMap(map: *StringHashMap([]const u8), allocator: std.mem.Allocator) void {
        var iterator = map.iterator();
        while (iterator.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        map.deinit();
    }
};

test "collections - ArrayList wrapper" {
    const testing = std.testing;
    var list = ArrayList(u32).init(testing.allocator);
    defer list.deinit();

    try list.append(testing.allocator, 42);
    try testing.expectEqual(@as(usize, 1), list.len());
    try testing.expectEqual(@as(u32, 42), list.items()[0]);
}

test "collections - StringHashMap wrapper" {
    const testing = std.testing;
    var map = StringHashMap(u32).init(testing.allocator);
    defer map.deinit();

    try map.put(testing.allocator, "key", 42);
    try testing.expectEqual(@as(?u32, 42), map.get("key"));
    try testing.expectEqual(@as(u32, 1), map.count());
}

test "collections - ArenaAllocator wrapper" {
    const testing = std.testing;
    var arena = ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const allocator = arena.allocator();
    const data = try allocator.alloc(u32, 100);
    _ = data;
    // Memory is automatically freed when arena deinit is called
}
