//! Core Collections Module
//!
//! Modern collection utilities with proper Zig 0.16 patterns
//! Provides standardized initialization and memory management

const std = @import("std");

/// Standardized ArrayList type alias (managed variant keeps allocator on the instance)
pub const ArrayList = std.array_list.Managed;

/// Standardized StringHashMap type alias
pub const StringHashMap = std.StringHashMap;

/// Standardized AutoHashMap wrapper with proper initialization
pub const AutoHashMap = std.AutoHashMap;

/// Memory-safe arena allocator wrapper
pub const ArenaAllocator = std.heap.ArenaAllocator;

/// Collection utilities for common patterns
pub const utils = struct {
    /// Frees the memory used by a StringHashMap, including its keys and values.
    pub fn cleanupStringMap(map: *std.StringHashMap([]const u8), allocator: std.mem.Allocator) void {
        var it = map.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        map.deinit(allocator);
    }

    /// Creates a new ArrayList with proper initialization
    pub fn createArrayList(comptime T: type, allocator: std.mem.Allocator) ArrayList(T) {
        return ArrayList(T).init(allocator);
    }

    /// Creates a new StringHashMap with proper initialization
    pub fn createStringHashMap(comptime V: type, allocator: std.mem.Allocator) StringHashMap(V) {
        return StringHashMap(V).init(allocator);
    }

    /// Creates a new AutoHashMap with proper initialization
    pub fn createAutoHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator) AutoHashMap(K, V) {
        return AutoHashMap(K, V).init(allocator);
    }
};

test "collections - arraylist" {
    const testing = std.testing;
    var list = utils.createArrayList(u32, testing.allocator);
    defer list.deinit();

    try list.append(10);
    try testing.expectEqual(@as(usize, 1), list.items.len);
    try testing.expectEqual(@as(u32, 10), list.items[0]);
}

test "collections - stringhashmap" {
    const testing = std.testing;
    var map = utils.createStringHashMap(u32, testing.allocator);
    defer map.deinit();

    try map.put("ten", 10);
    const value = map.get("ten");
    try testing.expect(value != null);
    try testing.expectEqual(@as(u32, 10), value.?);
}

test "collections - autohashmap" {
    const testing = std.testing;
    var map = utils.createAutoHashMap(u32, []const u8, testing.allocator);
    defer map.deinit();

    try map.put(10, "ten");
    const value = map.get(10);
    try testing.expect(value != null);
    try testing.expectEqualStrings("ten", value.?);
}
