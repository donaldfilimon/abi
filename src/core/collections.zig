//! Core Collections Module
//!
//! Modern collection utilities with proper Zig 0.16 patterns
//! Provides standardized initialization and memory management

const std = @import("std");

/// Standardized ArrayList type alias
pub const ArrayList = std.ArrayList;

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
};

test "collections - arraylist" {
    const testing = std.testing;
    var list = std.ArrayList(u32){};
    defer list.deinit(testing.allocator);

    try list.append(testing.allocator, 10);
    try testing.expectEqual(@as(usize, 1), list.items.len);
    try testing.expectEqual(@as(u32, 10), list.items[0]);
}

test "collections - stringhashmap" {
    const testing = std.testing;
    var map = std.StringHashMap(u32).init(testing.allocator);
    defer map.deinit();

    try map.put("ten", 10);
    const value = map.get("ten");
    try testing.expect(value != null);
    try testing.expectEqual(@as(u32, 10), value.?);
}
