//! Collections and Data Structures Module
//!
//! This module provides data structures and collection utilities
//! for the WDBX-AI system.

const std = @import("std");

/// Data structures and collections
pub const collections = struct {
    /// Re-export commonly used collection types
    pub const ArrayList = std.ArrayList;
    pub const ArrayListUnmanaged = std.ArrayListUnmanaged;
    pub const StringHashMap = std.StringHashMap;
    pub const AutoHashMap = std.AutoHashMap;
    pub const HashMap = std.HashMap;
    pub const BufMap = std.BufMap;
    pub const PriorityQueue = std.PriorityQueue;
    pub const BoundedArray = std.BoundedArray;
    pub const StaticBitSet = std.StaticBitSet;
    pub const DynamicBitSet = std.DynamicBitSet;

    /// Create a new ArrayList with the given allocator
    pub fn createArrayList(comptime T: type, alloc: std.mem.Allocator) !std.ArrayList(T) {
        return std.ArrayList(T).initCapacity(alloc, 0);
    }

    /// Create a new ArrayListUnmanaged
    pub fn createArrayListUnmanaged(comptime T: type) std.ArrayListUnmanaged(T) {
        return std.ArrayListUnmanaged(T){};
    }

    /// Create a new StringHashMap with the given allocator
    pub fn createStringHashMap(comptime V: type, alloc: std.mem.Allocator) std.StringHashMap(V) {
        return std.StringHashMap(V).init(alloc);
    }

    /// Create a new AutoHashMap with the given allocator
    pub fn createAutoHashMap(comptime K: type, comptime V: type, alloc: std.mem.Allocator) std.AutoHashMap(K, V) {
        return std.AutoHashMap(K, V).init(alloc);
    }

    /// Create a new HashMap with the given allocator
    pub fn createHashMap(comptime K: type, comptime V: type, alloc: std.mem.Allocator) std.AutoHashMap(K, V) {
        return std.AutoHashMap(K, V).init(alloc);
    }

    /// Create a new PriorityQueue with the given allocator
    pub fn createPriorityQueue(comptime T: type, alloc: std.mem.Allocator) std.PriorityQueue(T, void, std.sort.asc(T)) {
        return std.PriorityQueue(T, void, std.sort.asc(T)).init(alloc, {});
    }
};

/// Re-export commonly used collection types for convenience
pub const ArrayList = std.ArrayList;
pub const StringHashMap = std.StringHashMap;
pub const AutoHashMap = std.AutoHashMap;
pub const HashMap = std.HashMap;
pub const BufMap = std.BufMap;
pub const PriorityQueue = std.PriorityQueue;
pub const BoundedArray = std.BoundedArray;
pub const StaticBitSet = std.StaticBitSet;
pub const DynamicBitSet = std.DynamicBitSet;

/// String and text processing utilities
pub const string = struct {
    /// Check if a string starts with a prefix
    pub fn startsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.startsWith(u8, haystack, needle);
    }

    /// Check if a string ends with a suffix
    pub fn endsWith(haystack: []const u8, needle: []const u8) bool {
        return std.mem.endsWith(u8, haystack, needle);
    }

    /// Find the first occurrence of a substring
    pub fn indexOf(haystack: []const u8, needle: []const u8) ?usize {
        return std.mem.indexOf(u8, haystack, needle);
    }

    /// Find the last occurrence of a substring
    pub fn lastIndexOf(haystack: []const u8, needle: []const u8) ?usize {
        return std.mem.lastIndexOf(u8, haystack, needle);
    }

    /// Split a string by a delimiter
    pub fn split(allocator: std.mem.Allocator, text: []const u8, delimiter: []const u8) !ArrayList([]const u8) {
        var result = try ArrayList([]const u8).initCapacity(allocator, 0);
        var iter = std.mem.splitScalar(u8, text, delimiter[0]);
        while (iter.next()) |part| {
            try result.append(part);
        }
        return result;
    }

    /// Join strings with a separator
    pub fn join(allocator: std.mem.Allocator, parts: []const []const u8, separator: []const u8) ![]u8 {
        if (parts.len == 0) return &[_]u8{};
        if (parts.len == 1) return allocator.dupe(u8, parts[0]);

        var total_len: usize = 0;
        for (parts) |part| {
            total_len += part.len;
        }
        total_len += separator.len * (parts.len - 1);

        var result = try allocator.alloc(u8, total_len);
        var offset: usize = 0;

        for (parts, 0..) |part, i| {
            if (i > 0) {
                @memcpy(result[offset..][0..separator.len], separator);
                offset += separator.len;
            }
            @memcpy(result[offset..][0..part.len], part);
            offset += part.len;
        }

        return result;
    }

    /// Trim whitespace from both ends
    pub fn trim(text: []const u8) []const u8 {
        return std.mem.trim(u8, text, &std.ascii.whitespace);
    }

    /// Trim whitespace from the beginning
    pub fn trimStart(text: []const u8) []const u8 {
        return std.mem.trimLeft(u8, text, &std.ascii.whitespace);
    }

    /// Trim whitespace from the end
    pub fn trimEnd(text: []const u8) []const u8 {
        return std.mem.trimRight(u8, text, &std.ascii.whitespace);
    }

    /// Convert string to lowercase
    pub fn toLower(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        var result = try allocator.alloc(u8, text.len);
        for (text, 0..) |char, i| {
            result[i] = std.ascii.toLower(char);
        }
        return result;
    }

    /// Convert string to uppercase
    pub fn toUpper(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        var result = try allocator.alloc(u8, text.len);
        for (text, 0..) |char, i| {
            result[i] = std.ascii.toUpper(char);
        }
        return result;
    }
};

/// Initialize collections module
pub fn init(allocator_instance: anytype) !void {
    _ = allocator_instance;
    // Nothing to initialize for now
}

/// Cleanup collections module
pub fn deinit() void {
    // Nothing to cleanup for now
}

/// Get available collection types
pub fn getAvailableTypes() []const []const u8 {
    return &[_][]const u8{
        "ArrayList",
        "HashMap",
        "StringHashMap",
        "AutoHashMap",
        "PriorityQueue",
        "BoundedArray",
        "BitSet",
    };
}

test "Collections utilities" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test string utilities
    const test_str = "  Hello, World!  ";
    const trimmed = string.trim(test_str);
    try testing.expectEqualStrings("Hello, World!", trimmed);

    var split_result = try string.split(allocator, "a,b,c", ",");
    defer split_result.deinit();
    try testing.expectEqual(@as(usize, 3), split_result.items.len);
    try testing.expectEqualStrings("a", split_result.items[0]);

    const joined = try string.join(allocator, &[_][]const u8{ "a", "b", "c" }, "-");
    defer allocator.free(joined);
    try testing.expectEqualStrings("a-b-c", joined);

    // Test collections
    var list = try collections.createArrayList(u32, allocator);
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.append(3);

    try testing.expectEqual(@as(usize, 3), list.items.len);
    try testing.expectEqual(@as(u32, 1), list.items[0]);
}
