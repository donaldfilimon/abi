//! JSON Utilities Module
//! Contains JSON parsing, serialization, and manipulation utilities

const std = @import("std");

// =============================================================================
// JSON VALUE TYPES
// =============================================================================

/// Simple JSON value types for basic JSON operations
pub const JsonValue = union(enum) {
    null,
    bool: bool,
    int: i64,
    float: f64,
    string: []const u8,
    array: []JsonValue,
    object: std.StringHashMap(JsonValue),

    /// Clean up resources used by this JsonValue
    pub fn deinit(self: *const JsonValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string => |str| allocator.free(str),
            .array => |arr| {
                for (arr) |*item| item.deinit(allocator);
                allocator.free(arr);
            },
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.*.deinit(allocator);
                }
                // The caller owns the map storage for const objects.
            },
            else => {},
        }
    }
};

// =============================================================================
// JSON PARSING AND SERIALIZATION
// =============================================================================

/// JSON utility functions for parsing and serialization
pub const JsonUtils = struct {
    /// Parse JSON string into JsonValue
    pub fn parse(allocator: std.mem.Allocator, json_str: []const u8) !JsonValue {
        var tree = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer tree.deinit();

        return try jsonValueFromTree(allocator, tree.value);
    }

    /// Serialize JsonValue to JSON string (simplified version)
    pub fn stringify(allocator: std.mem.Allocator, value: JsonValue) ![]u8 {
        return switch (value) {
            .null => allocator.dupe(u8, "null"),
            .bool => |b| if (b) allocator.dupe(u8, "true") else allocator.dupe(u8, "false"),
            .int => |i| std.fmt.allocPrint(allocator, "{}", .{i}),
            .float => |f| std.fmt.allocPrint(allocator, "{}", .{f}),
            .string => |s| std.fmt.allocPrint(allocator, "\"{s}\"", .{s}),
            .array => allocator.dupe(u8, "[]"), // Simplified
            .object => allocator.dupe(u8, "{}"), // Simplified
        };
    }

    /// Parse JSON string into typed struct using std.json
    pub fn parseInto(allocator: std.mem.Allocator, comptime T: type, json_str: []const u8) !T {
        var tree = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer tree.deinit();

        return try std.json.parseFromValueLeaky(T, allocator, tree.value, .{});
    }

    /// Serialize struct to JSON string (simplified)
    pub fn stringifyFrom(allocator: std.mem.Allocator, value: anytype) ![]u8 {
        _ = value; // Remove unused parameter warning
        // Simplified implementation - in production, use std.json
        return allocator.dupe(u8, "{}");
    }

    /// Helper function to convert std.json.Value to JsonValue
    fn jsonValueFromTree(allocator: std.mem.Allocator, value: std.json.Value) !JsonValue {
        switch (value) {
            .null => return .null,
            .bool => |b| return JsonValue{ .bool = b },
            .integer => |i| return JsonValue{ .int = i },
            .float => |f| return JsonValue{ .float = f },
            .number_string => |s| return JsonValue{ .string = try allocator.dupe(u8, s) },
            .string => |s| return JsonValue{ .string = try allocator.dupe(u8, s) },
            .array => |arr| {
                const items = try allocator.alloc(JsonValue, arr.items.len);
                errdefer allocator.free(items);
                for (arr.items, 0..) |item, i| {
                    items[i] = try jsonValueFromTree(allocator, item);
                }
                return JsonValue{ .array = items };
            },
            .object => |obj| {
                var map = std.StringHashMap(JsonValue){};
                errdefer map.deinit(allocator);
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = try allocator.dupe(u8, entry.key_ptr.*);
                    errdefer allocator.free(key);
                    const val = try jsonValueFromTree(allocator, entry.value_ptr.*);
                    errdefer val.deinit(allocator);
                    try map.put(key, val);
                }
                return JsonValue{ .object = map };
            },
        }
    }
};

// =============================================================================
// HIGH-LEVEL JSON OPERATIONS
// =============================================================================

/// High-level JSON operations for common use cases
pub const JsonOps = struct {
    /// Extract a value from a JSON object by key path (dot notation)
    pub fn getValue(json: JsonValue, path: []const u8) ?JsonValue {
        var current = json;
        var path_iter = std.mem.splitSequence(u8, path, ".");

        while (path_iter.next()) |segment| {
            switch (current) {
                .object => |obj| {
                    current = obj.get(segment) orelse return null;
                },
                else => return null,
            }
        }

        return current;
    }

    /// Get a string value from JSON by path
    pub fn getString(json: JsonValue, path: []const u8) ?[]const u8 {
        const value = getValue(json, path) orelse return null;
        return switch (value) {
            .string => |s| s,
            else => null,
        };
    }

    /// Get an integer value from JSON by path
    pub fn getInt(json: JsonValue, path: []const u8) ?i64 {
        const value = getValue(json, path) orelse return null;
        return switch (value) {
            .int => |i| i,
            else => null,
        };
    }

    /// Get a boolean value from JSON by path
    pub fn getBool(json: JsonValue, path: []const u8) ?bool {
        const value = getValue(json, path) orelse return null;
        return switch (value) {
            .bool => |b| b,
            else => null,
        };
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "JsonUtils basic parsing" {
    const allocator = std.testing.allocator;

    // Test null parsing
    var null_value = try JsonUtils.parse(allocator, "null");
    defer null_value.deinit(allocator);
    try std.testing.expect(null_value == .null);

    // Test boolean parsing
    var bool_value = try JsonUtils.parse(allocator, "true");
    defer bool_value.deinit(allocator);
    try std.testing.expectEqual(true, bool_value.bool);

    // Test string parsing
    var str_value = try JsonUtils.parse(allocator, "\"hello\"");
    defer str_value.deinit(allocator);
    try std.testing.expectEqualStrings("hello", str_value.string);
}

test "JsonUtils stringify" {
    const allocator = std.testing.allocator;

    const str_result = try JsonUtils.stringify(allocator, .{ .string = "test" });
    defer allocator.free(str_result);
    try std.testing.expectEqualStrings("\"test\"", str_result);

    const int_result = try JsonUtils.stringify(allocator, .{ .int = 42 });
    defer allocator.free(int_result);
    try std.testing.expectEqualStrings("42", int_result);
}

test "JsonOps path access" {
    const allocator = std.testing.allocator;

    // Create a test object: {"user": {"name": "Alice", "age": 30}}
    var obj = std.StringHashMap(JsonValue){};
    defer {
        var it = obj.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(allocator);
        }
        obj.deinit(allocator);
    }

    var user_obj = std.StringHashMap(JsonValue){};
    errdefer user_obj.deinit(allocator);

    try user_obj.put(try allocator.dupe(u8, "name"), .{ .string = try allocator.dupe(u8, "Alice") });
    try user_obj.put(try allocator.dupe(u8, "age"), .{ .int = 30 });

    try obj.put(try allocator.dupe(u8, "user"), .{ .object = user_obj });

    const json_value = JsonValue{ .object = obj };

    // Test path access
    const name = JsonOps.getString(json_value, "user.name");
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("Alice", name.?);

    const age = JsonOps.getInt(json_value, "user.age");
    try std.testing.expect(age != null);
    try std.testing.expectEqual(@as(i64, 30), age.?);
}
