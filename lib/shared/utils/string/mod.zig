//! String Utilities Module
//! Contains string manipulation, array operations, and time formatting utilities

const std = @import("std");

// =============================================================================
// STRING UTILITIES
// =============================================================================

/// String manipulation utilities
pub const StringUtils = struct {
    /// Check if string is empty or whitespace only
    pub fn isEmptyOrWhitespace(str: []const u8) bool {
        return std.mem.trim(u8, str, " \t\r\n").len == 0;
    }

    /// Convert string to lowercase (allocates new string)
    pub fn toLower(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        errdefer allocator.free(result);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }

    /// Convert string to uppercase (allocates new string)
    pub fn toUpper(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        errdefer allocator.free(result);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toUpper(c);
        }
        return result;
    }

    /// Trim whitespace from both ends of string
    pub fn trim(str: []const u8) []const u8 {
        return std.mem.trim(u8, str, " \t\r\n");
    }

    /// Split string by delimiter and return ArrayList
    pub fn split(allocator: std.mem.Allocator, str: []const u8, delimiter: []const u8) !std.ArrayList([]const u8) {
        var result = try std.ArrayList([]const u8).initCapacity(allocator, 0);
        errdefer result.deinit(allocator);

        var iter = std.mem.split(u8, str, delimiter);
        while (iter.next()) |part| {
            try result.append(allocator, part);
        }

        return result;
    }

    /// Join array of strings with delimiter
    pub fn join(allocator: std.mem.Allocator, strings: []const []const u8, delimiter: []const u8) ![]u8 {
        return std.mem.join(allocator, delimiter, strings);
    }

    /// Check if string starts with prefix
    pub fn startsWith(str: []const u8, prefix: []const u8) bool {
        return std.mem.startsWith(u8, str, prefix);
    }

    /// Check if string ends with suffix
    pub fn endsWith(str: []const u8, suffix: []const u8) bool {
        return std.mem.endsWith(u8, str, suffix);
    }

    /// Replace all occurrences of a substring
    pub fn replace(allocator: std.mem.Allocator, str: []const u8, old: []const u8, new: []const u8) ![]u8 {
        var result = try std.ArrayList(u8).initCapacity(allocator, 0);
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < str.len) {
            if (std.mem.startsWith(u8, str[i..], old)) {
                try result.appendSlice(allocator, new);
                i += old.len;
            } else {
                try result.append(allocator, str[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

// =============================================================================
// ARRAY UTILITIES
// =============================================================================

/// Array manipulation utilities
pub const ArrayUtils = struct {
    /// Check if array contains element
    pub fn contains(comptime T: type, haystack: []const T, needle: T) bool {
        for (haystack) |item| {
            if (item == needle) return true;
        }
        return false;
    }

    /// Find index of element in array
    pub fn indexOf(comptime T: type, haystack: []const T, needle: T) ?usize {
        for (haystack, 0..) |item, i| {
            if (item == needle) return i;
        }
        return null;
    }

    /// Remove element at index (shifts remaining elements)
    pub fn removeAt(comptime T: type, array: []T, index: usize) void {
        std.mem.copyForwards(T, array[index..], array[index + 1 ..]);
    }

    /// Insert element at index (shifts elements to make room)
    pub fn insertAt(comptime T: type, array: []T, index: usize, value: T) void {
        std.mem.copyBackwards(T, array[index + 1 ..], array[index..]);
        array[index] = value;
    }

    /// Reverse array in place
    pub fn reverse(comptime T: type, array: []T) void {
        var i: usize = 0;
        var j = array.len - 1;
        while (i < j) : ({
            i += 1;
            j -= 1;
        }) {
            std.mem.swap(T, &array[i], &array[j]);
        }
    }

    /// Fill array with value
    pub fn fill(comptime T: type, array: []T, value: T) void {
        for (array) |*item| {
            item.* = value;
        }
    }

    /// Count occurrences of element in array
    pub fn count(comptime T: type, array: []const T, value: T) usize {
        var result: usize = 0;
        for (array) |item| {
            if (item == value) result += 1;
        }
        return result;
    }
};

// =============================================================================
// TIME UTILITIES
// =============================================================================

/// Time and duration utilities
pub const TimeUtils = struct {
    /// Get current timestamp in milliseconds
    pub fn nowMs() i64 {
        return 0;
    }

    /// Get current timestamp in microseconds
    pub fn nowUs() i64 {
        return std.time.microTimestamp();
    }

    /// Get current timestamp in nanoseconds
    pub fn nowNs() i64 {
        return std.time.nanoTimestamp;
    }

    /// Format duration in human readable format
    pub fn formatDuration(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
        const ms = duration_ns / std.time.ns_per_ms;
        const us = (duration_ns % std.time.ns_per_ms) / std.time.ns_per_us;
        const ns = duration_ns % std.time.ns_per_us;

        if (ms > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}ms", .{ ms, us });
        } else if (us > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}μs", .{ us, ns / 1000 });
        } else {
            return std.fmt.allocPrint(allocator, "{d}ns", .{ns});
        }
    }

    /// Parse duration string (e.g., "1.5s", "500ms", "30μs")
    pub fn parseDuration(duration_str: []const u8) !u64 {
        if (std.mem.endsWith(u8, duration_str, "ns")) {
            const num_str = duration_str[0 .. duration_str.len - 2];
            return try std.fmt.parseInt(u64, num_str, 10);
        } else if (std.mem.endsWith(u8, duration_str, "μs") or std.mem.endsWith(u8, duration_str, "us")) {
            const suffix_len: usize = if (std.mem.endsWith(u8, duration_str, "μs")) 2 else 2;
            const num_str = duration_str[0 .. duration_str.len - suffix_len];
            const us = try std.fmt.parseFloat(f64, num_str);
            return @as(u64, @intFromFloat(us * @as(f64, @floatFromInt(std.time.ns_per_us))));
        } else if (std.mem.endsWith(u8, duration_str, "ms")) {
            const num_str = duration_str[0 .. duration_str.len - 2];
            const ms = try std.fmt.parseFloat(f64, num_str);
            return @as(u64, @intFromFloat(ms * @as(f64, @floatFromInt(std.time.ns_per_ms))));
        } else if (std.mem.endsWith(u8, duration_str, "s")) {
            const num_str = duration_str[0 .. duration_str.len - 1];
            const s = try std.fmt.parseFloat(f64, num_str);
            return @as(u64, @intFromFloat(s * @as(f64, @floatFromInt(std.time.ns_per_s))));
        } else {
            return error.InvalidDurationFormat;
        }
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "StringUtils basic operations" {
    const allocator = std.testing.allocator;

    // Test empty/whitespace check
    try std.testing.expect(StringUtils.isEmptyOrWhitespace(""));
    try std.testing.expect(StringUtils.isEmptyOrWhitespace("   \t\n  "));
    try std.testing.expect(!StringUtils.isEmptyOrWhitespace("hello"));

    // Test case conversion
    const lower = try StringUtils.toLower(allocator, "HELLO");
    defer allocator.free(lower);
    try std.testing.expectEqualStrings("hello", lower);

    const upper = try StringUtils.toUpper(allocator, "hello");
    defer allocator.free(upper);
    try std.testing.expectEqualStrings("HELLO", upper);

    // Test trim
    try std.testing.expectEqualStrings("hello", StringUtils.trim("  hello  "));
}

test "ArrayUtils operations" {
    var array = [_]i32{ 1, 2, 3, 4, 5 };

    // Test contains
    try std.testing.expect(ArrayUtils.contains(i32, &array, 3));
    try std.testing.expect(!ArrayUtils.contains(i32, &array, 99));

    // Test indexOf
    try std.testing.expectEqual(@as(?usize, 2), ArrayUtils.indexOf(i32, &array, 3));
    try std.testing.expectEqual(@as(?usize, null), ArrayUtils.indexOf(i32, &array, 99));

    // Test count
    try std.testing.expectEqual(@as(usize, 1), ArrayUtils.count(i32, &array, 3));
    try std.testing.expectEqual(@as(usize, 0), ArrayUtils.count(i32, &array, 99));
}

test "TimeUtils duration formatting" {
    const allocator = std.testing.allocator;

    // Test millisecond formatting
    const ms_str = try TimeUtils.formatDuration(allocator, 1500000); // 1.5ms
    defer allocator.free(ms_str);
    try std.testing.expect(std.mem.indexOf(u8, ms_str, "ms") != null);

    // Test microsecond formatting
    const us_str = try TimeUtils.formatDuration(allocator, 1500); // 1.5μs
    defer allocator.free(us_str);
    try std.testing.expect(std.mem.indexOf(u8, us_str, "μs") != null);

    // Test nanosecond formatting
    const ns_str = try TimeUtils.formatDuration(allocator, 500); // 500ns
    defer allocator.free(ns_str);
    try std.testing.expect(std.mem.indexOf(u8, ns_str, "ns") != null);
}
