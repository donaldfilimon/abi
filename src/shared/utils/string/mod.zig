/// Generic string manipulation helpers.
///
/// This module purposely keeps allocations to a minimum.
/// All functions operate on slices and return slices unless a new allocation
/// is explicitly requested by the caller.
const std = @import("std");

pub const SplitPair = struct {
    head: []const u8,
    tail: []const u8,
};

/// Trim leading and trailing ASCII whitespace.
///
/// The slice returned shares storage with the original input; no heap
/// allocation is performed.
pub fn trimWhitespace(input: []const u8) []const u8 {
    return std.mem.trim(u8, input, " \t\r\n");
}

/// Split `input` at the first occurrence of `delimiter`.
///
/// Returns `null` if the delimiter does not occur.
pub fn splitOnce(input: []const u8, delimiter: u8) ?SplitPair {
    const pair = std.mem.splitOnce(u8, input, delimiter);
    return pair;
}

/// Parse a boolean value from a string slice.
///
/// Recognizes "true" / "false" (case‑insensitive) and "1" / "0".
pub fn parseBool(input: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(input, "true") or std.mem.eql(u8, input, "1")) {
        return true;
    }
    if (std.ascii.eqlIgnoreCase(input, "false") or std.mem.eql(u8, input, "0")) {
        return false;
    }
    return null;
}

/// Return a new slice containing the ASCII lowercase representation.
///
/// The caller must free the returned slice using the supplied allocator.
pub fn toLowerAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const copy = try allocator.alloc(u8, input.len);
    for (input, 0..) |char, i| {
        copy[i] = std.ascii.toLower(char);
    }
    return copy;
}

/// In-place lowercase conversion (mutates buffer).
/// @param buf Buffer to convert (modified in-place)
/// @return The same buffer (for chaining)
pub fn lowerStringMut(buf: []u8) []u8 {
    for (buf, 0..) |c, i| {
        buf[i] = std.ascii.toLower(c);
    }
    return buf;
}

/// Case-insensitive equality check (zero allocation).
/// @param a First string
/// @param b Second string
/// @return true if strings are equal ignoring case
pub inline fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ac, bc| {
        if (std.ascii.toLower(ac) != std.ascii.toLower(bc)) {
            return false;
        }
    }
    return true;
}

/// Hash for case-insensitive maps.
/// @param s String to hash
/// @return Hash value
pub fn hashIgnoreCase(s: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(0);
    for (s) |c| {
        hasher.update(&[_]u8{std.ascii.toLower(c)});
    }
    return hasher.final();
}

/// Case-insensitive comparator for sorting.
/// @param context Unused context
/// @param a First string
/// @param b Second string
/// @return Ordering relationship
pub fn orderIgnoreCase(_: void, a: []const u8, b: []const u8) std.math.Order {
    const min_len = @min(a.len, b.len);
    for (a[0..min_len], b[0..min_len]) |ac, bc| {
        const al = std.ascii.toLower(ac);
        const bl = std.ascii.toLower(bc);
        if (al < bl) return .lt;
        if (al > bl) return .gt;
    }
    return std.math.order(a.len, b.len);
}

/// Case-insensitive string search.
/// @param haystack String to search in
/// @param needle String to search for
/// @return Index of first match, or null if not found
pub fn indexOfIgnoreCase(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;

    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (eqlIgnoreCase(haystack[i..][0..needle.len], needle)) {
            return i;
        }
    }
    return null;
}

test "string helpers" {
    try std.testing.expectEqualStrings("hello", trimWhitespace("  hello \r\n"));

    const pair = splitOnce("a=b", '=') orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("a", pair.head);
    try std.testing.expectEqualStrings("b", pair.tail);

    try std.testing.expectEqual(@as(?bool, true), parseBool("TRUE"));
    try std.testing.expectEqual(@as(?bool, false), parseBool("0"));
    try std.testing.expectEqual(@as(?bool, null), parseBool("maybe"));

    const lower = try toLowerAscii(std.testing.allocator, "HeLLo");
    defer std.testing.allocator.free(lower);
    try std.testing.expectEqualStrings("hello", lower);
}

test "case-insensitive utilities" {
    var buf = "HeLLo WoRLd".*;
    const result = lowerStringMut(&buf);
    try std.testing.expectEqualStrings("hello world", result);

    try std.testing.expect(eqlIgnoreCase("Hello", "hello"));
    try std.testing.expect(eqlIgnoreCase("WORLD", "world"));
    try std.testing.expect(!eqlIgnoreCase("Hello", "world"));

    const h1 = hashIgnoreCase("Hello");
    const h2 = hashIgnoreCase("hello");
    try std.testing.expectEqual(h1, h2);

    try std.testing.expectEqual(std.math.Order.eq, orderIgnoreCase({}, "Hello", "hello"));
    try std.testing.expectEqual(std.math.Order.lt, orderIgnoreCase({}, "apple", "BANANA"));

    try std.testing.expectEqual(@as(?usize, 0), indexOfIgnoreCase("Hello World", "hello"));
    try std.testing.expectEqual(@as(?usize, 6), indexOfIgnoreCase("Hello World", "WORLD"));
    try std.testing.expectEqual(@as(?usize, null), indexOfIgnoreCase("Hello", "WORLD"));
}
