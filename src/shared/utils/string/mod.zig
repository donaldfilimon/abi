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
