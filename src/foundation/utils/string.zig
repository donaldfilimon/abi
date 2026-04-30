const std = @import("std");

/// Convert a string to lowercase ASCII (allocating).
///
/// Returns a newly allocated string that must be freed by the caller.
/// Non-ASCII characters are copied unchanged.
pub fn toLowerAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const copy = try allocator.alloc(u8, input.len);
    for (input, 0..) |char, i| {
        copy[i] = std.ascii.toLower(char);
    }
    return copy;
}

/// Convert a string to uppercase ASCII (allocating).
///
/// Returns a newly allocated string that must be freed by the caller.
/// Non-ASCII characters are copied unchanged.
pub fn toUpperAscii(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const copy = try allocator.alloc(u8, input.len);
    for (input, 0..) |char, i| {
        copy[i] = std.ascii.toUpper(char);
    }
    return copy;
}

test "toLowerAscii lowercases ASCII text" {
    const got = try toLowerAscii(std.testing.allocator, "HeLLo");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("hello", got);
}

test "toUpperAscii uppercases ASCII text" {
    const got = try toUpperAscii(std.testing.allocator, "HeLLo");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("HELLO", got);
}

test {
    std.testing.refAllDecls(@This());
}
