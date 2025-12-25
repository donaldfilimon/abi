const std = @import("std");

pub const SplitPair = struct {
    head: []const u8,
    tail: []const u8,
};

pub fn trimWhitespace(input: []const u8) []const u8 {
    return std.mem.trim(u8, input, " \t\r\n");
}

pub fn splitOnce(input: []const u8, delimiter: u8) ?SplitPair {
    const index = std.mem.indexOfScalar(u8, input, delimiter) orelse return null;
    return .{
        .head = input[0..index],
        .tail = input[index + 1 ..],
    };
}

pub fn parseBool(input: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(input, "true") or std.mem.eql(u8, input, "1")) {
        return true;
    }
    if (std.ascii.eqlIgnoreCase(input, "false") or std.mem.eql(u8, input, "0")) {
        return false;
    }
    return null;
}

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
