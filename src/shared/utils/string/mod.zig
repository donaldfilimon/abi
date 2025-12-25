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
