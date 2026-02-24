//! Shared tokenization utilities for evaluation metrics.

const std = @import("std");

/// Tokenize text by whitespace, returning slices into the original text.
pub fn tokenize(allocator: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var tokens = std.ArrayListUnmanaged([]const u8).empty;
    errdefer tokens.deinit(allocator);

    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) : (i += 1) {
        if (std.ascii.isWhitespace(text[i])) {
            if (i > start) {
                try tokens.append(allocator, text[start..i]);
            }
            start = i + 1;
        }
    }

    if (start < text.len) {
        try tokens.append(allocator, text[start..]);
    }

    return tokens.toOwnedSlice(allocator);
}

/// Count tokens without allocating.
pub fn countTokens(text: []const u8) usize {
    var count: usize = 0;
    var in_word = false;

    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (in_word) {
                count += 1;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }

    if (in_word) count += 1;
    return count;
}

test "tokenize basic" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "the cat sat");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqualStrings("the", tokens[0]);
    try std.testing.expectEqualStrings("cat", tokens[1]);
    try std.testing.expectEqualStrings("sat", tokens[2]);
}

test "tokenize empty" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "");
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "tokenize multiple spaces" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "  hello   world  ");
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 2), tokens.len);
}

test "count tokens" {
    try std.testing.expectEqual(@as(usize, 3), countTokens("the cat sat"));
    try std.testing.expectEqual(@as(usize, 0), countTokens(""));
    try std.testing.expectEqual(@as(usize, 2), countTokens("  hello   world  "));
}

test {
    std.testing.refAllDecls(@This());
}
