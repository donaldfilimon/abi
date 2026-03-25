//! JSON utility helpers for the ACP server.

const std = @import("std");

/// Escape a string for safe embedding in JSON output.
pub fn appendEscaped(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var hex_buf: [6]u8 = undefined;
                    const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch {
                        continue; // skip unprintable character on format error
                    };
                    try buf.appendSlice(allocator, hex);
                } else {
                    try buf.append(allocator, c);
                }
            },
        }
    }
}

test "appendEscaped handles all special chars" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try appendEscaped(allocator, &buf, "a\"b\\c\nd\re");
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd\\re", buf.items);
}

test "appendEscaped handles control characters" {
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    // Test tab and control char below 0x20
    try appendEscaped(allocator, &buf, "a\tb\x01c");
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\t") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\u") != null);
}
