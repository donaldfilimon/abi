const std = @import("std");

pub fn escapeString(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var list = std.ArrayList(u8).empty;
    errdefer list.deinit(allocator);
    try list.append(allocator, '"');
    for (input) |char| {
        switch (char) {
            '"' => try list.appendSlice(allocator, "\\\""),
            '\\' => try list.appendSlice(allocator, "\\\\"),
            '\n' => try list.appendSlice(allocator, "\\n"),
            '\r' => try list.appendSlice(allocator, "\\r"),
            '\t' => try list.appendSlice(allocator, "\\t"),
            else => try list.append(allocator, char),
        }
    }
    try list.append(allocator, '"');
    return list.toOwnedSlice(allocator);
}

pub fn writeString(writer: anytype, input: []const u8) !void {
    try writer.writeAll("\"");
    for (input) |char| {
        switch (char) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(char),
        }
    }
    try writer.writeAll("\"");
}

test "json string escape" {
    const allocator = std.testing.allocator;
    const escaped = try escapeString(allocator, "a\"b\n");
    defer allocator.free(escaped);
    try std.testing.expectEqualStrings("\"a\\\"b\\n\"", escaped);

    var storage: [64]u8 = undefined;
    var stream = std.io.fixedBufferStream(&storage);
    try writeString(stream.writer(), "a\"b\n");
    try std.testing.expectEqualStrings("\"a\\\"b\\n\"", stream.getWritten());
}
