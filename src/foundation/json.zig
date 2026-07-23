const std = @import("std");

/// Appends the escaped body of `value` without surrounding quotes.
fn appendEscapedBody(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
}

pub fn appendJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    try appendEscapedBody(out, allocator, value);
    try out.append(allocator, '"');
}

pub fn strField(v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

pub fn escapeJsonString(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendJsonString(&out, allocator, value);
    return try out.toOwnedSlice(allocator);
}

pub fn jsonStringAlloc(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    return escapeJsonString(allocator, value);
}

pub fn escapeJsonStringRaw(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendEscapedBody(&out, allocator, value);
    return try out.toOwnedSlice(allocator);
}

pub fn writeJsonString(writer: anytype, text: []const u8) !void {
    try writer.writeAll("\"");
    const hex = "0123456789abcdef";
    for (text) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x08 => try writer.writeAll("\\b"),
            0x0c => try writer.writeAll("\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => {
                const escaped = [_]u8{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 0x0f] };
                try writer.writeAll(&escaped);
            },
            else => try writer.writeAll(&.{byte}),
        }
    }
    try writer.writeAll("\"");
}

test "appendJsonString escapes metacharacters and control chars" {
    const allocator = std.testing.allocator;

    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(allocator);
    try appendJsonString(&out, allocator, "a\"b\\c\n\t");
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\n\\t\"", out.items);

    out.clearRetainingCapacity();
    try appendJsonString(&out, allocator, "\x01");
    try std.testing.expectEqualStrings("\"\\u0001\"", out.items);
}

test "escapeJsonString returns an owned slice" {
    const allocator = std.testing.allocator;
    const out = try escapeJsonString(allocator, "hello\nworld");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("\"hello\\nworld\"", out);
}

test {
    std.testing.refAllDecls(@This());
}
