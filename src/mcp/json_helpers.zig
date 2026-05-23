const std = @import("std");

pub fn appendJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
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
    try out.append(allocator, '"');
}

pub fn jsonStringAlloc(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendJsonString(&out, allocator, value);
    return out.toOwnedSlice(allocator);
}

pub fn valueToJson(value: std.json.Value, allocator: std.mem.Allocator) ![]u8 {
    return switch (value) {
        .null => try allocator.dupe(u8, "null"),
        .bool => |b| try allocator.dupe(u8, if (b) "true" else "false"),
        .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(allocator, "{d}", .{f}),
        .number_string => |s| try allocator.dupe(u8, s),
        .string => |s| try jsonStringAlloc(allocator, s),
        .array => |arr| blk: {
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(allocator);
            try out.append(allocator, '[');
            for (arr.items, 0..) |item, idx| {
                if (idx > 0) try out.append(allocator, ',');
                const item_json = try valueToJson(item, allocator);
                defer allocator.free(item_json);
                try out.appendSlice(allocator, item_json);
            }
            try out.append(allocator, ']');
            break :blk out.toOwnedSlice(allocator);
        },
        .object => |obj| blk: {
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(allocator);
            try out.append(allocator, '{');
            var it = obj.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try out.append(allocator, ',');
                first = false;
                try appendJsonString(&out, allocator, entry.key_ptr.*);
                try out.append(allocator, ':');
                const val_json = try valueToJson(entry.value_ptr.*, allocator);
                defer allocator.free(val_json);
                try out.appendSlice(allocator, val_json);
            }
            try out.append(allocator, '}');
            break :blk out.toOwnedSlice(allocator);
        },
    };
}
