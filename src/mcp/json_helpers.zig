const std = @import("std");

const foundation_json = @import("abi").foundation.json;

pub const appendJsonString = foundation_json.appendJsonString;

pub fn jsonStringAlloc(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
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
            var out: std.ArrayListUnmanaged(u8) = .empty;
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
            var out: std.ArrayListUnmanaged(u8) = .empty;
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

test "json_helpers: jsonStringAlloc escapes metacharacters and control chars" {
    const allocator = std.testing.allocator;

    const out = try jsonStringAlloc(allocator, "a\"b\\c\n\t");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("\"a\\\"b\\\\c\\n\\t\"", out);

    const ctrl = try jsonStringAlloc(allocator, "\x01");
    defer allocator.free(ctrl);
    try std.testing.expectEqualStrings("\"\\u0001\"", ctrl);
}

test "json_helpers: valueToJson serializes a value that re-parses cleanly" {
    const allocator = std.testing.allocator;

    const src = "{\"k\":[1,2.5,true,null,\"q\\\"x\"]}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, src, .{});
    defer parsed.deinit();

    const out = try valueToJson(parsed.value, allocator);
    defer allocator.free(out);

    const reparsed = try std.json.parseFromSlice(std.json.Value, allocator, out, .{});
    defer reparsed.deinit();
    const arr = reparsed.value.object.get("k").?.array;
    try std.testing.expectEqual(@as(usize, 5), arr.items.len);
    try std.testing.expectEqualStrings("q\"x", arr.items[4].string);
}

test {
    std.testing.refAllDecls(@This());
}
