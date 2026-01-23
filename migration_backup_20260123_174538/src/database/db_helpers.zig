const std = @import("std");

pub const HelperError = error{
    InvalidVector,
};

pub fn parseVector(allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    var list = std.ArrayListUnmanaged(f32).empty;
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |chunk| {
        const trimmed = std.mem.trim(u8, chunk, " \t\r\n");
        if (trimmed.len == 0) continue;
        const value = std.fmt.parseFloat(f32, trimmed) catch return HelperError.InvalidVector;
        try list.append(allocator, value);
    }
    if (list.items.len == 0) return HelperError.InvalidVector;
    return list.toOwnedSlice(allocator);
}

pub fn formatVector(allocator: std.mem.Allocator, vector: []const f32) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);
    for (vector, 0..) |value, i| {
        if (i > 0) try list.append(allocator, ',');
        try list.print(allocator, "{d:.4}", .{value});
    }
    return list.toOwnedSlice(allocator);
}

test "vector parse and format" {
    const allocator = std.testing.allocator;
    const vector = try parseVector(allocator, "1.0, 2.5,3");
    defer allocator.free(vector);
    try std.testing.expectEqual(@as(usize, 3), vector.len);
    try std.testing.expect(std.math.approxEqAbs(f32, vector[1], 2.5, 0.0001));

    const formatted = try formatVector(allocator, vector);
    defer allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "1.0000") != null);
}

test "vector parse rejects empty input" {
    try std.testing.expectError(HelperError.InvalidVector, parseVector(std.testing.allocator, ""));
}
