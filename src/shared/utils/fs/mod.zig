const std = @import("std");

pub fn join(allocator: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    if (parts.len == 0) return allocator.alloc(u8, 0);
    var total: usize = 0;
    for (parts) |part| {
        total += part.len;
    }
    total += parts.len - 1;

    const output = try allocator.alloc(u8, total);
    var index: usize = 0;
    for (parts, 0..) |part, i| {
        std.mem.copyForwards(u8, output[index..][0..part.len], part);
        index += part.len;
        if (i + 1 < parts.len) {
            output[index] = std.fs.path.sep;
            index += 1;
        }
    }
    return output;
}

pub fn hasExtension(path: []const u8, extension: []const u8) bool {
    if (extension.len == 0) return false;
    return std.mem.endsWith(u8, path, extension);
}

pub fn basename(path: []const u8) []const u8 {
    return std.fs.path.basename(path);
}

test "fs helpers" {
    const allocator = std.testing.allocator;
    const expected = try std.fmt.allocPrint(
        allocator,
        "a{c}b{c}c",
        .{std.fs.path.sep},
    );
    defer allocator.free(expected);

    const joined = try join(allocator, &.{ "a", "b", "c" });
    defer allocator.free(joined);
    try std.testing.expectEqualStrings(expected, joined);
    try std.testing.expect(hasExtension("file.zig", ".zig"));
    const path = try std.fmt.allocPrint(allocator, "dir{c}file.zig", .{std.fs.path.sep});
    defer allocator.free(path);
    try std.testing.expectEqualStrings("file.zig", basename(path));
}
