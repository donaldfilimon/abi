const std = @import("std");

pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        "Aviva creative exploration: {s}\n\nExploring multiple perspectives and creative angles for this topic...",
        .{input},
    );
}

test {
    std.testing.refAllDecls(@This());
}

test "aviva processInput returns creative response" {
    const allocator = std.testing.allocator;
    const result = try processInput(allocator, "what is consciousness?");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Aviva creative exploration") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "creative") != null);
}
