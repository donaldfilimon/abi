const std = @import("std");

pub fn processInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        "Abi action: {s}\n\nExecuting requested operation with minimal overhead.",
        .{input},
    );
}

test {
    std.testing.refAllDecls(@This());
}

test "abi_profile processInput returns concise response" {
    const allocator = std.testing.allocator;
    const result = try processInput(allocator, "deploy to production");
    defer allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "Abi action") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Executing") != null);
}
