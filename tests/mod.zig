const std = @import("std");
const abi = @import("abi");

test "abi version returns non-empty string" {
    try std.testing.expect(abi.version().len > 0);
}

test "framework init and shutdown" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    try std.testing.expect(!framework.isRunning());
}
