// macOS cross-platform tests
const std = @import("std");
const builtin = @import("builtin");

test "macOS file operations" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    // Test macOS-specific file operations
    // Note: allocator is not needed in this test block
    // const allocator = std.testing.allocator;

    // Test macOS path conventions
    const home_dir = std.os.getenv("HOME") orelse return error.SkipZigTest;

    try std.testing.expect(std.mem.startsWith(u8, home_dir, "/Users/"));
}

test "macOS networking" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    // Test macOS networking stack
    const net = std.net;

    // Test local address resolution
    const allocator = std.testing.allocator;
    const addresses = try net.getAddressList(allocator, "localhost", 80);
    defer addresses.deinit();

    try std.testing.expect(addresses.addrs.len > 0);
}
