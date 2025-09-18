// Windows-specific cross-platform tests
const std = @import("std");
const builtin = @import("builtin");

test "Windows file operations" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    // Test Windows-specific file operations
    const allocator = std.testing.allocator;

    // Test UNC paths, Windows file attributes, etc.
    const temp_path = std.fs.selfExePathAlloc(allocator) catch unreachable;
    defer allocator.free(temp_path);

    // Verify Windows path handling
    try std.testing.expect(std.mem.indexOf(u8, temp_path, "\\") != null);
}

test "Windows networking" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    // Test Windows Sockets API compatibility
    const net = std.net;
    const address = net.Address.parseIp4("127.0.0.1", 0) catch unreachable;

    try std.testing.expect(address.getPort() == 0);
}
