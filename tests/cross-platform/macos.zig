// macOS cross-platform tests
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const ai = @import("../../lib/mod.zig");

test "macos Metal acceleration" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test Metal GPU operations on macOS
    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Test Metal-specific features
    const backend = accel.getBackendType();
    try testing.expect(backend == .metal or backend == .cpu); // Fallback to CPU if Metal unavailable
}

test "macos file system" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    // Test macOS-specific file system features
    const home_dir = std.os.getenv("HOME") orelse return error.SkipZigTest;
    const test_path = try std.fs.path.join(testing.allocator, &[_][]const u8{ home_dir, "test_macos.tmp" });
    defer testing.allocator.free(test_path);
    defer std.fs.deleteFileAbsolute(test_path) catch {};

    const file = try std.fs.createFileAbsolute(test_path, .{});
    defer file.close();

    try file.writeAll("macOS test data");
    try file.seekTo(0);
    var buffer: [100]u8 = undefined;
    const bytes_read = try file.read(&buffer);
    try testing.expect(std.mem.eql(u8, buffer[0..bytes_read], "macOS test data"));
}

test "macos networking" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    // Test networking with macOS-specific considerations
    const address = try std.net.Address.parseIp4("127.0.0.1", 8080);
    try testing.expect(address.getPort() == 8080);

    // Test getaddrinfo (macOS might have different behavior)
    const peer = try std.net.tcpConnectToAddress(address);
    defer peer.close();

    try testing.expect(peer.stream.handle != std.fs.INVALID_FILE_DESCRIPTOR);
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
