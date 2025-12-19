// Windows-specific cross-platform tests
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const ai = @import("../../lib/mod.zig");

test "windows GPU acceleration" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test GPU operations on Windows
    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Basic GPU memory test
    const mem = try accel.alloc(1024);
    defer accel.free(&mem);

    try testing.expect(mem.size >= 1024);
}

test "windows CUDA GPU operations" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Skip if not CUDA
    if (accel.backend != .cuda) return error.SkipZigTest;

    // Test CUDA memory operations
    const size = 1024 * @sizeOf(f32);
    const mem = try accel.alloc(size);
    defer accel.free(&mem);

    // Test data transfer
    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try accel.copyToDevice(mem, std.mem.sliceAsBytes(&host_data));

    var read_back = [_]f32{0} ** 4;
    try accel.copyFromDevice(std.mem.sliceAsBytes(&read_back), mem);

    try testing.expectApproxEqAbs(host_data[0], read_back[0], 0.001);
}

test "windows file I/O" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    // Test Windows-specific file operations
    const path = "test_windows.tmp";

    // Create a test file
    const file = try std.fs.createFileAbsolute(path, .{});
    defer {
        file.close();
        std.fs.deleteFileAbsolute(path) catch {};
    }

    try file.writeAll("test data");
}

test "windows networking" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    // Test basic networking on Windows
    const address = try std.net.Address.parseIp4("127.0.0.1", 8080);
    try testing.expect(address.getPort() == 8080);
}

test "Windows networking" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;

    // Test Windows Sockets API compatibility
    const net = std.net;
    const address = net.Address.parseIp4("127.0.0.1", 0) catch unreachable;

    try std.testing.expect(address.getPort() == 0);
}
