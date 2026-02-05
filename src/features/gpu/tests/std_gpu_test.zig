const std = @import("std");
const std_gpu_integration = @import("../backends/std_gpu_integration.zig");

test "std.gpu device initialization" {
    const allocator = std.testing.allocator;

    var device = try std_gpu_integration.initStdGpuDevice(allocator);
    defer device.deinit();

    // Device should initialize successfully (even if emulated)
    try std.testing.expect(true);
}

test "std.gpu queue creation" {
    const allocator = std.testing.allocator;

    var device = try std_gpu_integration.initStdGpuDevice(allocator);
    defer device.deinit();

    if (std_gpu_integration.isStdGpuAvailable()) {
        var queue = try device.createQueue();
        defer queue.deinit();
    } else {
        // std.gpu not available - this is expected in current Zig versions
        return error.SkipZigTest;
    }
}

test "std.gpu buffer allocation" {
    const allocator = std.testing.allocator;

    var device = try std_gpu_integration.initStdGpuDevice(allocator);
    defer device.deinit();

    if (std_gpu_integration.isStdGpuAvailable()) {
        var buffer = try device.createBuffer(.{
            .size = 1024,
            .usage = .{ .storage = true, .copy_dst = true },
        });
        defer buffer.deinit();

        try std.testing.expectEqual(@as(usize, 1024), buffer.size);
    } else {
        // Use CPU fallback
        var buffer = try device.createBuffer(.{
            .size = 1024,
            .usage = .{ .storage = true, .copy_dst = true },
        });
        defer buffer.deinit();

        try std.testing.expectEqual(@as(usize, 1024), buffer.size);

        // Test buffer read/write
        const test_data = "Hello, GPU!";
        try buffer.write(0, test_data);

        var read_buffer: [32]u8 = undefined;
        try buffer.read(0, read_buffer[0..test_data.len]);

        try std.testing.expectEqualStrings(test_data, read_buffer[0..test_data.len]);
    }
}
