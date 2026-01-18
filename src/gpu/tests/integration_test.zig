const std = @import("std");
const device = @import("../device.zig");
const backend_factory = @import("../backend_factory.zig");
const exec_coordinator = @import("../execution_coordinator.zig");

test "full stack: auto-detect → execute → fallback" {
    const allocator = std.testing.allocator;

    // Enumerate all devices
    const all_devices = try device.enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    // Should have at least stdgpu (CPU fallback)
    try std.testing.expect(all_devices.len >= 1);

    // Perform vector operation with execution coordinator
    var coordinator = try exec_coordinator.ExecutionCoordinator.init(allocator, .{});
    defer coordinator.deinit();

    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var result = [_]f32{0} ** 8;

    const method = try coordinator.vectorAdd(&a, &b, &result);

    // Should use some method (not failed)
    try std.testing.expect(method != .failed);

    // Verify results
    try std.testing.expectEqual(@as(f32, 9), result[0]);
    try std.testing.expectEqual(@as(f32, 9), result[7]);
}

test "multi-GPU device selection" {
    const allocator = std.testing.allocator;

    const devices = try device.enumerateAllDevices(allocator);
    defer allocator.free(devices);

    // Should always have at least one device (CPU fallback)
    try std.testing.expect(devices.len >= 1);

    if (devices.len > 1) {
        // Test we can select specific device
        const best = try device.selectBestDevice(allocator, .{
            .prefer_discrete = true,
        });

        if (best) |dev| {
            try std.testing.expect(dev.device_type == .discrete or
                dev.device_type == .integrated or
                dev.device_type == .cpu);
        }
    }
}

test "backend detection and selection" {
    const allocator = std.testing.allocator;

    // Detect all available backends
    const backends = try backend_factory.detectAvailableBackends(allocator);
    defer allocator.free(backends);

    // Should always have stdgpu
    try std.testing.expect(backends.len >= 1);

    var found_stdgpu = false;
    for (backends) |b| {
        if (b == .stdgpu) found_stdgpu = true;
    }
    try std.testing.expect(found_stdgpu);

    // Test backend selection with fallback
    const best = try backend_factory.selectBestBackendWithFallback(allocator, .{
        .fallback_chain = &.{ .cuda, .vulkan, .metal, .stdgpu },
    });

    // Should select something
    try std.testing.expect(best != null);
}

test "device enumeration per backend" {
    const allocator = std.testing.allocator;

    // Test CUDA enumeration (may be empty on non-NVIDIA systems)
    const cuda_devices = try device.enumerateDevicesForBackend(allocator, .cuda);
    defer allocator.free(cuda_devices);

    // All returned devices should be CUDA
    for (cuda_devices) |dev| {
        try std.testing.expectEqual(.cuda, dev.backend);
    }

    // Test stdgpu enumeration (should always work)
    const stdgpu_devices = try device.enumerateDevicesForBackend(allocator, .stdgpu);
    defer allocator.free(stdgpu_devices);

    try std.testing.expect(stdgpu_devices.len >= 1);
    for (stdgpu_devices) |dev| {
        try std.testing.expectEqual(.stdgpu, dev.backend);
        try std.testing.expectEqual(.cpu, dev.device_type);
    }
}

test "execution method fallback chain" {
    const allocator = std.testing.allocator;

    // Test with GPU disabled - should fall back to SIMD/scalar
    var coordinator = try exec_coordinator.ExecutionCoordinator.init(allocator, .{
        .prefer_gpu = false,
        .fallback_chain = &.{ .simd, .scalar },
    });
    defer coordinator.deinit();

    const input_a = [_]f32{ 1, 2, 3, 4 };
    const input_b = [_]f32{ 5, 6, 7, 8 };
    var result = [_]f32{ 0, 0, 0, 0 };

    const method = try coordinator.vectorAdd(&input_a, &input_b, &result);

    // Should not use GPU
    try std.testing.expect(method != .gpu);
    try std.testing.expect(method != .failed);

    // Verify computation correctness
    try std.testing.expectEqual(@as(f32, 6), result[0]);
    try std.testing.expectEqual(@as(f32, 8), result[1]);
    try std.testing.expectEqual(@as(f32, 10), result[2]);
    try std.testing.expectEqual(@as(f32, 12), result[3]);
}

test "device capability queries" {
    const allocator = std.testing.allocator;

    const all_devices = try device.enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    // Check capabilities on each device
    for (all_devices) |dev| {
        // Basic sanity checks
        try std.testing.expect(dev.name.len > 0);
        try std.testing.expect(dev.id >= 0);

        // Score should be reasonable
        const score_val = dev.score();
        try std.testing.expect(score_val > 0);

        // Feature queries should work
        _ = dev.supportsFeature(.int8);
        _ = dev.maxWorkgroupSize();
        _ = dev.maxSharedMemory();
    }
}
