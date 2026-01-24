const std = @import("std");
const device = @import("../device.zig");
const backend_factory = @import("../backend_factory.zig");
const exec_coordinator = @import("../execution_coordinator.zig");
const multi_device = @import("../multi_device.zig");
const GPUCluster = multi_device.GPUCluster;
const DeviceId = multi_device.DeviceId;

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

test "multi-GPU scheduling: round-robin load balancing" {
    const allocator = std.testing.allocator;
    var group = try multi_device.DeviceGroup.init(allocator, .{
        .strategy = .round_robin,
        .min_distribute_size = 1000,
    });
    defer group.deinit();

    const device_count = group.activeDeviceCount();
    try std.testing.expect(device_count >= 1);

    // Test round-robin selection
    var selected_devices = std.ArrayList(DeviceId).init(allocator);
    defer selected_devices.deinit();

    for (0..device_count * 3) |_| {
        const device_id = group.selectDevice(1024);
        try selected_devices.append(device_id);
    }

    // Should cycle through available devices
    for (selected_devices.items, 0..) |dev_id, i| {
        const expected_idx = i % device_count;
        const expected_dev = group.active_devices.items[expected_idx];
        try std.testing.expectEqual(expected_dev, dev_id);
    }
}

test "multi-GPU scheduling: memory-aware load balancing" {
    const allocator = std.testing.allocator;
    var group = try multi_device.DeviceGroup.init(allocator, .{
        .strategy = .memory_aware,
    });
    defer group.deinit();

    const dist = try group.distributeWork(5000);
    defer allocator.free(dist);

    // Work should be distributed appropriately
    try std.testing.expect(dist.len >= 1);
    var total_size: usize = 0;
    for (dist) |d| {
        total_size += d.size;
    }
    try std.testing.expectEqual(@as(usize, 5000), total_size);
}

test "multi-GPU scheduling: capability-weighted distribution" {
    const allocator = std.testing.allocator;
    var group = try multi_device.DeviceGroup.init(allocator, .{
        .strategy = .capability_weighted,
    });
    defer group.deinit();

    const large_workload = 10000;
    const dist = try group.distributeWork(large_workload);
    defer allocator.free(dist);

    try std.testing.expect(dist.len >= 1);

    // Verify distribution sums to total workload
    var distributed_size: usize = 0;
    for (dist) |d| {
        distributed_size += d.size;
    }
    try std.testing.expectEqual(large_workload, distributed_size);
}

test "multi-GPU scheduling: device enable/disable" {
    const allocator = std.testing.allocator;
    var group = try multi_device.DeviceGroup.init(allocator, .{});
    defer group.deinit();

    const initial_count = group.activeDeviceCount();

    // Test disabling devices
    if (initial_count > 1) {
        const device_to_disable = group.active_devices.items[0];
        group.disableDevice(device_to_disable);
        try std.testing.expectEqual(initial_count - 1, group.activeDeviceCount());

        // Re-enable the device
        try group.enableDevice(device_to_disable);
        try std.testing.expectEqual(initial_count, group.activeDeviceCount());
    }
}

test "multi-GPU scheduling: workload validation cross-device" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{
        .parallelism = .data_parallel,
    }) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const work_distribution = try cluster.distributeWork(8192);
    defer allocator.free(work_distribution);

    // Validate distribution
    var total_size: usize = 0;
    var valid_devices: usize = 0;

    for (work_distribution) |dist| {
        if (cluster.getContext(dist.device_id) != null) {
            valid_devices += 1;
        }
        total_size += dist.size;

        // Each distribution should have valid offsets
        try std.testing.expect(dist.size > 0);
    }

    try std.testing.expectEqual(@as(usize, 8192), total_size);
    try std.testing.expect(valid_devices > 0);
}

test "multi-GPU scheduling: fault tolerance and recovery" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{
        .parallelism = .data_parallel,
    }) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const initial_device_count = cluster.deviceCount();

    // Simulate device failure by disabling devices
    if (initial_device_count > 1) {
        const failing_device = cluster.device_group.active_devices.items[0];
        cluster.device_group.disableDevice(failing_device);

        // Cluster should still function with remaining devices
        const work_distribution = try cluster.distributeWork(4096);
        defer allocator.free(work_distribution);

        try std.testing.expectEqual(initial_device_count - 1, work_distribution.len);

        // Verify no work assigned to failed device
        for (work_distribution) |dist| {
            try std.testing.expect(dist.device_id != failing_device);
        }
    }
}
