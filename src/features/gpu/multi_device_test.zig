//! Tests for multi-device GPU management.

const std = @import("std");
const device_group_mod = @import("device_group.zig");
const gpu_cluster_mod = @import("gpu_cluster.zig");
const gradient_sync_mod = @import("gradient_sync.zig");

const DeviceGroup = device_group_mod.DeviceGroup;
const GPUCluster = gpu_cluster_mod.GPUCluster;
const GradientBucket = gradient_sync_mod.GradientBucket;
const GradientBucketManager = gradient_sync_mod.GradientBucketManager;

test "device group creation" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{});
    defer group.deinit();

    try std.testing.expect(group.deviceCount() >= 1);
    try std.testing.expect(group.activeDeviceCount() >= 1);
}

test "device group exposes backend metadata" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{});
    defer group.deinit();

    const devices = group.getAllDevices();
    try std.testing.expect(devices.len >= 1);
    for (devices) |device| {
        try std.testing.expect(device.name_len > 0);
        _ = device.backend;
    }
}

test "device selection" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{ .strategy = .round_robin });
    defer group.deinit();

    const id1 = group.selectDevice(1024);
    const id2 = group.selectDevice(1024);

    // With round robin, different calls should potentially select different devices
    // (depends on device count)
    _ = id1;
    _ = id2;
}

test "work distribution" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{ .min_distribute_size = 100 });
    defer group.deinit();

    // Small work - single device
    const dist1 = try group.distributeWork(50);
    defer allocator.free(dist1);
    try std.testing.expectEqual(@as(usize, 1), dist1.len);

    // Larger work - may be distributed
    const dist2 = try group.distributeWork(1000);
    defer allocator.free(dist2);
    try std.testing.expect(dist2.len >= 1);
}

test "group stats" {
    const allocator = std.testing.allocator;
    var group = try DeviceGroup.init(allocator, .{});
    defer group.deinit();

    const stats = group.getStats();
    try std.testing.expect(stats.device_count >= 1);
    try std.testing.expect(stats.total_memory_mb > 0);
}

test "gpu cluster creation" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        // GPU initialization may fail in test environment
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    try std.testing.expect(cluster.deviceCount() >= 1);

    const stats = cluster.getStats();
    try std.testing.expect(stats.device_count >= 1);
}

test "gpu cluster wires backend interfaces for initialized contexts" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    var wired_dispatcher = false;
    var iter = cluster.gpu_contexts.iterator();
    while (iter.next()) |entry| {
        if (entry.value_ptr.*.dispatcher) |dispatcher| {
            if (dispatcher.backend_interface != null) {
                wired_dispatcher = true;
                break;
            }
        }
    }

    try std.testing.expect(wired_dispatcher or cluster.gpu_contexts.count() == 0);
}

test "model partitioning" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const partitions = try cluster.partitionModel(12);
    defer allocator.free(partitions);

    try std.testing.expect(partitions.len >= 1);

    // First partition should start at layer 0
    if (partitions.len > 0) {
        try std.testing.expectEqual(@as(usize, 0), partitions[0].layer_start);
        try std.testing.expect(partitions[0].is_first);
    }

    // Last partition should end at last layer
    if (partitions.len > 0) {
        try std.testing.expect(partitions[partitions.len - 1].is_last);
    }
}

test "tensor partitioning" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const partitions = try cluster.partitionTensor(1024, 256);
    defer allocator.free(partitions);

    try std.testing.expect(partitions.len >= 1);

    // Sum of partition sizes should equal split_dim_size
    var total_size: usize = 0;
    for (partitions) |p| {
        total_size += p.size;
    }
    try std.testing.expectEqual(@as(usize, 256), total_size);
}

test "allreduce operations" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{
        .allreduce_algo = .direct,
    }) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // AllReduce should work (though with 1 device it's a no-op)
    try cluster.allReduce(&data, .sum);
}

test "gradient bucket" {
    const allocator = std.testing.allocator;
    var bucket = try GradientBucket.init(allocator, 100);
    defer bucket.deinit();

    const grad1 = [_]f32{ 0.1, 0.2, 0.3 };
    const grad2 = [_]f32{ 0.4, 0.5 };

    try std.testing.expect(try bucket.add(0, &grad1));
    try std.testing.expect(try bucket.add(1, &grad2));

    try std.testing.expectEqual(@as(usize, 5), bucket.used);

    const extracted = bucket.extractGradient(0).?;
    try std.testing.expectEqual(@as(usize, 3), extracted.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), extracted[0], 1e-6);
}

test "gradient bucket manager" {
    const allocator = std.testing.allocator;
    var manager = try GradientBucketManager.init(allocator, 3, 50);
    defer manager.deinit();

    const grad1 = [_]f32{ 0.1, 0.2, 0.3 };
    try manager.addGradient(0, &grad1);

    // Reset and verify
    manager.reset();
    try std.testing.expectEqual(@as(usize, 0), manager.current_bucket);
}

test "scatter gather" {
    const allocator = std.testing.allocator;
    var cluster = GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const chunks = try cluster.scatter(&data, 3);
    defer {
        for (chunks) |chunk| {
            if (chunk.data.len > 0) {
                allocator.free(chunk.data);
            }
        }
        allocator.free(chunks);
    }

    try std.testing.expect(chunks.len >= 1);

    // Gather back
    var output: [6]f32 = undefined;
    try cluster.gather(chunks, &output);

    // Should have some data
    try std.testing.expect(output[0] == 1.0);
}

test {
    std.testing.refAllDecls(@This());
}
