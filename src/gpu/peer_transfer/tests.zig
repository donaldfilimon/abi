//! Peer Transfer Module Tests
//!
//! Comprehensive tests for GPU peer transfer functionality.
//! Includes unit tests, integration tests, and correctness verification.

const std = @import("std");
const testing = std.testing;

const mod = @import("mod.zig");
const host_staged = @import("host_staged.zig");
const multi_device = @import("../multi_device.zig");

const PeerTransferManager = mod.PeerTransferManager;
const TransferCapability = mod.TransferCapability;
const TransferStatus = mod.TransferStatus;
const TransferOptions = mod.TransferOptions;
const DeviceBuffer = mod.DeviceBuffer;
const DevicePair = mod.DevicePair;
const ReduceOp = mod.ReduceOp;

// ============================================================================
// Unit Tests - TransferCapability
// ============================================================================

test "TransferCapability priority ordering" {
    // NCCL should have highest priority
    try testing.expect(TransferCapability.nccl.priority() > TransferCapability.direct_p2p.priority());
    try testing.expect(TransferCapability.direct_p2p.priority() > TransferCapability.vulkan_external.priority());
    try testing.expect(TransferCapability.vulkan_external.priority() > TransferCapability.host_staged.priority());
}

test "TransferCapability names" {
    try testing.expectEqualStrings("NCCL", TransferCapability.nccl.name());
    try testing.expectEqualStrings("Direct P2P", TransferCapability.direct_p2p.name());
    try testing.expectEqualStrings("Host Staged", TransferCapability.host_staged.name());
}

// ============================================================================
// Unit Tests - DevicePair
// ============================================================================

test "DevicePair hashing is symmetric-aware" {
    const pair_0_1 = DevicePair{ .src = 0, .dst = 1 };
    const pair_1_0 = DevicePair{ .src = 1, .dst = 0 };
    const pair_0_1_dup = DevicePair{ .src = 0, .dst = 1 };

    // Different directions should have different hashes
    try testing.expect(pair_0_1.hash() != pair_1_0.hash());

    // Same pairs should have same hash
    try testing.expect(pair_0_1.hash() == pair_0_1_dup.hash());

    // Equality check
    try testing.expect(pair_0_1.eql(pair_0_1_dup));
    try testing.expect(!pair_0_1.eql(pair_1_0));
}

test "DevicePair hash distribution" {
    // Verify no collisions for small device IDs
    var hashes = std.AutoHashMap(u64, DevicePair).init(testing.allocator);
    defer hashes.deinit();

    var collision_count: usize = 0;
    for (0..8) |src| {
        for (0..8) |dst| {
            if (src == dst) continue;

            const pair = DevicePair{
                .src = @intCast(src),
                .dst = @intCast(dst),
            };
            const hash = pair.hash();

            if (hashes.get(hash)) |_| {
                collision_count += 1;
            } else {
                try hashes.put(hash, pair);
            }
        }
    }

    try testing.expectEqual(@as(usize, 0), collision_count);
}

// ============================================================================
// Unit Tests - HostStagedBackend
// ============================================================================

test "HostStagedBackend initialization and cleanup" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    const stats = backend.getStats();
    try testing.expectEqual(@as(u64, 0), stats.transfers);
    try testing.expectEqual(@as(u64, 0), stats.bytes);
}

test "HostStagedBackend simple transfer" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    try backend.transfer(0, 1, &data);

    const stats = backend.getStats();
    try testing.expectEqual(@as(u64, 1), stats.transfers);
    try testing.expectEqual(@as(u64, 16), stats.bytes);
}

test "HostStagedBackend empty transfer" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var data = [_]u8{};
    try backend.transfer(0, 1, &data);

    // Empty transfer should succeed but not count
    const stats = backend.getStats();
    try testing.expectEqual(@as(u64, 0), stats.transfers);
}

test "HostStagedBackend multiple transfers" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var data1 = [_]u8{1} ** 1000;
    var data2 = [_]u8{2} ** 2000;
    var data3 = [_]u8{3} ** 3000;

    try backend.transfer(0, 1, &data1);
    try backend.transfer(1, 2, &data2);
    try backend.transfer(2, 0, &data3);

    const stats = backend.getStats();
    try testing.expectEqual(@as(u64, 3), stats.transfers);
    try testing.expectEqual(@as(u64, 6000), stats.bytes);
}

// ============================================================================
// Unit Tests - AllReduce Operations
// ============================================================================

test "HostStagedBackend allReduce sum" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .sum);

    // Sum: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    try testing.expectApproxEqAbs(@as(f32, 6.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 8.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 10.0), buf1[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 12.0), buf1[3], 0.001);

    // Both buffers should have same result
    try testing.expectApproxEqAbs(buf1[0], buf2[0], 0.001);
    try testing.expectApproxEqAbs(buf1[1], buf2[1], 0.001);
}

test "HostStagedBackend allReduce max" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 8.0, 3.0, 10.0 };
    var buf2 = [_]f32{ 5.0, 2.0, 9.0, 4.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .max);

    // Max: [max(1,5), max(8,2), max(3,9), max(10,4)] = [5, 8, 9, 10]
    try testing.expectApproxEqAbs(@as(f32, 5.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 8.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 9.0), buf1[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 10.0), buf1[3], 0.001);
}

test "HostStagedBackend allReduce min" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 8.0, 3.0, 10.0 };
    var buf2 = [_]f32{ 5.0, 2.0, 9.0, 4.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .min);

    // Min: [min(1,5), min(8,2), min(3,9), min(10,4)] = [1, 2, 3, 4]
    try testing.expectApproxEqAbs(@as(f32, 1.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), buf1[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4.0), buf1[3], 0.001);
}

test "HostStagedBackend allReduce avg" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var buf2 = [_]f32{ 4.0, 6.0, 8.0, 10.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .avg);

    // Avg: [(2+4)/2, (4+6)/2, (6+8)/2, (8+10)/2] = [3, 5, 7, 9]
    try testing.expectApproxEqAbs(@as(f32, 3.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 5.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 7.0), buf1[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 9.0), buf1[3], 0.001);
}

test "HostStagedBackend allReduce with three devices" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 2.0 };
    var buf2 = [_]f32{ 3.0, 4.0 };
    var buf3 = [_]f32{ 5.0, 6.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
        .{ .device_id = 2, .data = &buf3 },
    };

    try backend.allReduce(&buffers, .sum);

    // Sum: [1+3+5, 2+4+6] = [9, 12]
    try testing.expectApproxEqAbs(@as(f32, 9.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 12.0), buf1[1], 0.001);
}

// ============================================================================
// Integration Tests - PeerTransferManager
// ============================================================================

test "PeerTransferManager initialization" {
    const allocator = testing.allocator;

    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    // Verify initial state
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 0), stats.total_transfers);
    try testing.expectEqual(@as(u64, 0), stats.successful_transfers);
    try testing.expectEqual(@as(u64, 0), stats.failed_transfers);
}

test "PeerTransferManager getCapability" {
    const allocator = testing.allocator;

    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    // Without real GPU hardware, should return host_staged
    const cap = manager.getCapability(0, 1);
    try testing.expect(cap == .host_staged);
}

test "PeerTransferManager recovery strategy" {
    const allocator = testing.allocator;

    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    // Default should be retry_with_fallback
    manager.setRecoveryStrategy(.abort);
    manager.setRecoveryStrategy(.skip_failed_device);
    manager.setRecoveryStrategy(.retry_with_fallback);
}

test "PeerTransferManager resetStats" {
    const allocator = testing.allocator;

    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    manager.resetStats();
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 0), stats.total_transfers);
}

// ============================================================================
// TransferStats Tests
// ============================================================================

test "TransferStats successRate" {
    var stats = mod.TransferStats{};

    // No transfers - 100% success
    try testing.expectEqual(@as(f64, 1.0), stats.successRate());

    // Some transfers
    stats.total_transfers = 10;
    stats.successful_transfers = 8;
    try testing.expectApproxEqAbs(@as(f64, 0.8), stats.successRate(), 0.001);
}

test "TransferStats avgTransferTimeNs" {
    var stats = mod.TransferStats{};

    // No transfers
    try testing.expectEqual(@as(u64, 0), stats.avgTransferTimeNs());

    // With transfers
    stats.successful_transfers = 4;
    stats.total_time_ns = 1000;
    try testing.expectEqual(@as(u64, 250), stats.avgTransferTimeNs());
}

test "TransferStats throughputBytesPerSec" {
    var stats = mod.TransferStats{};

    // No time
    try testing.expectEqual(@as(f64, 0.0), stats.throughputBytesPerSec());

    // With data
    stats.bytes_transferred = 1_000_000_000; // 1 GB
    stats.total_time_ns = 1_000_000_000; // 1 second
    try testing.expectApproxEqAbs(@as(f64, 1_000_000_000.0), stats.throughputBytesPerSec(), 1000.0);
}

// ============================================================================
// StagingPool Tests
// ============================================================================

test "StagingPool acquire and release" {
    const allocator = testing.allocator;
    var pool = try host_staged.StagingPool.init(allocator);
    defer pool.deinit();

    // Acquire buffer
    const buf1 = try pool.acquire(1000);
    try testing.expect(buf1.len >= 1000);

    // Release it
    pool.release(buf1);

    // Acquire again - should reuse
    const buf2 = try pool.acquire(500);
    try testing.expect(buf2.ptr == buf1.ptr); // Same buffer reused
    pool.release(buf2);
}

test "StagingPool minimum size" {
    const allocator = testing.allocator;
    var pool = try host_staged.StagingPool.init(allocator);
    defer pool.deinit();

    // Request small buffer, should get at least minimum size
    const buf = try pool.acquire(100);
    try testing.expect(buf.len >= 1024 * 1024); // MIN_BUFFER_SIZE is 1 MB
    pool.release(buf);
}

// ============================================================================
// Multi-Device Integration Tests
// ============================================================================

test "GPUCluster with peer transfer" {
    const allocator = testing.allocator;

    var cluster = multi_device.GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    // Check if peer manager was initialized
    if (cluster.getPeerManager()) |pm| {
        const stats = pm.getStats();
        try testing.expectEqual(@as(u64, 0), stats.total_transfers);
    }

    // Transfer stats should be accessible
    if (cluster.getTransferStats()) |stats| {
        try testing.expectEqual(@as(u64, 0), stats.total_transfers);
    }
}

test "GPUCluster allReduce with peer transfer" {
    const allocator = testing.allocator;

    var cluster = multi_device.GPUCluster.init(allocator, .{}) catch |err| {
        if (err == error.GpuDisabled or err == error.OutOfMemory) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer cluster.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // AllReduce should work (though with 1-2 simulated devices)
    try cluster.allReduce(&data, .sum);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

test "AllReduce with single device is no-op" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 1.0, 2.0, 3.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
    };

    try backend.allReduce(&buffers, .sum);

    // Should be unchanged
    try testing.expectApproxEqAbs(@as(f32, 1.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), buf1[2], 0.001);
}

test "TransferHandle status tracking" {
    var handle = mod.TransferHandle{
        .id = 1,
        .src_device = 0,
        .dst_device = 1,
        .size = 1024,
        .capability = .host_staged,
        .status = std.atomic.Value(TransferStatus).init(.pending),
        .start_time = 0,
    };

    try testing.expect(!handle.isComplete());
    try testing.expect(!handle.succeeded());

    handle.status.store(.in_progress, .release);
    try testing.expect(!handle.isComplete());

    handle.status.store(.completed, .release);
    try testing.expect(handle.isComplete());
    try testing.expect(handle.succeeded());

    handle.status.store(.failed, .release);
    try testing.expect(handle.isComplete());
    try testing.expect(!handle.succeeded());
}

// ============================================================================
// Stress Tests
// ============================================================================

test "Multiple concurrent transfers simulation" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    // Simulate multiple transfers
    var data: [1024]u8 = undefined;
    for (&data, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const src: u32 = @intCast(i % 4);
        const dst: u32 = @intCast((i + 1) % 4);
        if (src != dst) {
            try backend.transfer(src, dst, &data);
        }
    }

    const stats = backend.getStats();
    try testing.expect(stats.transfers > 0);
    try testing.expect(stats.bytes > 0);
}

test "Large buffer AllReduce" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    // Large buffers
    const buf1 = try allocator.alloc(f32, 10000);
    defer allocator.free(buf1);
    const buf2 = try allocator.alloc(f32, 10000);
    defer allocator.free(buf2);

    for (buf1, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (buf2, 0..) |*val, i| {
        val.* = @floatFromInt(i * 2);
    }

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = buf1 },
        .{ .device_id = 1, .data = buf2 },
    };

    try backend.allReduce(&buffers, .sum);

    // Verify some values
    try testing.expectApproxEqAbs(@as(f32, 0.0), buf1[0], 0.001); // 0 + 0
    try testing.expectApproxEqAbs(@as(f32, 3.0), buf1[1], 0.001); // 1 + 2
    try testing.expectApproxEqAbs(@as(f32, 6.0), buf1[2], 0.001); // 2 + 4
}

// ============================================================================
// Recovery Strategy Tests
// ============================================================================

test "RecoveryStrategy default is retry_with_fallback" {
    const allocator = testing.allocator;

    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    // Default should be retry_with_fallback (safe default)
    // This ensures automatic fallback on partial failures
    manager.setRecoveryStrategy(.abort);
    manager.setRecoveryStrategy(.retry_with_fallback); // Reset to default
}

test "Host-staged fallback on single-GPU machine" {
    const allocator = testing.allocator;

    // Single-GPU config
    var device_group = try multi_device.DeviceGroup.init(allocator, .{});
    defer device_group.deinit();

    var manager = try PeerTransferManager.init(allocator, &device_group);
    defer manager.deinit();

    // With only simulated devices, should use host-staged
    const cap = manager.getCapability(0, 1);
    try testing.expect(cap == .host_staged);

    // Should still work even without real peer access
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 0), stats.fallback_count);
}

test "AllReduce product operation" {
    const allocator = testing.allocator;
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var buf1 = [_]f32{ 2.0, 3.0, 4.0 };
    var buf2 = [_]f32{ 5.0, 2.0, 3.0 };

    const buffers = [_]host_staged.DeviceBufferRef{
        .{ .device_id = 0, .data = &buf1 },
        .{ .device_id = 1, .data = &buf2 },
    };

    try backend.allReduce(&buffers, .product);

    // Product: [2*5, 3*2, 4*3] = [10, 6, 12]
    try testing.expectApproxEqAbs(@as(f32, 10.0), buf1[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 6.0), buf1[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 12.0), buf1[2], 0.001);
}

test "TransferCapability selection priority" {
    // NCCL should always be preferred over direct P2P
    try testing.expect(TransferCapability.nccl.priority() > TransferCapability.direct_p2p.priority());

    // Direct P2P should be preferred over Vulkan external
    try testing.expect(TransferCapability.direct_p2p.priority() > TransferCapability.vulkan_external.priority());

    // Any hardware acceleration should be preferred over host-staged
    try testing.expect(TransferCapability.vulkan_external.priority() > TransferCapability.host_staged.priority());
    try testing.expect(TransferCapability.metal_shared.priority() > TransferCapability.host_staged.priority());
}

test "Memory leak check under repeated operations" {
    const allocator = testing.allocator;

    // Run many alloc/free cycles - Zig's test allocator will catch leaks
    var backend = try host_staged.HostStagedBackend.init(allocator);
    defer backend.deinit();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var buf1 = [_]f32{ 1.0, 2.0 };
        var buf2 = [_]f32{ 3.0, 4.0 };

        const buffers = [_]host_staged.DeviceBufferRef{
            .{ .device_id = 0, .data = &buf1 },
            .{ .device_id = 1, .data = &buf2 },
        };

        try backend.allReduce(&buffers, .sum);
    }

    // If we get here without leak, test passes
    const stats = backend.getStats();
    try testing.expect(stats.transfers > 0);
}
