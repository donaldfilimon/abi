const std = @import("std");
const device_mod = @import("../device.zig");
const backend_factory = @import("../backend_factory.zig");
const backend_mod = @import("../backend.zig");
const Backend = backend_mod.Backend;

test "all backends device enumeration" {
    const allocator = std.testing.allocator;

    // Test enumeration for each backend type
    inline for (std.meta.tags(Backend)) |backend_tag| {
        const devices = try device_mod.enumerateDevicesForBackend(allocator, backend_tag);
        defer allocator.free(devices);

        // All returned devices should match the backend
        for (devices) |dev| {
            try std.testing.expectEqual(backend_tag, dev.backend);
            try std.testing.expect(dev.name.len > 0);
        }
    }
}

test "backend factory creates valid instances" {
    const allocator = std.testing.allocator;

    // Try creating the best available backend
    const instance = backend_factory.createBestBackend(allocator) catch |err| {
        // It's okay if no backends are available
        if (err == backend_factory.FactoryError.NoBackendsAvailable) {
            return;
        }
        return err;
    };
    defer backend_factory.destroyBackend(instance);

    // Instance should have a valid VTable backend
    try std.testing.expect(instance.backend.getDeviceCount() >= 1);
}

test "feature-based backend selection" {
    const allocator = std.testing.allocator;

    // Test selection with various feature requirements
    const fp16_backend = try backend_factory.selectBackendWithFeatures(allocator, .{
        .required_features = &.{.fp16},
        .fallback_to_cpu = true,
    });

    // Should always return something when fallback_to_cpu is true
    try std.testing.expect(fp16_backend != null);

    // Test without CPU fallback
    const strict_backend = try backend_factory.selectBackendWithFeatures(allocator, .{
        .required_features = &.{ .tensor_cores, .cooperative_groups },
        .fallback_to_cpu = false,
    });

    // May be null if no GPU supports these features
    if (strict_backend) |backend| {
        // If returned, should be CUDA (only backend with these features)
        try std.testing.expectEqual(.cuda, backend);
    }
}

test "device scoring and selection" {
    const allocator = std.testing.allocator;

    const all_devices = try device_mod.enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    if (all_devices.len == 0) return;

    // Find highest scoring device
    var best_score: u32 = 0;
    var best_device: ?device_mod.Device = null;

    for (all_devices) |dev| {
        const score = dev.score();
        if (score > best_score) {
            best_score = score;
            best_device = dev;
        }
    }

    try std.testing.expect(best_device != null);
    try std.testing.expect(best_score > 0);
}

test "device capability queries" {
    const allocator = std.testing.allocator;

    const all_devices = try device_mod.enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    for (all_devices) |dev| {
        // All devices should have valid workgroup limits
        const workgroup_size = dev.maxWorkgroupSize();
        try std.testing.expect(workgroup_size > 0);
        try std.testing.expect(workgroup_size <= 1024); // Common max

        // Shared memory should be reasonable
        const shared_mem = dev.maxSharedMemory();
        try std.testing.expect(shared_mem >= 1024); // At least 1KB
    }
}
