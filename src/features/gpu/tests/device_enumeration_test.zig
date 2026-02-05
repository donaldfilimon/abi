const std = @import("std");
const device = @import("../device.zig");
const backend_factory = @import("../backend_factory.zig");

test "enumerate all available GPU devices" {
    const allocator = std.testing.allocator;

    const devices = try device.enumerateAllDevices(allocator);
    defer allocator.free(devices);

    // Should find at least CPU fallback
    try std.testing.expect(devices.len >= 1);

    // Verify each device has valid properties
    for (devices) |dev| {
        try std.testing.expect(dev.name.len > 0);
        try std.testing.expect(dev.id >= 0);
    }
}

test "enumerate devices per backend" {
    const allocator = std.testing.allocator;

    const cuda_devices = try device.enumerateDevicesForBackend(allocator, .cuda);
    defer allocator.free(cuda_devices);

    // May be 0 on non-NVIDIA systems
    for (cuda_devices) |dev| {
        try std.testing.expectEqual(.cuda, dev.backend);
    }
}

test "select best device with custom selector" {
    const allocator = std.testing.allocator;

    const criteria = device.DeviceSelectionCriteria{
        .prefer_discrete = true,
        .min_memory_gb = 4,
        .required_features = &.{.fp16},
    };

    const best_device = try device.selectBestDevice(allocator, criteria);

    if (best_device) |d| {
        if (d.total_memory) |mem| {
            try std.testing.expect(mem >= 4 * 1024 * 1024 * 1024);
        }
    }
}
