const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_gpu = true,
    }) catch |err| {
        std.debug.print("GPU initialization failed: {}\n", .{err});
        std.debug.print("Falling back to CPU-only operations...\n", .{});

        // Initialize with GPU disabled for CPU fallback
        var cpu_framework = abi.init(allocator, abi.FrameworkOptions{
            .enable_gpu = false,
        }) catch |cpu_err| {
            std.debug.print("Even CPU initialization failed: {}\n", .{cpu_err});
            return cpu_err;
        };
        defer abi.shutdown(&cpu_framework);

        // Test CPU SIMD operations
        const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
        const result = abi.vectorDot(&vec_a, &vec_b);
        std.debug.print("CPU SIMD dot product: {d:.3}\n", .{result});
        std.debug.print("GPU status: unavailable, using CPU SIMD fallback\n", .{});
        return;
    };
    defer abi.shutdown(&framework);

    // Check GPU module status
    const gpu_enabled = abi.gpu.moduleEnabled();
    std.debug.print("GPU module: {s}\n", .{if (gpu_enabled) "enabled" else "disabled"});

    if (!gpu_enabled) {
        std.debug.print("GPU features are not available\n", .{});
        return;
    }

    // Try to get GPU backend information
    const backends = abi.gpu.availableBackends(allocator) catch |err| {
        std.debug.print("Failed to enumerate GPU backends: {}\n", .{err});
        return err;
    };
    defer allocator.free(backends);

    std.debug.print("Available GPU backends: {d}\n", .{backends.len});
    for (backends) |backend| {
        const avail = abi.gpu.backendAvailability(backend);
        std.debug.print("  {t}: {t} ({d} devices)\n", .{ backend, avail.level, avail.device_count });
    }

    // Try to list GPU devices
    const devices = abi.gpu.listDevices(allocator) catch |err| {
        std.debug.print("Failed to list GPU devices: {}\n", .{err});
        return err;
    };
    defer allocator.free(devices);

    std.debug.print("Detected GPU devices: {d}\n", .{devices.len});
    for (devices, 0..) |device, i| {
        std.debug.print("  Device {d}: {s} ({t})\n", .{ i + 1, device.name, device.backend });
    }

    // Test SIMD support
    std.debug.print("SIMD support: {s}\n", .{if (abi.hasSimdSupport()) "yes" else "no"});
}
