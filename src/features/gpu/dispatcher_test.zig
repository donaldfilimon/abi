//! Tests for GPU Kernel Dispatcher

const std = @import("std");
const dispatcher_mod = @import("dispatch/coordinator.zig");

const LaunchConfig = dispatcher_mod.LaunchConfig;
const ExecutionResult = dispatcher_mod.ExecutionResult;
const KernelDispatcher = dispatcher_mod.KernelDispatcher;
const Device = dispatcher_mod.Device;

test "LaunchConfig for1D" {
    const config = LaunchConfig.for1D(1024, 256);
    try std.testing.expectEqual(@as(u32, 1024), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 256), config.local_size.?[0]);

    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 4), grid[0]);
}

test "LaunchConfig for2D" {
    const config = LaunchConfig.for2D(512, 512, 16, 16);
    try std.testing.expectEqual(@as(u32, 512), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 512), config.global_size[1]);

    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 32), grid[0]);
    try std.testing.expectEqual(@as(u32, 32), grid[1]);
}

test "ExecutionResult throughput" {
    const result = ExecutionResult{
        .execution_time_ns = 1_000_000_000, // 1 second
        .elements_processed = 1_000_000,
        .bytes_transferred = 1024 * 1024 * 1024, // 1 GB
        .backend = .cuda,
        .device_id = 0,
        .gpu_executed = true,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.throughputGBps(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1_000_000.0), result.elementsPerSecond(), 1.0);
}

test "KernelDispatcher init and deinit" {
    const device = Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var dispatcher = try KernelDispatcher.init(std.testing.allocator, .stdgpu, &device);
    defer dispatcher.deinit();

    const stats = dispatcher.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_compiled);
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_executed);
}
