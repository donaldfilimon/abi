//! GPU Dispatcher Tests — Extended Coverage
//!
//! Tests kernel dispatch types, launch configs, execution results,
//! and dispatcher lifecycle.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const gpu = if (build_options.enable_gpu) abi.gpu else struct {};
const dispatcher = if (build_options.enable_gpu) gpu.dispatcher else struct {};

// ============================================================================
// LaunchConfig Tests
// ============================================================================

test "gpu dispatcher: LaunchConfig for1D basic" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const LaunchConfig = dispatcher.LaunchConfig;

    const config = LaunchConfig.for1D(2048, 128);
    try std.testing.expectEqual(@as(u32, 2048), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 1), config.global_size[1]);
    try std.testing.expectEqual(@as(u32, 1), config.global_size[2]);

    const local = config.local_size.?;
    try std.testing.expectEqual(@as(u32, 128), local[0]);
}

test "gpu dispatcher: LaunchConfig for2D square" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const LaunchConfig = dispatcher.LaunchConfig;

    const config = LaunchConfig.for2D(1024, 1024, 32, 32);
    try std.testing.expectEqual(@as(u32, 1024), config.global_size[0]);
    try std.testing.expectEqual(@as(u32, 1024), config.global_size[1]);

    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 32), grid[0]);
    try std.testing.expectEqual(@as(u32, 32), grid[1]);
}

test "gpu dispatcher: LaunchConfig gridDimensions rounds up" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const LaunchConfig = dispatcher.LaunchConfig;

    // 100 items with local_size 32 → ceil(100/32) = 4 groups
    const config = LaunchConfig.for1D(100, 32);
    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 4), grid[0]); // ceil(100/32) = 4
}

test "gpu dispatcher: LaunchConfig gridDimensions without local_size" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const LaunchConfig = dispatcher.LaunchConfig;

    // No local_size → defaults to {256, 1, 1}
    // So: x = ceil(256/256)=1, y = ceil(256/1)=256, z = ceil(1/1)=1
    const config = LaunchConfig{
        .global_size = .{ 256, 256, 1 },
        .local_size = null,
    };
    const grid = config.gridDimensions();
    try std.testing.expectEqual(@as(u32, 1), grid[0]); // 256/256 = 1
    try std.testing.expectEqual(@as(u32, 256), grid[1]); // 256/1 = 256
    try std.testing.expectEqual(@as(u32, 1), grid[2]); // 1/1 = 1
}

// ============================================================================
// ExecutionResult Tests
// ============================================================================

test "gpu dispatcher: ExecutionResult throughputGBps" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const ExecutionResult = dispatcher.ExecutionResult;

    const result = ExecutionResult{
        .execution_time_ns = 500_000_000, // 0.5 second
        .elements_processed = 1_000_000,
        .bytes_transferred = 2 * 1024 * 1024 * 1024, // 2 GB
        .backend = .stdgpu,
        .device_id = 0,
        .gpu_executed = true,
    };

    // 2 GB in 0.5s = 4 GB/s
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result.throughputGBps(), 0.01);
}

test "gpu dispatcher: ExecutionResult elementsPerSecond" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const ExecutionResult = dispatcher.ExecutionResult;

    const result = ExecutionResult{
        .execution_time_ns = 2_000_000_000, // 2 seconds
        .elements_processed = 10_000_000,
        .bytes_transferred = 0,
        .backend = .stdgpu,
        .device_id = 0,
        .gpu_executed = false,
    };

    // 10M in 2s = 5M/s
    try std.testing.expectApproxEqAbs(@as(f64, 5_000_000.0), result.elementsPerSecond(), 1.0);
}

test "gpu dispatcher: ExecutionResult zero time" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const ExecutionResult = dispatcher.ExecutionResult;

    const result = ExecutionResult{
        .execution_time_ns = 0,
        .elements_processed = 100,
        .bytes_transferred = 100,
        .backend = .stdgpu,
        .device_id = 0,
        .gpu_executed = true,
    };

    // Zero time should not crash (return 0 or inf)
    _ = result.throughputGBps();
    _ = result.elementsPerSecond();
}

// ============================================================================
// KernelDispatcher Lifecycle Tests
// ============================================================================

test "gpu dispatcher: init and deinit with stdgpu" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const KernelDispatcher = dispatcher.KernelDispatcher;
    const Device = dispatcher.Device;

    const device = Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Test StdGPU Device",
        .device_type = .discrete,
        .vendor = .unknown,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
        .pci_bus_id = null,
        .driver_version = null,
    };

    var disp = KernelDispatcher.init(allocator, .stdgpu, &device) catch |err| {
        // If init fails due to backend unavailability, skip
        if (err == error.NoBackendAvailable) return error.SkipZigTest;
        return err;
    };
    defer disp.deinit();

    const stats = disp.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_compiled);
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_executed);
    try std.testing.expectEqual(@as(u64, 0), stats.cache_hits);
}

test "gpu dispatcher: init and deinit with simulated" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const KernelDispatcher = dispatcher.KernelDispatcher;
    const Device = dispatcher.Device;

    const device = Device{
        .id = 0,
        .backend = .simulated,
        .name = "Simulated Device",
        .device_type = .integrated,
        .vendor = .unknown,
        .total_memory = 1024 * 1024 * 256,
        .available_memory = 1024 * 1024 * 128,
        .is_emulated = true,
        .capability = .{},
        .compute_units = 4,
        .clock_mhz = 1000,
        .pci_bus_id = null,
        .driver_version = null,
    };

    var disp = KernelDispatcher.init(allocator, .simulated, &device) catch |err| {
        if (err == error.NoBackendAvailable) return error.SkipZigTest;
        return err;
    };
    defer disp.deinit();

    const stats = disp.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.kernels_compiled);
}

// ============================================================================
// DispatchError Tests
// ============================================================================

test "gpu dispatcher: DispatchError members accessible" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    // Verify the DispatchError type is accessible and has expected members
    const T = dispatcher.DispatchError;
    const info = @typeInfo(T);
    // DispatchError should be an error set
    try std.testing.expect(info == .error_set);
}

// ============================================================================
// Batched Dispatch Tests
// ============================================================================

test "gpu dispatcher: BatchedOp priority enum" {
    if (!build_options.enable_gpu) return error.SkipZigTest;
    const BatchedOp = dispatcher.BatchedOp;

    // Verify priority enum values exist
    try std.testing.expectEqual(BatchedOp.Priority.high, BatchedOp.Priority.high);
    try std.testing.expectEqual(BatchedOp.Priority.normal, BatchedOp.Priority.normal);
    try std.testing.expectEqual(BatchedOp.Priority.low, BatchedOp.Priority.low);
}
