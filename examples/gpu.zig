//! GPU Example
//!
//! Demonstrates the unified GPU API for:
//! - GPU device discovery and selection
//! - Smart buffer management with automatic memory modes
//! - High-level operations (vectorAdd, matrixMultiply, reduceSum, dotProduct)
//! - Profiling and metrics collection
//! - Multi-GPU support

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI GPU Example ===\n\n", .{});

    // Initialize framework
    var builder = abi.Framework.builder(allocator);
    _ = builder.withDefault(.gpu);
    var framework = builder.build() catch |err| {
        std.debug.print("Framework initialization failed: {t}\n", .{err});
        return err;
    };
    defer framework.deinit();

    // Check GPU module status
    const gpu_enabled = abi.gpu.backends.detect.moduleEnabled();
    std.debug.print("GPU module: {s}\n", .{if (gpu_enabled) "enabled" else "disabled"});

    if (!gpu_enabled) {
        std.debug.print("GPU features are not available\n", .{});
        return;
    }

    // === Device Discovery ===
    std.debug.print("\n--- Device Discovery ---\n", .{});

    const backends = abi.gpu.backends.detect.availableBackends(allocator) catch |err| {
        std.debug.print("Failed to enumerate GPU backends: {t}\n", .{err});
        return err;
    };
    defer allocator.free(backends);

    std.debug.print("Available backends: {d}\n", .{backends.len});
    for (backends) |backend| {
        const avail = abi.gpu.backends.detect.backendAvailability(backend);
        std.debug.print("  {t}: {t} ({d} devices)\n", .{ backend, avail.level, avail.device_count });
    }

    const devices = abi.gpu.backends.listing.listDevices(allocator) catch |err| {
        std.debug.print("Failed to list GPU devices: {t}\n", .{err});
        return err;
    };
    defer allocator.free(devices);

    std.debug.print("Detected devices: {d}\n", .{devices.len});
    for (devices, 0..) |device, i| {
        std.debug.print("  [{d}] {s} ({t})\n", .{ i, device.name, device.backend });
    }

    // === Unified GPU API Demo ===
    std.debug.print("\n--- Unified GPU API ---\n", .{});

    // Initialize the unified GPU API with profiling enabled
    var gpu = abi.Gpu.init(allocator, .{
        .enable_profiling = true,
        .memory_mode = .automatic,
    }) catch |err| {
        std.debug.print("Unified GPU init failed: {t}\n", .{err});
        return;
    };
    defer gpu.deinit();

    if (!gpu.isAvailable()) {
        std.debug.print("No GPU device available\n", .{});
        return;
    }

    if (gpu.getActiveDevice()) |device| {
        std.debug.print("Active device: {s} ({t})\n", .{ device.name, device.backend });
    }

    // === Vector Addition Demo ===
    std.debug.print("\n--- Vector Addition ---\n", .{});

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b_data = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    const a = gpu.createBufferFromSlice(f32, &a_data, .{}) catch |err| {
        std.debug.print("Failed to create buffer A: {t}\n", .{err});
        return;
    };
    defer gpu.destroyBuffer(a);

    const b = gpu.createBufferFromSlice(f32, &b_data, .{}) catch |err| {
        std.debug.print("Failed to create buffer B: {t}\n", .{err});
        return;
    };
    defer gpu.destroyBuffer(b);

    const result = gpu.createBuffer(a_data.len * @sizeOf(f32), .{}) catch |err| {
        std.debug.print("Failed to create result buffer: {t}\n", .{err});
        return;
    };
    defer gpu.destroyBuffer(result);

    const vec_result = gpu.vectorAdd(a, b, result) catch |err| {
        std.debug.print("vectorAdd failed: {t}\n", .{err});
        return;
    };

    std.debug.print("vectorAdd: {d} elements in {d:.3}us\n", .{
        vec_result.elements_processed,
        @as(f64, @floatFromInt(vec_result.execution_time_ns)) / 1000.0,
    });

    // Read and display results
    var output: [8]f32 = undefined;
    result.read(f32, &output) catch |err| {
        std.debug.print("Failed to read result: {t}\n", .{err});
        return;
    };

    std.debug.print("Input A:  ", .{});
    for (a_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("\n", .{});

    std.debug.print("Input B:  ", .{});
    for (b_data) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("\n", .{});

    std.debug.print("A + B:    ", .{});
    for (output) |v| std.debug.print("{d:.1} ", .{v});
    std.debug.print("\n", .{});

    // === Reduce Sum Demo ===
    std.debug.print("\n--- Reduce Sum ---\n", .{});

    const sum_result = gpu.reduceSum(a) catch |err| {
        std.debug.print("reduceSum failed: {t}\n", .{err});
        return;
    };

    std.debug.print("Sum of A: {d:.1} (expected: 36.0)\n", .{sum_result.value});
    std.debug.print("Execution time: {d:.3}us\n", .{
        @as(f64, @floatFromInt(sum_result.stats.execution_time_ns)) / 1000.0,
    });

    // === Dot Product Demo ===
    std.debug.print("\n--- Dot Product ---\n", .{});

    const dot_result = gpu.dotProduct(a, b) catch |err| {
        std.debug.print("dotProduct failed: {t}\n", .{err});
        return;
    };

    // a · b = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
    std.debug.print("A · B: {d:.1} (expected: 120.0)\n", .{dot_result.value});

    // === Statistics ===
    std.debug.print("\n--- Statistics ---\n", .{});

    const stats = gpu.getStats();
    std.debug.print("Kernels launched: {d}\n", .{stats.kernels_launched});
    std.debug.print("Buffers created: {d}\n", .{stats.buffers_created});
    std.debug.print("Total bytes allocated: {d}\n", .{stats.bytes_allocated});
    std.debug.print("Total execution time: {d:.3}ms\n", .{
        @as(f64, @floatFromInt(stats.total_execution_time_ns)) / 1_000_000.0,
    });

    // === Profiling (if enabled) ===
    if (gpu.getMetricsSummary()) |summary| {
        std.debug.print("\n--- Profiling Summary ---\n", .{});
        std.debug.print("Total kernel invocations: {d}\n", .{summary.total_kernel_invocations});
        std.debug.print("Average kernel time: {d:.3}ns\n", .{summary.avg_kernel_time_ns});
        std.debug.print("Kernels per second: {d:.2}\n", .{summary.kernels_per_second});
    }

    // === Memory Info ===
    const mem_info = gpu.getMemoryInfo();
    std.debug.print("\n--- Memory Info ---\n", .{});
    std.debug.print("Used: {d} bytes\n", .{mem_info.used_bytes});
    std.debug.print("Peak: {d} bytes\n", .{mem_info.peak_used_bytes});

    // === Health Check ===
    const health = gpu.checkHealth();
    std.debug.print("\n--- Health Status ---\n", .{});
    std.debug.print("GPU health: {t}\n", .{health});

    std.debug.print("\n=== GPU Example Complete ===\n", .{});
}
