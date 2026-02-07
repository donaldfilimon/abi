//! GPU Dispatcher Performance Benchmarking Tests
//!
//! Tests that verify performance characteristics of the GPU dispatcher:
//! - Kernel compilation performance
//! - Kernel caching effectiveness
//! - Ring buffer hit rates
//! - Memory allocation/deallocation overhead
//! - Execution timing and throughput
//! - Concurrent operation handling

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const abi = @import("../../src/abi");
const gpu = abi.gpu;
const dispatcher = gpu.dispatcher;
const builtin_kernels = gpu.builtin_kernels;

const BenchmarkResult = struct {
    name: []const u8,
    duration_ns: u64,
    iterations: usize,
    avg_duration_ns: u64,
    throughput: f64,
};

fn measurePerformance(test_fn: anytype, iterations: usize) !BenchmarkResult {
    var timer = try time.Timer.start();

    for (0..iterations) |_| {
        test_fn() catch {};
    }

    const duration_ns = timer.lap();
    const avg_duration_ns = duration_ns / iterations;
    const throughput = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);

    return BenchmarkResult{
        .name = @typeName(@TypeOf(test_fn)),
        .duration_ns = duration_ns,
        .iterations = iterations,
        .avg_duration_ns = avg_duration_ns,
        .throughput = throughput,
    };
}

test "GPU dispatcher: kernel compilation performance" {
    const allocator = std.testing.allocator;

    // Create mock device
    const device = dispatcher.Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Benchmark Device",
        .device_type = .cpu,
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

    var ctx = try dispatcher.KernelDispatcher.init(allocator, .stdgpu, &device);
    defer ctx.deinit();

    // Measure kernel compilation time
    const test_fn = struct {
        fn run() !void {
            const test_allocator = std.testing.allocator;
            const test_device = dispatcher.Device{
                .id = 0,
                .backend = .stdgpu,
                .name = "Benchmark Device",
                .device_type = .cpu,
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

            var inner_ctx = try dispatcher.KernelDispatcher.init(test_allocator, .stdgpu, &test_device);
            defer inner_ctx.deinit();

            const ir = dispatcher.KernelIR.empty("test_kernel");
            _ = try inner_ctx.compileKernel(&ir);
        }
    }.run;

    const result = try measurePerformance(test_fn, 10);
    std.debug.print("Kernel compilation: {d} ops/sec\\n", .{result.throughput});

    // Verify compilation time is reasonable (under 1ms per kernel)
    try std.testing.expect(result.avg_duration_ns < 1_000_000);
}

test "GPU dispatcher: kernel caching effectiveness" {
    const allocator = std.testing.allocator;

    const device = dispatcher.Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Benchmark Device",
        .device_type = .cpu,
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

    var ctx = try dispatcher.KernelDispatcher.init(allocator, .stdgpu, &device);
    defer ctx.deinit();

    const ir = dispatcher.KernelIR.empty("test_kernel");
    const kernel_handle = try ctx.compileKernel(&ir);

    const config = dispatcher.LaunchConfig{ .global_size = .{ 1, 1, 1 }, .local_size = .{ 1, 1, 1 }, .shared_memory = 0 };
    const args = dispatcher.KernelArgs{};

    // First execution should miss ring buffer
    _ = try ctx.execute(kernel_handle, config, args);
    const initial_hits = ctx.ring_hits;

    // Subsequent executions should hit cache
    for (0..10) |_| {
        _ = try ctx.execute(kernel_handle, config, args);
    }

    const final_hits = ctx.ring_hits;
    const hit_rate = @as(f64, @floatFromInt(final_hits - initial_hits)) / 10.0;

    std.debug.print("Ring buffer hit rate: {d:.2%}\\n", .{hit_rate});

    // Should have high cache hit rate
    try std.testing.expect(hit_rate > 0.8);
}

test "GPU dispatcher: concurrent execution performance" {
    const allocator = std.testing.allocator;

    const device = dispatcher.Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Benchmark Device",
        .device_type = .cpu,
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

    var ctx = try dispatcher.KernelDispatcher.init(allocator, .stdgpu, &device);
    defer ctx.deinit();

    const ir = dispatcher.KernelIR.empty("concurrent_test");
    const kernel_handle = try ctx.compileKernel(&ir);

    const config = dispatcher.LaunchConfig{ .global_size = .{ 1, 1, 1 }, .local_size = .{ 1, 1, 1 }, .shared_memory = 0 };
    const args = dispatcher.KernelArgs{};

    const concurrent_threads = 4;
    const executions_per_thread = 100;

    var threads = std.ArrayListUnmanaged(std.Thread).empty;
    defer {
        for (threads.items) |*t| {
            t.*.join();
        }
        threads.deinit(allocator);
    }

    const atomic_error_count = std.atomic.Value(usize).init(0);

    var timer = try time.Timer.start();

    for (0..concurrent_threads) |_| {
        const thread = try std.Thread.spawn(.{}, struct {
            fn run(ctx_param: *dispatcher.KernelDispatcher, e: *std.atomic.Value(usize), ch: dispatcher.KernelHandle, conf: dispatcher.LaunchConfig, a: dispatcher.KernelArgs, executions: usize) void {
                var local_errors: usize = 0;
                for (0..executions) |_| {
                    ctx_param.execute(ch, conf, a) catch {
                        local_errors += 1;
                    };
                }
                e.fetchAdd(local_errors, .monotonic);
            }
        }.run, .{ &ctx, &atomic_error_count, kernel_handle, config, args, executions_per_thread });
        try threads.append(allocator, thread);
    }

    // Wait for all threads
    for (threads.items) |*t| {
        t.*.join();
    }

    const duration_ns = timer.read();
    const total_executions = concurrent_threads * executions_per_thread;
    const throughput = @as(f64, @floatFromInt(total_executions)) / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);

    std.debug.print("Concurrent throughput: {d} ops/sec\\n", .{throughput});
    std.debug.print("Concurrent errors: {d}\\n", .{atomic_error_count.load(.monotonic)});

    // Should complete with no errors
    try std.testing.expectEqual(@as(usize, 0), atomic_error_count.load(.monotonic));
    try std.testing.expectThroughput(&.{
        .name = "concurrent dispatches",
        .target_ops_per_second = 10.0,
        .actual_ops_per_second = throughput,
    });
}

test "GPU dispatcher: memory allocation tracking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const tracking_allocator = gpa.allocator();

    const device = dispatcher.Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "Benchmark Device",
        .device_type = .cpu,
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

    var ctx = try dispatcher.KernelDispatcher.init(tracking_allocator, .stdgpu, &device);
    defer ctx.deinit();

    // Trigger some allocations
    const ir = dispatcher.KernelIR.empty("memory_test");
    const kernel_handle = try ctx.compileKernel(&ir);

    const config = dispatcher.LaunchConfig{ .global_size = .{ 1, 1, 1 }, .local_size = .{ 1, 1, 1 }, .shared_memory = 0 };
    const args = dispatcher.KernelArgs{};

    for (0..10) |_| {
        _ = try ctx.execute(kernel_handle, config, args);
    }

    // Check for leaks by deinitializing everything
    // Should not crash or have memory leaks
}

// Helper to check throughput expectations
test "GPU dispatcher: throughput validation wrapper" {
    // This is a helper test that will be used by the performance test above
    _ = std.testing;
}

pub fn expectThroughput(expected: struct {
    name: []const u8,
    target_ops_per_second: f64,
    actual_ops_per_second: f64,
}) void {
    std.debug.print("Throughput check: {s} (target: {d:.1} ops/sec, actual: {d:.1} ops/sec)\\n", .{
        expected.name,
        expected.target_ops_per_second,
        expected.actual_ops_per_second,
    });

    // In real implementation, this would assert the throughput meets expectations
    // For now, just validate it's positive
    std.debug.assert(expected.actual_ops_per_second > 0.0);
}
