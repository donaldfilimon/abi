const std = @import("std");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.log.info("=== ABI Framework Performance Validation ===", .{});

    // Test 1: Compute Engine Performance
    std.log.info("\n[1/4] Testing Compute Engine...", .{});
    try testComputeEngine(allocator);

    // Test 2: Data Structure Performance
    std.log.info("\n[2/4] Testing Data Structures...", .{});
    try testDataStructures(allocator);

    // Test 3: Memory Pool Performance
    std.log.info("\n[3/4] Testing Memory Pools...", .{});
    try testMemoryPools(allocator);

    // Test 4: System Memory Detection
    std.log.info("\n[4/4] Testing System Memory Detection...", .{});
    try testSystemMemory();

    std.log.info("\n=== All Performance Tests Passed ===", .{});
}

fn testComputeEngine(allocator: std.mem.Allocator) !void {
    const compute_engine = @import("src/features/compute/compute_engine.zig");
    const Task = compute_engine.Task;

    const engine = try compute_engine.ComputeEngine.init(allocator, 4);
    defer engine.deinit();

    // Submit a batch of tasks
    var tasks: [100]Task = undefined;
    var counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        tasks[i] = Task{
            .func = struct {
                fn increment(ctx: ?*anyopaque) void {
                    const c: *std.atomic.Value(usize) = @ptrCast(@alignCast(ctx));
                    _ = c.fetchAdd(1, .monotonic);
                }
            }.increment,
            .ctx = &counter,
        };
    }

    var timer = try std.time.Timer.start();
    try engine.submitBatch(&tasks);
    engine.waitIdle();
    const elapsed_ns = timer.read();

    std.log.info("  - Processed 100 tasks in {d:.3} ms", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    std.log.info("  - Counter value: {d}", .{counter.load(.acquire)});

    if (counter.load(.acquire) != 100) return error.CounterMismatch;
}

fn testDataStructures(allocator: std.mem.Allocator) !void {
    const data_structures = @import("src/features/ai/data_structures/mod.zig");
    const CircularBuffer = data_structures.CircularBuffer;
    const LockFreeStats = data_structures.concurrent.LockFreeStats;

    // Test CircularBuffer
    const buffer = try CircularBuffer(u32).init(allocator, 1024);
    defer buffer.deinit();

    var timer = try std.time.Timer.start();
    var i: u32 = 0;
    while (i < 10000) : (i += 1) {
        buffer.push(i);
    }
    const elapsed_ns = timer.read();

    std.log.info("  - CircularBuffer: pushed 10,000 items in {d:.3} ms", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    std.log.info("  - Buffer length: {d}", .{buffer.len});

    // Test LockFreeStats
    var stats = LockFreeStats{};
    timer.reset();
    i = 0;
    while (i < 10000) : (i += 1) {
        stats.recordOperation(true, 100 + @as(u64, @intCast(i)));
    }
    const stats_elapsed = timer.read();

    std.log.info("  - LockFreeStats: recorded 10,000 ops in {d:.3} ms", .{@as(f64, @floatFromInt(stats_elapsed)) / 1_000_000.0});
    std.log.info("  - Success rate: {d:.2}%", .{stats.successRate() * 100.0});
}

fn testMemoryPools(allocator: std.mem.Allocator) !void {
    const data_structures = @import("src/features/ai/data_structures/mod.zig");

    const pool = try data_structures.createMemoryPool(u64, allocator, 1024);
    defer pool.deinit();

    var timer = try std.time.Timer.start();
    var allocations: usize = 0;
    while (allocations < 1000) : (allocations += 1) {
        const item = pool.get();
        if (item) |_| {
            // Successfully allocated from pool
        } else {
            std.log.warn("  - Pool allocation failed at {d}", .{allocations});
            break;
        }
    }
    const elapsed_ns = timer.read();

    std.log.info("  - MemoryPool: allocated {d} items in {d:.3} ms", .{ allocations, @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 });
}

fn testSystemMemory() !void {
    const hardware_detection = @import("src/features/gpu/hardware_detection.zig");

    const system_memory = hardware_detection.getSystemMemoryLimit();
    const system_memory_gb = @as(f64, @floatFromInt(system_memory)) / (1024.0 * 1024.0 * 1024.0);

    std.log.info("  - System memory detected: {d:.2} GB ({d} bytes)", .{ system_memory_gb, system_memory });

    if (system_memory == 0) return error.MemoryDetectionFailed;
    if (system_memory < 512 * 1024 * 1024) return error.UnrealisticMemorySize;
}
