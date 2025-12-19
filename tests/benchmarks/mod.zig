//! Performance benchmarks for the ABI framework

const std = @import("std");
const ai_benchmarks = @import("ai_benchmarks.zig");

/// Benchmark framework initialization
pub fn benchmarkFrameworkInit(_: std.mem.Allocator) !void {
    const iterations = 100;
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        var framework = try std.heap.page_allocator.create(std.heap.GeneralPurposeAllocator(.{}));
        defer _ = framework.deinit();
    }

    const time_ns = timer.read();
    const avg_time = @as(f64, @floatFromInt(time_ns)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Framework initialization: {d:.2}μs per iteration\n", .{avg_time / 1000.0});
}

/// Benchmark memory operations
pub fn benchmarkMemoryOperations(allocator: std.mem.Allocator) !void {
    const iterations = 10000;
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        const mem = try allocator.alloc(u8, 1024);
        defer allocator.free(mem);
    }

    const time_ns = timer.read();
    const avg_time = @as(f64, @floatFromInt(time_ns)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Memory allocation: {d:.2}μs per 1KB allocation\n", .{avg_time / 1000.0});
}

/// Benchmark SIMD operations
pub fn benchmarkSIMDOperations(allocator: std.mem.Allocator) !void {
    _ = allocator; // Not used in this benchmark
    const iterations = 100000;
    var timer = try std.time.Timer.start();

    // Simple SIMD-like operations using vectors
    const a: @Vector(4, f32) = @splat(1.0);
    const b: @Vector(4, f32) = @splat(2.0);

    for (0..iterations) |_| {
        const result = a + b * a;
        std.mem.doNotOptimizeAway(result);
    }

    const time_ns = timer.read();
    const avg_time = @as(f64, @floatFromInt(time_ns)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("SIMD operations: {d:.2}ns per vector operation\n", .{avg_time});
}

/// Run all benchmarks
pub fn runBenchmarks(allocator: std.mem.Allocator) !void {
    std.debug.print("=== ABI Framework Benchmarks ===\n", .{});

    try benchmarkFrameworkInit(allocator);
    try benchmarkMemoryOperations(allocator);
    try benchmarkSIMDOperations(allocator);

    // Run AI module benchmarks
    try ai_benchmarks.runAIBenchmarks(allocator);

    std.debug.print("=== Benchmarks Complete ===\n", .{});
}

test "run benchmarks" {
    try runBenchmarks(std.testing.allocator);
}
