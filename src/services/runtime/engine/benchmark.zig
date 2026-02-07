//! Benchmark utilities for compute runtime.
//!
//! Note: Full benchmark implementation is in benchmarks/legacy.zig.
//! This module provides minimal exports for API compatibility.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// Result of a benchmark run.
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    duration_ns: u64,
    ops_per_sec: f64,
};

/// Run a suite of compute benchmarks.
/// For comprehensive benchmarks, use `zig build benchmarks` instead.
pub fn runBenchmarks(allocator: std.mem.Allocator) ![]BenchmarkResult {
    var list = std.ArrayListUnmanaged(BenchmarkResult).empty;
    errdefer list.deinit(allocator);

    // Add basic allocation benchmark
    try list.append(allocator, try runAllocationBenchmark(allocator));

    return list.toOwnedSlice(allocator);
}

fn runAllocationBenchmark(allocator: std.mem.Allocator) !BenchmarkResult {
    const iterations: u64 = 10_000;

    var timer = try time.Timer.start();

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const data = try allocator.alloc(u8, 1024);
        std.mem.doNotOptimizeAway(data.ptr);
        allocator.free(data);
    }

    const elapsed = timer.read();
    const seconds = @as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0;
    const ops = if (seconds == 0) 0 else @as(f64, @floatFromInt(iterations)) / seconds;

    return .{
        .name = "alloc_1kb",
        .iterations = iterations,
        .duration_ns = elapsed,
        .ops_per_sec = ops,
    };
}

test "benchmark result" {
    const result = BenchmarkResult{
        .name = "test",
        .iterations = 100,
        .duration_ns = 1000,
        .ops_per_sec = 100_000_000.0,
    };
    try std.testing.expectEqualStrings("test", result.name);
}
