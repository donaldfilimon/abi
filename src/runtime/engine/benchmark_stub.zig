//! Benchmark Stub for WASM/Freestanding Targets
//!
//! Provides stub implementations when benchmarking is not available.

const std = @import("std");

/// Stub benchmark result
pub const BenchmarkResult = struct {
    name: []const u8 = "stub",
    iterations: usize = 0,
    total_ns: u64 = 0,
    avg_ns: u64 = 0,
    min_ns: u64 = 0,
    max_ns: u64 = 0,
};

/// Stub run benchmarks - returns empty results
pub fn runBenchmarks(allocator: std.mem.Allocator) ![]BenchmarkResult {
    _ = allocator;
    return &[_]BenchmarkResult{};
}

test {
    std.testing.refAllDecls(@This());
}
