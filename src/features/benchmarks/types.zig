//! Shared types for the benchmarks feature (mod + stub).

const std = @import("std");
const core_config = @import("../../core/config/benchmarks.zig");

pub const Config = core_config.BenchmarksConfig;

pub const BenchmarksError = error{
    BenchmarksDisabled,
    FeatureDisabled,
    OutOfMemory,
    InvalidConfig,
    BenchmarkFailed,
};

/// Function signature for benchmark bodies.
pub const BenchmarkFn = *const fn (state: *BenchmarkState) void;

/// State passed into every benchmark function.
pub const BenchmarkState = struct {
    iteration: usize = 0,
    total_iterations: usize = 0,
    /// Scratch allocator for benchmark use.
    allocator: std.mem.Allocator,

    /// Prevents the compiler from optimizing away a computed value.
    pub fn doNotOptimize(_: *BenchmarkState, value: anytype) void {
        var v = value;
        _ = &v;
    }
};

/// Timing results for a single benchmark.
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    median_ns: u64,

    /// Compute throughput in operations per second from the mean.
    pub fn opsPerSecond(self: BenchmarkResult) f64 {
        if (self.mean_ns == 0) return 0.0;
        return 1_000_000_000.0 / @as(f64, @floatFromInt(self.mean_ns));
    }
};
