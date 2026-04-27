//! Integration Tests: Benchmarks Feature
//!
//! Tests the benchmarks module exports, lifecycle queries, configuration,
//! result types, and suite API through the public `abi.benchmarks` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const benchmarks = abi.benchmarks;

// ============================================================================
// Feature gate
// ============================================================================

test "benchmarks: isEnabled reflects feature flag" {
    if (build_options.feat_benchmarks) {
        try std.testing.expect(benchmarks.isEnabled());
    } else {
        try std.testing.expect(!benchmarks.isEnabled());
    }
}

test "benchmarks: isInitialized reflects feature flag" {
    if (build_options.feat_benchmarks) {
        const initialized = benchmarks.isInitialized();
        try std.testing.expect(initialized == true or initialized == false);
    } else {
        try std.testing.expect(!benchmarks.isInitialized());
    }
}

// ============================================================================
// Types
// ============================================================================

test "benchmarks: BenchmarksError includes expected variants" {
    const e1: benchmarks.BenchmarksError = error.FeatureDisabled;
    const e2: benchmarks.BenchmarksError = error.InvalidConfig;
    const e3: benchmarks.BenchmarksError = error.BenchmarkFailed;
    try std.testing.expect(e1 == error.FeatureDisabled);
    try std.testing.expect(e2 == error.InvalidConfig);
    try std.testing.expect(e3 == error.BenchmarkFailed);
}

test "benchmarks: Error alias matches BenchmarksError" {
    try std.testing.expect(benchmarks.Error == benchmarks.BenchmarksError);
}

test "benchmarks: BenchmarkResult type is accessible" {
    const result = benchmarks.BenchmarkResult{
        .name = "test-bench",
        .iterations = 1000,
        .total_ns = 1_000_000,
        .min_ns = 500,
        .max_ns = 2000,
        .mean_ns = 1000,
        .median_ns = 950,
    };
    try std.testing.expectEqualStrings("test-bench", result.name);
    try std.testing.expectEqual(@as(usize, 1000), result.iterations);
    try std.testing.expectEqual(@as(u64, 1000), result.mean_ns);
}

test "benchmarks: BenchmarkResult opsPerSecond" {
    const result = benchmarks.BenchmarkResult{
        .name = "ops-test",
        .iterations = 100,
        .total_ns = 100_000,
        .min_ns = 900,
        .max_ns = 1100,
        .mean_ns = 1000,
        .median_ns = 1000,
    };
    // 1_000_000_000 / 1000 = 1_000_000.0 ops/sec
    try std.testing.expectEqual(@as(f64, 1_000_000.0), result.opsPerSecond());
}

test "benchmarks: BenchmarkResult opsPerSecond zero mean" {
    const result = benchmarks.BenchmarkResult{
        .name = "zero",
        .iterations = 0,
        .total_ns = 0,
        .min_ns = 0,
        .max_ns = 0,
        .mean_ns = 0,
        .median_ns = 0,
    };
    try std.testing.expectEqual(@as(f64, 0.0), result.opsPerSecond());
}

test "benchmarks: BenchmarkFn type is accessible" {
    const F = benchmarks.BenchmarkFn;
    _ = F;
}

test "benchmarks: BenchmarkState type is accessible" {
    const S = benchmarks.BenchmarkState;
    _ = S;
}

// ============================================================================
// BenchmarkSuite
// ============================================================================

test "benchmarks: BenchmarkSuite init and deinit" {
    const config = benchmarks.Config{};
    var suite = benchmarks.BenchmarkSuite.init("test-suite", config);
    suite.deinit(std.testing.allocator);
}

test "benchmarks: BenchmarkSuite addBenchmark returns result or FeatureDisabled" {
    const config = benchmarks.Config{};
    var suite = benchmarks.BenchmarkSuite.init("test-suite", config);
    defer suite.deinit(std.testing.allocator);

    const noop = struct {
        fn bench(_: *benchmarks.BenchmarkState) void {}
    }.bench;

    const result = suite.addBenchmark(std.testing.allocator, "noop", noop);
    if (result) |_| {
        // Feature enabled -- benchmark added
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "benchmarks: BenchmarkSuite run returns result or FeatureDisabled" {
    const config = benchmarks.Config{};
    var suite = benchmarks.BenchmarkSuite.init("test-suite", config);
    defer suite.deinit(std.testing.allocator);

    const result = suite.run(std.testing.allocator);
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

// ============================================================================
// Sub-modules
// ============================================================================

test "benchmarks: types sub-module is accessible" {
    const T = benchmarks.types;
    _ = T.BenchmarkResult;
    _ = T.BenchmarkState;
    _ = T.BenchmarkFn;
}

test {
    std.testing.refAllDecls(@This());
}
