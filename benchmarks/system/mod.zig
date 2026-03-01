//! Benchmark System Module
//!
//! System-level benchmarking infrastructure including:
//! - **framework**: Core benchmark runner with statistical analysis
//! - **baseline_store**: JSON-based baseline persistence for regression detection
//! - **baseline_comparator**: Comparison and reporting tools

const std = @import("std");

// Core benchmark framework
pub const framework = @import("framework.zig");

// Baseline persistence
pub const baseline_store = @import("baseline_store.zig");

// Baseline comparison
pub const baseline_comparator = @import("baseline_comparator.zig");

// Re-export commonly used types for convenience

/// Baseline storage for benchmark results
pub const BaselineStore = baseline_store.BaselineStore;

/// A benchmark result that can be persisted
pub const BenchmarkResult = baseline_store.BenchmarkResult;

/// Result of comparing current vs baseline
pub const ComparisonResult = baseline_comparator.ComparisonResult;

/// Comprehensive regression report
pub const RegressionReport = baseline_comparator.RegressionReport;

/// Configuration for comparison thresholds
pub const ComparisonConfig = baseline_comparator.ComparisonConfig;

/// Framework benchmark result with statistics
pub const BenchResult = framework.BenchResult;

/// Framework benchmark configuration
pub const BenchConfig = framework.BenchConfig;

/// Framework statistical summary
pub const Statistics = framework.Statistics;

/// Framework benchmark runner
pub const BenchmarkRunner = framework.BenchmarkRunner;

// Re-export comparison functions

/// Compare all benchmark results against baselines
pub const compareAll = baseline_comparator.compareAll;

/// Compare all results with custom configuration
pub const compareAllWithConfig = baseline_comparator.compareAllWithConfig;

/// Compare a single benchmark result
pub const compareSingle = baseline_comparator.compareSingle;

/// Convert framework BenchResult to persistable BenchmarkResult
pub fn benchResultToBaseline(
    result: BenchResult,
    git_commit: ?[]const u8,
    git_branch: ?[]const u8,
) BenchmarkResult {
    return .{
        .name = result.config.name,
        .metric = "ops_per_sec",
        .value = result.stats.opsPerSecond(),
        .unit = "ops/s",
        .timestamp = result.timestamp,
        .git_commit = git_commit,
        .git_branch = git_branch,
        .category = result.config.category,
        .std_dev = result.stats.std_dev_ns,
        .sample_count = result.stats.iterations,
        .p99_ns = result.stats.p99_ns,
        .memory_bytes = result.memory_allocated,
    };
}

/// Batch convert framework results to persistable baselines
pub fn benchResultsToBaselines(
    allocator: std.mem.Allocator,
    results: []const BenchResult,
    git_commit: ?[]const u8,
    git_branch: ?[]const u8,
) ![]BenchmarkResult {
    var baselines = try allocator.alloc(BenchmarkResult, results.len);
    errdefer allocator.free(baselines);

    for (results, 0..) |result, i| {
        baselines[i] = benchResultToBaseline(result, git_commit, git_branch);
    }

    return baselines;
}

/// Run benchmarks and compare against baselines in one call
pub fn runAndCompare(
    allocator: std.mem.Allocator,
    runner: *BenchmarkRunner,
    store: *BaselineStore,
    config: ComparisonConfig,
) !RegressionReport {
    const baselines = try benchResultsToBaselines(
        allocator,
        runner.results.items,
        null,
        null,
    );
    defer allocator.free(baselines);

    return compareAllWithConfig(store, baselines, allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "module exports" {
    _ = BaselineStore;
    _ = BenchmarkResult;
    _ = ComparisonResult;
    _ = RegressionReport;
    _ = ComparisonConfig;
    _ = BenchResult;
    _ = BenchConfig;
    _ = Statistics;
    _ = BenchmarkRunner;
    _ = compareAll;
    _ = compareAllWithConfig;
    _ = compareSingle;
}

test "benchResultToBaseline conversion" {
    const bench_result = BenchResult{
        .config = .{
            .name = "test_bench",
            .category = "test",
        },
        .stats = .{
            .min_ns = 100,
            .max_ns = 200,
            .mean_ns = 150,
            .median_ns = 145,
            .std_dev_ns = 25,
            .p50_ns = 145,
            .p90_ns = 180,
            .p95_ns = 190,
            .p99_ns = 198,
            .iterations = 1000,
            .outliers_removed = 5,
            .total_time_ns = 150000,
        },
        .memory_allocated = 4096,
        .memory_freed = 4096,
        .timestamp = 1706000000,
    };

    const baseline = benchResultToBaseline(bench_result, "abc123", "main");

    try std.testing.expectEqualStrings("test_bench", baseline.name);
    try std.testing.expectEqualStrings("ops_per_sec", baseline.metric);
    try std.testing.expectEqualStrings("main", baseline.git_branch.?);
    try std.testing.expectEqualStrings("abc123", baseline.git_commit.?);
    try std.testing.expect(baseline.value > 0);
}

test {
    _ = baseline_store;
    _ = baseline_comparator;
    _ = framework;
}
