//! Benchmark System Module
//!
//! System-level benchmarking infrastructure including:
//! - **framework**: Core benchmark runner with statistical analysis
//! - **ci_integration**: CI/CD pipeline integration (GitHub Actions, GitLab, etc.)
//! - **industry_standard**: ANN-Benchmarks compatibility and industry metrics
//! - **baseline_store**: JSON-based baseline persistence for regression detection
//! - **baseline_comparator**: Comparison and reporting tools
//!
//! ## Quick Start
//!
//! ```zig
//! const system = @import("benchmarks/system/mod.zig");
//!
//! // Run benchmarks
//! var runner = system.framework.BenchmarkRunner.init(allocator);
//! defer runner.deinit();
//!
//! const result = try runner.run(.{ .name = "my_bench" }, myFunction, .{});
//!
//! // Save as baseline
//! var store = system.BaselineStore.init(allocator, "benchmarks/baselines");
//! defer store.deinit();
//! try store.saveBaseline(system.baseline_store.BenchmarkResult{
//!     .name = "my_bench",
//!     .metric = "ops_per_sec",
//!     .value = result.stats.opsPerSecond(),
//!     .unit = "ops/s",
//!     .timestamp = std.time.timestamp(),
//!     .git_branch = "main",
//! });
//!
//! // Compare against baseline
//! const report = try system.compareAll(&store, current_results, allocator);
//! defer report.deinit(allocator);
//!
//! if (report.hasRegressions()) {
//!     try report.format(std.io.getStdErr().writer());
//! }
//! ```
//!
//! ## Directory Structure for Baselines
//!
//! ```
//! benchmarks/baselines/
//! ├── main/                    # Main branch baselines
//! │   ├── vector_dot_128.json
//! │   ├── database_insert.json
//! │   └── ...
//! ├── releases/                # Release tag baselines
//! │   ├── v1.0.0/
//! │   │   └── ...
//! │   └── v1.1.0/
//! │       └── ...
//! └── branches/                # Feature branch baselines
//!     ├── feature_simd/
//!     │   └── ...
//!     └── fix_memory_leak/
//!         └── ...
//! ```

const std = @import("std");

// Core benchmark framework
pub const framework = @import("framework.zig");

// CI/CD integration
pub const ci_integration = @import("ci_integration.zig");

// Industry-standard metrics
pub const industry_standard = @import("industry_standard.zig");

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

// Re-export industry standard types

/// Hardware capability detection
pub const HardwareCapabilities = industry_standard.HardwareCapabilities;

/// Cache profiling statistics
pub const CacheStats = industry_standard.CacheStats;

/// Energy efficiency metrics
pub const EnergyMetrics = industry_standard.EnergyMetrics;

/// Scaling analysis results
pub const ScalingAnalysis = industry_standard.ScalingAnalysis;

/// ANN-Benchmarks compatible result
pub const AnnBenchmarkResult = industry_standard.AnnBenchmarkResult;

// Re-export CI integration types

/// CI benchmark report
pub const CiBenchmarkReport = ci_integration.CiBenchmarkReport;

/// CI runner configuration
pub const CiRunnerConfig = ci_integration.CiRunnerConfig;

/// CI platform detection
pub const CiPlatform = ci_integration.CiPlatform;

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
    // Convert runner results to baseline format
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
// Entry Point for Direct Execution
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Benchmark System Module\n", .{});
    std.debug.print("=======================\n\n", .{});

    // Run framework example
    std.debug.print("Running sample benchmark...\n", .{});
    var runner = BenchmarkRunner.init(allocator);
    defer runner.deinit();

    _ = try runner.run(
        .{
            .name = "sample_op",
            .category = "demo",
            .min_time_ns = 100_000_000, // 100ms
            .warmup_iterations = 10,
        },
        struct {
            fn op() u64 {
                var sum: u64 = 0;
                for (0..100) |i| {
                    sum +%= i * i;
                }
                return sum;
            }
        }.op,
        .{},
    );

    runner.printSummaryDebug();
    runner.exportJson();
}

// ============================================================================
// Tests
// ============================================================================

test "module exports" {
    // Verify all exports are accessible
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
    _ = HardwareCapabilities;
    _ = CacheStats;
    _ = EnergyMetrics;
    _ = ScalingAnalysis;
    _ = AnnBenchmarkResult;
    _ = CiBenchmarkReport;
    _ = CiRunnerConfig;
    _ = CiPlatform;
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
    // Run tests from submodules
    _ = baseline_store;
    _ = baseline_comparator;
    _ = framework;
}
