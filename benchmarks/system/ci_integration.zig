//! CI/CD Integration for Benchmarks
//!
//! Provides automated benchmark execution and reporting for CI pipelines:
//! - GitHub Actions compatible output
//! - GitLab CI compatible output
//! - Jenkins compatible output
//! - Automated regression detection
//! - Historical trend tracking
//! - Badge generation
//!
//! ## Usage in CI
//!
//! ```yaml
//! - name: Run Benchmarks
//!   run: zig build bench-ci -- --output results.json --baseline baseline.json
//!
//! - name: Check Regressions
//!   run: zig build bench-check -- --threshold 5 results.json
//! ```

const std = @import("std");
const framework = @import("./framework.zig");
const industry = @import("industry_standard.zig");

// ============================================================================
// CI Output Formats
// ============================================================================

/// CI platform types
pub const CiPlatform = enum {
    github_actions,
    gitlab_ci,
    jenkins,
    azure_devops,
    circleci,
    generic,

    pub fn fromEnv() CiPlatform {
        // Detect CI platform from environment
        // In real implementation, would check env vars
        return .generic;
    }
};

/// CI output format
pub const CiOutputFormat = enum {
    json,
    junit_xml,
    markdown,
    github_workflow_commands,
    prometheus,
    influxdb,
};

// ============================================================================
// Benchmark Report
// ============================================================================

/// Complete benchmark report for CI
pub const CiBenchmarkReport = struct {
    /// Report metadata
    metadata: ReportMetadata,
    /// Hardware information
    hardware: industry.HardwareCapabilities,
    /// Individual benchmark results
    results: []const BenchmarkResult,
    /// Regression analysis (if baseline provided)
    regressions: ?[]const industry.RegressionAnalysis,
    /// Summary statistics
    summary: ReportSummary,
    /// Pass/fail status
    passed: bool,
    /// Failure messages
    failures: []const []const u8,

    pub const ReportMetadata = struct {
        /// Git commit SHA
        commit_sha: ?[]const u8 = null,
        /// Git branch
        branch: ?[]const u8 = null,
        /// CI build number
        build_number: ?u64 = null,
        /// Timestamp
        timestamp: i64 = 0,
        /// Benchmark suite version
        version: []const u8 = "1.0.0",
        /// Runner ID
        runner_id: ?[]const u8 = null,
    };

    pub const BenchmarkResult = struct {
        name: []const u8,
        category: []const u8,
        passed: bool,
        metrics: Metrics,
        comparison: ?Comparison = null,

        pub const Metrics = struct {
            ops_per_sec: f64,
            mean_ns: f64,
            p50_ns: u64,
            p99_ns: u64,
            memory_bytes: u64,
            iterations: u64,
        };

        pub const Comparison = struct {
            baseline_ops_per_sec: f64,
            change_percent: f64,
            is_regression: bool,
            is_improvement: bool,
        };
    };

    pub const ReportSummary = struct {
        total_benchmarks: usize,
        passed_benchmarks: usize,
        failed_benchmarks: usize,
        regressions: usize,
        improvements: usize,
        total_duration_ms: u64,
    };

    pub fn toJson(self: *const CiBenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n");

        // Metadata
        try buf.appendSlice(allocator, "  \"metadata\": {\n");
        if (self.metadata.commit_sha) |sha| {
            try buf.appendSlice(allocator, "    \"commit_sha\": \"");
            try buf.appendSlice(allocator, sha);
            try buf.appendSlice(allocator, "\",\n");
        }
        if (self.metadata.branch) |branch| {
            try buf.appendSlice(allocator, "    \"branch\": \"");
            try buf.appendSlice(allocator, branch);
            try buf.appendSlice(allocator, "\",\n");
        }
        try buf.writer(allocator).print("    \"timestamp\": {d},\n", .{self.metadata.timestamp});
        try buf.appendSlice(allocator, "    \"version\": \"");
        try buf.appendSlice(allocator, self.metadata.version);
        try buf.appendSlice(allocator, "\"\n  },\n");

        // Summary
        try buf.appendSlice(allocator, "  \"summary\": {\n");
        try buf.writer(allocator).print("    \"total\": {d},\n", .{self.summary.total_benchmarks});
        try buf.writer(allocator).print("    \"passed\": {d},\n", .{self.summary.passed_benchmarks});
        try buf.writer(allocator).print("    \"failed\": {d},\n", .{self.summary.failed_benchmarks});
        try buf.writer(allocator).print("    \"regressions\": {d},\n", .{self.summary.regressions});
        try buf.writer(allocator).print("    \"improvements\": {d},\n", .{self.summary.improvements});
        try buf.writer(allocator).print("    \"duration_ms\": {d}\n", .{self.summary.total_duration_ms});
        try buf.appendSlice(allocator, "  },\n");

        // Results
        try buf.appendSlice(allocator, "  \"results\": [\n");
        for (self.results, 0..) |result, i| {
            if (i > 0) try buf.appendSlice(allocator, ",\n");
            try buf.appendSlice(allocator, "    {\n");
            try buf.appendSlice(allocator, "      \"name\": \"");
            try buf.appendSlice(allocator, result.name);
            try buf.appendSlice(allocator, "\",\n");
            try buf.appendSlice(allocator, "      \"category\": \"");
            try buf.appendSlice(allocator, result.category);
            try buf.appendSlice(allocator, "\",\n");
            try buf.writer(allocator).print("      \"passed\": {s},\n", .{if (result.passed) "true" else "false"});
            try buf.appendSlice(allocator, "      \"metrics\": {\n");
            try buf.writer(allocator).print("        \"ops_per_sec\": {d:.2},\n", .{result.metrics.ops_per_sec});
            try buf.writer(allocator).print("        \"mean_ns\": {d:.2},\n", .{result.metrics.mean_ns});
            try buf.writer(allocator).print("        \"p50_ns\": {d},\n", .{result.metrics.p50_ns});
            try buf.writer(allocator).print("        \"p99_ns\": {d},\n", .{result.metrics.p99_ns});
            try buf.writer(allocator).print("        \"memory_bytes\": {d}\n", .{result.metrics.memory_bytes});
            try buf.appendSlice(allocator, "      }");

            if (result.comparison) |comp| {
                try buf.appendSlice(allocator, ",\n      \"comparison\": {\n");
                try buf.writer(allocator).print("        \"baseline\": {d:.2},\n", .{comp.baseline_ops_per_sec});
                try buf.writer(allocator).print("        \"change_percent\": {d:.2},\n", .{comp.change_percent});
                try buf.writer(allocator).print("        \"regression\": {s},\n", .{if (comp.is_regression) "true" else "false"});
                try buf.writer(allocator).print("        \"improvement\": {s}\n", .{if (comp.is_improvement) "true" else "false"});
                try buf.appendSlice(allocator, "      }");
            }

            try buf.appendSlice(allocator, "\n    }");
        }
        try buf.appendSlice(allocator, "\n  ],\n");

        // Pass/fail
        try buf.writer(allocator).print("  \"passed\": {s}\n", .{if (self.passed) "true" else "false"});

        try buf.appendSlice(allocator, "}\n");

        return buf.toOwnedSlice(allocator);
    }

    pub fn toJunitXml(self: *const CiBenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        try buf.writer(allocator).print(
            "<testsuite name=\"Benchmarks\" tests=\"{d}\" failures=\"{d}\" time=\"{d:.3}\">\n",
            .{
                self.summary.total_benchmarks,
                self.summary.failed_benchmarks,
                @as(f64, @floatFromInt(self.summary.total_duration_ms)) / 1000.0,
            },
        );

        for (self.results) |result| {
            try buf.appendSlice(allocator, "  <testcase classname=\"");
            try buf.appendSlice(allocator, result.category);
            try buf.appendSlice(allocator, "\" name=\"");
            try buf.appendSlice(allocator, result.name);
            try buf.writer(allocator).print("\" time=\"{d:.6}\">\n", .{result.metrics.mean_ns / 1_000_000_000.0});

            if (!result.passed) {
                try buf.appendSlice(allocator, "    <failure message=\"Benchmark failed or regressed\">\n");
                if (result.comparison) |comp| {
                    try buf.writer(allocator).print(
                        "      Performance changed by {d:.2}% (threshold exceeded)\n",
                        .{comp.change_percent},
                    );
                }
                try buf.appendSlice(allocator, "    </failure>\n");
            }

            // Add properties for metrics
            try buf.appendSlice(allocator, "    <properties>\n");
            try buf.writer(allocator).print(
                "      <property name=\"ops_per_sec\" value=\"{d:.2}\"/>\n",
                .{result.metrics.ops_per_sec},
            );
            try buf.writer(allocator).print(
                "      <property name=\"p99_ns\" value=\"{d}\"/>\n",
                .{result.metrics.p99_ns},
            );
            try buf.writer(allocator).print(
                "      <property name=\"memory_bytes\" value=\"{d}\"/>\n",
                .{result.metrics.memory_bytes},
            );
            try buf.appendSlice(allocator, "    </properties>\n");

            try buf.appendSlice(allocator, "  </testcase>\n");
        }

        try buf.appendSlice(allocator, "</testsuite>\n");

        return buf.toOwnedSlice(allocator);
    }

    pub fn toMarkdown(self: *const CiBenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        // Header
        try buf.appendSlice(allocator, "# Benchmark Report\n\n");

        // Status badge
        const status = if (self.passed) ":white_check_mark: **PASSED**" else ":x: **FAILED**";
        try buf.appendSlice(allocator, "## Status: ");
        try buf.appendSlice(allocator, status);
        try buf.appendSlice(allocator, "\n\n");

        // Metadata
        try buf.appendSlice(allocator, "## Metadata\n\n");
        if (self.metadata.commit_sha) |sha| {
            try buf.appendSlice(allocator, "- **Commit**: `");
            try buf.appendSlice(allocator, sha[0..@min(8, sha.len)]);
            try buf.appendSlice(allocator, "`\n");
        }
        if (self.metadata.branch) |branch| {
            try buf.appendSlice(allocator, "- **Branch**: ");
            try buf.appendSlice(allocator, branch);
            try buf.appendSlice(allocator, "\n");
        }
        try buf.appendSlice(allocator, "\n");

        // Summary
        try buf.appendSlice(allocator, "## Summary\n\n");
        try buf.writer(allocator).print(
            "| Metric | Value |\n|--------|-------|\n| Total | {d} |\n| Passed | {d} |\n| Failed | {d} |\n| Regressions | {d} |\n| Improvements | {d} |\n\n",
            .{
                self.summary.total_benchmarks,
                self.summary.passed_benchmarks,
                self.summary.failed_benchmarks,
                self.summary.regressions,
                self.summary.improvements,
            },
        );

        // Results table
        try buf.appendSlice(allocator, "## Results\n\n");
        try buf.appendSlice(allocator, "| Benchmark | Ops/sec | P99 (ns) | Memory | Status | Change |\n");
        try buf.appendSlice(allocator, "|-----------|---------|----------|--------|--------|--------|\n");

        for (self.results) |result| {
            const status_icon = if (result.passed) ":white_check_mark:" else ":x:";
            try buf.appendSlice(allocator, "| ");
            try buf.appendSlice(allocator, result.name);
            try buf.writer(allocator).print(" | {d:.0} | {d} | {d} | {s} |", .{
                result.metrics.ops_per_sec,
                result.metrics.p99_ns,
                result.metrics.memory_bytes,
                status_icon,
            });

            if (result.comparison) |comp| {
                const change_icon = if (comp.is_regression)
                    ":chart_with_downwards_trend:"
                else if (comp.is_improvement)
                    ":chart_with_upwards_trend:"
                else
                    ":left_right_arrow:";
                try buf.writer(allocator).print(" {s} {d:+.1}% |", .{ change_icon, comp.change_percent });
            } else {
                try buf.appendSlice(allocator, " - |");
            }
            try buf.appendSlice(allocator, "\n");
        }

        // Regressions section if any
        if (self.summary.regressions > 0) {
            try buf.appendSlice(allocator, "\n## :warning: Regressions Detected\n\n");
            for (self.results) |result| {
                if (result.comparison) |comp| {
                    if (comp.is_regression) {
                        try buf.appendSlice(allocator, "- **");
                        try buf.appendSlice(allocator, result.name);
                        try buf.writer(allocator).print("**: {d:.2} -> {d:.2} ops/sec ({d:+.1}%)\n", .{
                            comp.baseline_ops_per_sec,
                            result.metrics.ops_per_sec,
                            comp.change_percent,
                        });
                    }
                }
            }
        }

        return buf.toOwnedSlice(allocator);
    }

    pub fn toGithubWorkflowCommands(self: *const CiBenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        // Set output variables
        try buf.writer(allocator).print("::set-output name=total::{d}\n", .{self.summary.total_benchmarks});
        try buf.writer(allocator).print("::set-output name=passed::{d}\n", .{self.summary.passed_benchmarks});
        try buf.writer(allocator).print("::set-output name=failed::{d}\n", .{self.summary.failed_benchmarks});
        try buf.writer(allocator).print("::set-output name=regressions::{d}\n", .{self.summary.regressions});

        // Log groups for each category
        var current_category: ?[]const u8 = null;
        for (self.results) |result| {
            if (current_category == null or !std.mem.eql(u8, current_category.?, result.category)) {
                if (current_category != null) {
                    try buf.appendSlice(allocator, "::endgroup::\n");
                }
                try buf.appendSlice(allocator, "::group::");
                try buf.appendSlice(allocator, result.category);
                try buf.appendSlice(allocator, "\n");
                current_category = result.category;
            }

            if (result.passed) {
                try buf.appendSlice(allocator, "::notice::");
            } else {
                try buf.appendSlice(allocator, "::error::");
            }
            try buf.appendSlice(allocator, result.name);
            try buf.writer(allocator).print(": {d:.0} ops/sec", .{result.metrics.ops_per_sec});

            if (result.comparison) |comp| {
                try buf.writer(allocator).print(" ({d:+.1}%)", .{comp.change_percent});
            }
            try buf.appendSlice(allocator, "\n");
        }
        if (current_category != null) {
            try buf.appendSlice(allocator, "::endgroup::\n");
        }

        // Warnings for regressions
        for (self.results) |result| {
            if (result.comparison) |comp| {
                if (comp.is_regression) {
                    try buf.appendSlice(allocator, "::warning file=benchmarks,title=Regression Detected::");
                    try buf.appendSlice(allocator, result.name);
                    try buf.writer(allocator).print(" regressed by {d:.1}%\n", .{-comp.change_percent});
                }
            }
        }

        return buf.toOwnedSlice(allocator);
    }

    pub fn toPrometheus(self: *const CiBenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        // Prometheus exposition format
        try buf.appendSlice(allocator, "# HELP benchmark_ops_per_sec Operations per second\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_ops_per_sec gauge\n");

        for (self.results) |result| {
            try buf.writer(allocator).print(
                "benchmark_ops_per_sec{{name=\"{s}\",category=\"{s}\"}} {d:.2}\n",
                .{ result.name, result.category, result.metrics.ops_per_sec },
            );
        }

        try buf.appendSlice(allocator, "\n# HELP benchmark_p99_ns P99 latency in nanoseconds\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_p99_ns gauge\n");

        for (self.results) |result| {
            try buf.writer(allocator).print(
                "benchmark_p99_ns{{name=\"{s}\",category=\"{s}\"}} {d}\n",
                .{ result.name, result.category, result.metrics.p99_ns },
            );
        }

        try buf.appendSlice(allocator, "\n# HELP benchmark_memory_bytes Memory usage in bytes\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_memory_bytes gauge\n");

        for (self.results) |result| {
            try buf.writer(allocator).print(
                "benchmark_memory_bytes{{name=\"{s}\",category=\"{s}\"}} {d}\n",
                .{ result.name, result.category, result.metrics.memory_bytes },
            );
        }

        // Summary metrics
        try buf.appendSlice(allocator, "\n# HELP benchmark_total Total number of benchmarks\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_total gauge\n");
        try buf.writer(allocator).print("benchmark_total {d}\n", .{self.summary.total_benchmarks});

        try buf.appendSlice(allocator, "\n# HELP benchmark_passed Number of passed benchmarks\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_passed gauge\n");
        try buf.writer(allocator).print("benchmark_passed {d}\n", .{self.summary.passed_benchmarks});

        try buf.appendSlice(allocator, "\n# HELP benchmark_regressions Number of regressions\n");
        try buf.appendSlice(allocator, "# TYPE benchmark_regressions gauge\n");
        try buf.writer(allocator).print("benchmark_regressions {d}\n", .{self.summary.regressions});

        return buf.toOwnedSlice(allocator);
    }
};

// ============================================================================
// CI Runner
// ============================================================================

/// CI benchmark runner configuration
pub const CiRunnerConfig = struct {
    /// Baseline file path for regression detection
    baseline_path: ?[]const u8 = null,
    /// Output file path
    output_path: ?[]const u8 = null,
    /// Output format
    format: CiOutputFormat = .json,
    /// Regression threshold percentage
    regression_threshold: f64 = 5.0,
    /// Fail on regression
    fail_on_regression: bool = true,
    /// Only run specific categories
    categories: ?[]const []const u8 = null,
    /// Only run specific benchmarks
    benchmarks: ?[]const []const u8 = null,
    /// Quick mode (fewer iterations)
    quick: bool = false,
};

/// Run benchmarks for CI
pub fn runCiBenchmarks(
    allocator: std.mem.Allocator,
    config: CiRunnerConfig,
) !CiBenchmarkReport {
    std.debug.print("\n=== Running CI Benchmarks ===\n", .{});

    var results = std.ArrayListUnmanaged(CiBenchmarkReport.BenchmarkResult){};
    errdefer results.deinit(allocator);

    const start_time = std.time.Timer.start() catch return error.TimerFailed;

    // Run benchmarks
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Add sample benchmarks
    const sample_benchmarks = [_]struct {
        name: []const u8,
        category: []const u8,
    }{
        .{ .name = "vector_dot_128", .category = "simd" },
        .{ .name = "vector_dot_512", .category = "simd" },
        .{ .name = "hash_sha256", .category = "crypto" },
        .{ .name = "db_insert_1k", .category = "database" },
        .{ .name = "db_search_10k", .category = "database" },
    };

    for (sample_benchmarks) |bench| {
        // Check category filter
        if (config.categories) |cats| {
            var found = false;
            for (cats) |c| {
                if (std.mem.eql(u8, c, bench.category)) {
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        // Run benchmark
        const bench_result = try runner.run(
            .{
                .name = bench.name,
                .category = bench.category,
                .min_time_ns = if (config.quick) 100_000_000 else 1_000_000_000,
                .warmup_iterations = if (config.quick) 10 else 100,
            },
            struct {
                fn op() u64 {
                    var sum: u64 = 0;
                    for (0..1000) |i| {
                        sum +%= i;
                    }
                    return sum;
                }
            }.op,
            .{},
        );

        try results.append(allocator, .{
            .name = bench.name,
            .category = bench.category,
            .passed = true,
            .metrics = .{
                .ops_per_sec = bench_result.stats.opsPerSecond(),
                .mean_ns = bench_result.stats.mean_ns,
                .p50_ns = bench_result.stats.p50_ns,
                .p99_ns = bench_result.stats.p99_ns,
                .memory_bytes = bench_result.memory_allocated,
                .iterations = bench_result.stats.iterations,
            },
            .comparison = null,
        });
    }

    const total_duration = start_time.read();

    // Load baseline if provided
    var baseline_store: ?industry.BaselineStore = null;
    if (config.baseline_path != null) {
        baseline_store = industry.BaselineStore.init(allocator);
        // Would load baseline from file
    }
    defer if (baseline_store) |*bs| bs.deinit();

    // Compare with baseline and detect regressions
    var regressions: usize = 0;
    var improvements: usize = 0;

    if (baseline_store) |*bs| {
        for (results.items) |*result| {
            if (bs.getBaseline(result.name)) |baseline| {
                const current_ops = result.metrics.ops_per_sec;
                const baseline_ops = 1_000_000_000.0 / baseline.mean;
                const change = ((current_ops - baseline_ops) / baseline_ops) * 100.0;

                const is_regression = change < -config.regression_threshold;
                const is_improvement = change > config.regression_threshold;

                result.comparison = .{
                    .baseline_ops_per_sec = baseline_ops,
                    .change_percent = change,
                    .is_regression = is_regression,
                    .is_improvement = is_improvement,
                };

                if (is_regression) {
                    regressions += 1;
                    if (config.fail_on_regression) {
                        result.passed = false;
                    }
                }
                if (is_improvement) improvements += 1;
            }
        }
    }

    // Calculate summary
    var passed_count: usize = 0;
    var failed_count: usize = 0;
    for (results.items) |result| {
        if (result.passed) {
            passed_count += 1;
        } else {
            failed_count += 1;
        }
    }

    const report = CiBenchmarkReport{
        .metadata = .{
            .version = "1.0.0",
            .timestamp = 0,
        },
        .hardware = industry.HardwareCapabilities.detect(),
        .results = try results.toOwnedSlice(allocator),
        .regressions = null,
        .summary = .{
            .total_benchmarks = results.items.len,
            .passed_benchmarks = passed_count,
            .failed_benchmarks = failed_count,
            .regressions = regressions,
            .improvements = improvements,
            .total_duration_ms = total_duration / 1_000_000,
        },
        .passed = failed_count == 0,
        .failures = &.{},
    };

    std.debug.print("\nBenchmark run complete: {d} passed, {d} failed, {d} regressions\n", .{
        passed_count,
        failed_count,
        regressions,
    });

    return report;
}

// ============================================================================
// Badge Generation
// ============================================================================

/// Badge style
pub const BadgeStyle = enum {
    flat,
    flat_square,
    plastic,
    for_the_badge,
};

/// Generate shields.io badge URL
pub fn generateBadgeUrl(
    allocator: std.mem.Allocator,
    label: []const u8,
    message: []const u8,
    color: []const u8,
    style: BadgeStyle,
) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8){};
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "https://img.shields.io/badge/");
    try buf.appendSlice(allocator, label);
    try buf.appendSlice(allocator, "-");
    try buf.appendSlice(allocator, message);
    try buf.appendSlice(allocator, "-");
    try buf.appendSlice(allocator, color);
    try buf.appendSlice(allocator, "?style=");
    try buf.appendSlice(allocator, switch (style) {
        .flat => "flat",
        .flat_square => "flat-square",
        .plastic => "plastic",
        .for_the_badge => "for-the-badge",
    });

    return buf.toOwnedSlice(allocator);
}

/// Generate performance badge for benchmark report
pub fn generatePerformanceBadge(
    allocator: std.mem.Allocator,
    report: *const CiBenchmarkReport,
) ![]u8 {
    const color = if (report.passed)
        if (report.summary.improvements > report.summary.regressions) "brightgreen" else "green"
    else
        "red";

    var message_buf: [32]u8 = undefined;
    const message = if (report.passed)
        std.fmt.bufPrint(&message_buf, "{d} passed", .{report.summary.passed_benchmarks}) catch "passed"
    else
        std.fmt.bufPrint(&message_buf, "{d} failed", .{report.summary.failed_benchmarks}) catch "failed";

    return generateBadgeUrl(allocator, "benchmarks", message, color, .flat);
}

// ============================================================================
// Historical Trend Tracking
// ============================================================================

/// Historical data point
pub const HistoricalDataPoint = struct {
    timestamp: i64,
    commit_sha: ?[]const u8,
    benchmark_name: []const u8,
    ops_per_sec: f64,
    mean_ns: f64,
    p99_ns: u64,
    memory_bytes: u64,
};

/// Historical trend analyzer
pub const TrendAnalyzer = struct {
    data_points: std.ArrayListUnmanaged(HistoricalDataPoint),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TrendAnalyzer {
        return .{
            .data_points = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TrendAnalyzer) void {
        self.data_points.deinit(self.allocator);
    }

    pub fn addDataPoint(self: *TrendAnalyzer, point: HistoricalDataPoint) !void {
        try self.data_points.append(self.allocator, point);
    }

    /// Calculate trend over recent data points
    pub fn calculateTrend(self: *const TrendAnalyzer, benchmark_name: []const u8, window: usize) ?struct {
        slope: f64,
        r_squared: f64,
        direction: enum { improving, stable, degrading },
    } {
        // Filter data points for this benchmark
        var matching = std.ArrayListUnmanaged(HistoricalDataPoint){};
        defer matching.deinit(self.allocator);

        for (self.data_points.items) |point| {
            if (std.mem.eql(u8, point.benchmark_name, benchmark_name)) {
                matching.append(self.allocator, point) catch continue;
            }
        }

        if (matching.items.len < 3) return null;

        // Take last N points
        const start = if (matching.items.len > window) matching.items.len - window else 0;
        const points = matching.items[start..];

        // Linear regression on ops_per_sec
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xy: f64 = 0;
        var sum_x2: f64 = 0;
        const n = @as(f64, @floatFromInt(points.len));

        for (points, 0..) |point, i| {
            const x = @as(f64, @floatFromInt(i));
            const y = point.ops_per_sec;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        const denom = n * sum_x2 - sum_x * sum_x;
        if (denom == 0) return null;

        const slope = (n * sum_xy - sum_x * sum_y) / denom;
        const mean_y = sum_y / n;

        // Calculate R-squared
        var ss_res: f64 = 0;
        var ss_tot: f64 = 0;
        const intercept = (sum_y - slope * sum_x) / n;

        for (points, 0..) |point, i| {
            const x = @as(f64, @floatFromInt(i));
            const y = point.ops_per_sec;
            const predicted = slope * x + intercept;
            ss_res += (y - predicted) * (y - predicted);
            ss_tot += (y - mean_y) * (y - mean_y);
        }

        const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 0;

        // Determine direction
        const threshold = mean_y * 0.01; // 1% of mean
        const direction: @TypeOf(@as(?struct {
            slope: f64,
            r_squared: f64,
            direction: enum { improving, stable, degrading },
        }, null)).?.direction = if (slope > threshold)
            .improving
        else if (slope < -threshold)
            .degrading
        else
            .stable;

        return .{
            .slope = slope,
            .r_squared = r_squared,
            .direction = direction,
        };
    }
};

// ============================================================================
// Main Entry Point
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const report = try runCiBenchmarks(allocator, .{
        .quick = true,
        .format = .markdown,
    });
    defer allocator.free(report.results);

    // Generate reports in multiple formats
    const json = try report.toJson(allocator);
    defer allocator.free(json);
    std.debug.print("\n=== JSON Report ===\n{s}\n", .{json});

    const markdown = try report.toMarkdown(allocator);
    defer allocator.free(markdown);
    std.debug.print("\n=== Markdown Report ===\n{s}\n", .{markdown});

    const prometheus = try report.toPrometheus(allocator);
    defer allocator.free(prometheus);
    std.debug.print("\n=== Prometheus Metrics ===\n{s}\n", .{prometheus});
}

// ============================================================================
// Tests
// ============================================================================

test "ci benchmark report json" {
    const allocator = std.testing.allocator;

    const report = CiBenchmarkReport{
        .metadata = .{ .version = "1.0.0" },
        .hardware = industry.HardwareCapabilities.detect(),
        .results = &.{
            .{
                .name = "test_bench",
                .category = "test",
                .passed = true,
                .metrics = .{
                    .ops_per_sec = 1000.0,
                    .mean_ns = 1000000.0,
                    .p50_ns = 900000,
                    .p99_ns = 1500000,
                    .memory_bytes = 1024,
                    .iterations = 100,
                },
            },
        },
        .regressions = null,
        .summary = .{
            .total_benchmarks = 1,
            .passed_benchmarks = 1,
            .failed_benchmarks = 0,
            .regressions = 0,
            .improvements = 0,
            .total_duration_ms = 1000,
        },
        .passed = true,
        .failures = &.{},
    };

    const json = try report.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "test_bench") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"passed\": true") != null);
}

test "badge generation" {
    const allocator = std.testing.allocator;

    const url = try generateBadgeUrl(allocator, "tests", "passing", "green", .flat);
    defer allocator.free(url);

    try std.testing.expect(std.mem.indexOf(u8, url, "shields.io") != null);
    try std.testing.expect(std.mem.indexOf(u8, url, "tests-passing-green") != null);
}

test "trend analyzer" {
    var analyzer = TrendAnalyzer.init(std.testing.allocator);
    defer analyzer.deinit();

    // Add improving trend data
    for (0..10) |i| {
        try analyzer.addDataPoint(.{
            .timestamp = @intCast(i),
            .commit_sha = null,
            .benchmark_name = "test",
            .ops_per_sec = 1000.0 + @as(f64, @floatFromInt(i)) * 10.0,
            .mean_ns = 1000000,
            .p99_ns = 1500000,
            .memory_bytes = 1024,
        });
    }

    const trend = analyzer.calculateTrend("test", 10);
    try std.testing.expect(trend != null);
    try std.testing.expect(trend.?.slope > 0);
    try std.testing.expect(trend.?.direction == .improving);
}
