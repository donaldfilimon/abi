//! Benchmark Baseline Comparator
//!
//! Provides comparison functionality for benchmark results against baselines:
//! - Individual and batch comparisons
//! - Configurable regression thresholds
//! - Detailed regression reports
//! - Multiple output formats (text, JSON, markdown)
//!
//! ## Usage
//!
//! ```zig
//! const comparator = @import("baseline_comparator.zig");
//! const store = @import("baseline_store.zig");
//!
//! var baseline_store = store.BaselineStore.init(allocator, "benchmarks/baselines");
//! defer baseline_store.deinit();
//!
//! const report = try comparator.compareAll(&baseline_store, current_results, allocator);
//! defer report.deinit(allocator);
//!
//! if (report.hasRegressions()) {
//!     var io_backend = std.Io.Threaded.init(allocator, .{
//!         .environ = std.process.Environ.empty,
//!     });
//!     defer io_backend.deinit();
//!     var stderr_buffer: [4096]u8 = undefined;
//!     var stderr_writer = std.Io.File.stderr().writer(io_backend.io(), &stderr_buffer);
//!     try report.format(&stderr_writer);
//!     return error.RegressionDetected;
//! }
//! ```

const std = @import("std");
const baseline_store = @import("baseline_store.zig");

pub const BenchmarkResult = baseline_store.BenchmarkResult;
pub const BaselineStore = baseline_store.BaselineStore;

/// Result of comparing a single benchmark against its baseline
pub const ComparisonResult = struct {
    /// Name of the benchmark
    benchmark_name: []const u8,
    /// Category (if available)
    category: ?[]const u8,
    /// Metric being compared
    metric: []const u8,
    /// Baseline value
    baseline_value: f64,
    /// Current value
    current_value: f64,
    /// Percentage change (positive = improvement for throughput)
    change_percent: f64,
    /// Comparison status
    status: Status,
    /// Baseline timestamp
    baseline_timestamp: i64,
    /// Current timestamp
    current_timestamp: i64,
    /// Baseline git commit (if available)
    baseline_commit: ?[]const u8,
    /// Current git commit (if available)
    current_commit: ?[]const u8,

    pub const Status = enum {
        /// Performance improved by more than the improvement threshold
        improved,
        /// Performance is within acceptable bounds
        stable,
        /// Performance regressed by more than the regression threshold
        regressed,
        /// No baseline data available for comparison
        no_baseline,

        pub fn symbol(self: Status) []const u8 {
            return switch (self) {
                .improved => "+",
                .stable => "=",
                .regressed => "-",
                .no_baseline => "?",
            };
        }

        pub fn description(self: Status) []const u8 {
            return switch (self) {
                .improved => "IMPROVED",
                .stable => "STABLE",
                .regressed => "REGRESSED",
                .no_baseline => "NO BASELINE",
            };
        }
    };

    /// Format the comparison result as a string
    pub fn formatLine(self: ComparisonResult, writer: anytype) !void {
        const symbol = self.status.symbol();
        const status_desc = self.status.description();

        try writer.print("[{s}] {s}: ", .{ symbol, self.benchmark_name });

        if (self.status == .no_baseline) {
            try writer.print("{d:.2} {s} (no baseline)\n", .{
                self.current_value,
                self.metric,
            });
        } else {
            try writer.print("{d:.2} -> {d:.2} ({d:+.2}%) [{s}]\n", .{
                self.baseline_value,
                self.current_value,
                self.change_percent,
                status_desc,
            });
        }
    }

    /// Free allocated strings
    pub fn deinit(self: *ComparisonResult, allocator: std.mem.Allocator) void {
        allocator.free(self.benchmark_name);
        allocator.free(self.metric);
        if (self.category) |c| allocator.free(c);
        if (self.baseline_commit) |bc| allocator.free(bc);
        if (self.current_commit) |cc| allocator.free(cc);
    }
};

/// Configuration for regression detection
pub const ComparisonConfig = struct {
    /// Threshold for detecting regression (percentage)
    /// Performance drop greater than this is considered a regression
    regression_threshold: f64 = 5.0,
    /// Threshold for detecting improvement (percentage)
    /// Performance gain greater than this is considered an improvement
    improvement_threshold: f64 = 5.0,
    /// Whether to use the main branch as fallback when no branch-specific baseline exists
    fallback_to_main: bool = true,
    /// Categories to include (null = all)
    include_categories: ?[]const []const u8 = null,
    /// Categories to exclude
    exclude_categories: ?[]const []const u8 = null,
    /// Minimum sample count for valid comparison
    min_sample_count: u64 = 10,
};

/// Comprehensive report of all benchmark comparisons
pub const RegressionReport = struct {
    /// Total number of benchmarks compared
    total_benchmarks: u32,
    /// Number of improved benchmarks
    improvements: u32,
    /// Number of regressed benchmarks
    regressions: u32,
    /// Number of stable benchmarks
    stable: u32,
    /// Number of benchmarks without baseline
    no_baseline: u32,
    /// Individual comparison results
    results: []ComparisonResult,
    /// Configuration used for comparison
    config: ComparisonConfig,
    /// Report generation timestamp
    timestamp: i64,
    /// Git branch being tested
    branch: ?[]const u8,
    /// Git commit being tested
    commit: ?[]const u8,

    /// Check if any regressions were detected
    pub fn hasRegressions(self: RegressionReport) bool {
        return self.regressions > 0;
    }

    /// Check if the benchmark suite passed (no regressions)
    pub fn passed(self: RegressionReport) bool {
        return !self.hasRegressions();
    }

    /// Get the worst regression (if any)
    pub fn worstRegression(self: RegressionReport) ?ComparisonResult {
        var worst: ?ComparisonResult = null;
        var worst_change: f64 = 0;

        for (self.results) |result| {
            if (result.status == .regressed) {
                if (worst == null or result.change_percent < worst_change) {
                    worst = result;
                    worst_change = result.change_percent;
                }
            }
        }

        return worst;
    }

    /// Get the best improvement (if any)
    pub fn bestImprovement(self: RegressionReport) ?ComparisonResult {
        var best: ?ComparisonResult = null;
        var best_change: f64 = 0;

        for (self.results) |result| {
            if (result.status == .improved) {
                if (best == null or result.change_percent > best_change) {
                    best = result;
                    best_change = result.change_percent;
                }
            }
        }

        return best;
    }

    /// Format the report as human-readable text
    pub fn format(self: RegressionReport, writer: anytype) !void {
        try writer.writeAll("\n");
        try writer.writeAll("================================================================================\n");
        try writer.writeAll("                     BENCHMARK REGRESSION REPORT\n");
        try writer.writeAll("================================================================================\n\n");

        // Summary
        try writer.writeAll("SUMMARY\n");
        try writer.writeAll("-------\n");
        try writer.print("Total Benchmarks: {d}\n", .{self.total_benchmarks});
        try writer.print("  Improved:       {d}\n", .{self.improvements});
        try writer.print("  Stable:         {d}\n", .{self.stable});
        try writer.print("  Regressed:      {d}\n", .{self.regressions});
        try writer.print("  No Baseline:    {d}\n", .{self.no_baseline});
        try writer.writeAll("\n");

        // Configuration
        try writer.print("Thresholds: regression={d:.1}%, improvement={d:.1}%\n\n", .{
            self.config.regression_threshold,
            self.config.improvement_threshold,
        });

        // Status
        if (self.hasRegressions()) {
            try writer.writeAll("STATUS: FAILED - Regressions detected!\n\n");
        } else {
            try writer.writeAll("STATUS: PASSED - No regressions detected.\n\n");
        }

        // Detailed results
        if (self.regressions > 0) {
            try writer.writeAll("REGRESSIONS\n");
            try writer.writeAll("-----------\n");
            for (self.results) |result| {
                if (result.status == .regressed) {
                    try result.formatLine(writer);
                }
            }
            try writer.writeAll("\n");
        }

        if (self.improvements > 0) {
            try writer.writeAll("IMPROVEMENTS\n");
            try writer.writeAll("------------\n");
            for (self.results) |result| {
                if (result.status == .improved) {
                    try result.formatLine(writer);
                }
            }
            try writer.writeAll("\n");
        }

        // All results
        try writer.writeAll("ALL RESULTS\n");
        try writer.writeAll("-----------\n");
        for (self.results) |result| {
            try result.formatLine(writer);
        }

        try writer.writeAll("\n================================================================================\n");
    }

    /// Format the report as JSON
    pub fn toJson(self: RegressionReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n");

        // Summary
        try buf.appendSlice(allocator, "  \"summary\": {\n");

        const total_str = try std.fmt.allocPrint(allocator, "    \"total\": {d},\n", .{self.total_benchmarks});
        defer allocator.free(total_str);
        try buf.appendSlice(allocator, total_str);

        const imp_str = try std.fmt.allocPrint(allocator, "    \"improvements\": {d},\n", .{self.improvements});
        defer allocator.free(imp_str);
        try buf.appendSlice(allocator, imp_str);

        const stable_str = try std.fmt.allocPrint(allocator, "    \"stable\": {d},\n", .{self.stable});
        defer allocator.free(stable_str);
        try buf.appendSlice(allocator, stable_str);

        const reg_str = try std.fmt.allocPrint(allocator, "    \"regressions\": {d},\n", .{self.regressions});
        defer allocator.free(reg_str);
        try buf.appendSlice(allocator, reg_str);

        const nobase_str = try std.fmt.allocPrint(allocator, "    \"no_baseline\": {d},\n", .{self.no_baseline});
        defer allocator.free(nobase_str);
        try buf.appendSlice(allocator, nobase_str);

        const passed_str = try std.fmt.allocPrint(allocator, "    \"passed\": {s}\n", .{if (self.passed()) "true" else "false"});
        defer allocator.free(passed_str);
        try buf.appendSlice(allocator, passed_str);
        try buf.appendSlice(allocator, "  },\n");

        // Config
        try buf.appendSlice(allocator, "  \"config\": {\n");
        const regthresh_str = try std.fmt.allocPrint(allocator, "    \"regression_threshold\": {d:.2},\n", .{self.config.regression_threshold});
        defer allocator.free(regthresh_str);
        try buf.appendSlice(allocator, regthresh_str);

        const impthresh_str = try std.fmt.allocPrint(allocator, "    \"improvement_threshold\": {d:.2}\n", .{self.config.improvement_threshold});
        defer allocator.free(impthresh_str);
        try buf.appendSlice(allocator, impthresh_str);
        try buf.appendSlice(allocator, "  },\n");

        // Metadata
        const ts_str = try std.fmt.allocPrint(allocator, "  \"timestamp\": {d},\n", .{self.timestamp});
        defer allocator.free(ts_str);
        try buf.appendSlice(allocator, ts_str);

        if (self.branch) |b| {
            try buf.appendSlice(allocator, "  \"branch\": \"");
            try buf.appendSlice(allocator, b);
            try buf.appendSlice(allocator, "\",\n");
        }
        if (self.commit) |c| {
            try buf.appendSlice(allocator, "  \"commit\": \"");
            try buf.appendSlice(allocator, c);
            try buf.appendSlice(allocator, "\",\n");
        }

        // Results
        try buf.appendSlice(allocator, "  \"results\": [\n");
        for (self.results, 0..) |result, i| {
            if (i > 0) try buf.appendSlice(allocator, ",\n");
            try buf.appendSlice(allocator, "    {\n");
            try buf.appendSlice(allocator, "      \"name\": \"");
            try buf.appendSlice(allocator, result.benchmark_name);
            try buf.appendSlice(allocator, "\",\n");
            try buf.appendSlice(allocator, "      \"metric\": \"");
            try buf.appendSlice(allocator, result.metric);
            try buf.appendSlice(allocator, "\",\n");

            const base_str = try std.fmt.allocPrint(allocator, "      \"baseline\": {d:.6},\n", .{result.baseline_value});
            defer allocator.free(base_str);
            try buf.appendSlice(allocator, base_str);

            const curr_str = try std.fmt.allocPrint(allocator, "      \"current\": {d:.6},\n", .{result.current_value});
            defer allocator.free(curr_str);
            try buf.appendSlice(allocator, curr_str);

            const change_str = try std.fmt.allocPrint(allocator, "      \"change_percent\": {d:.4},\n", .{result.change_percent});
            defer allocator.free(change_str);
            try buf.appendSlice(allocator, change_str);

            try buf.appendSlice(allocator, "      \"status\": \"");
            try buf.appendSlice(allocator, result.status.description());
            try buf.appendSlice(allocator, "\"\n");
            try buf.appendSlice(allocator, "    }");
        }
        try buf.appendSlice(allocator, "\n  ]\n");

        try buf.appendSlice(allocator, "}\n");

        return buf.toOwnedSlice(allocator);
    }

    /// Format the report as Markdown
    pub fn toMarkdown(self: RegressionReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        // Header
        try buf.appendSlice(allocator, "# Benchmark Regression Report\n\n");

        // Status badge
        if (self.hasRegressions()) {
            try buf.appendSlice(allocator, "**Status:** :x: FAILED - Regressions detected\n\n");
        } else {
            try buf.appendSlice(allocator, "**Status:** :white_check_mark: PASSED\n\n");
        }

        // Summary table
        try buf.appendSlice(allocator, "## Summary\n\n");
        try buf.appendSlice(allocator, "| Metric | Count |\n");
        try buf.appendSlice(allocator, "|--------|-------|\n");

        const total_row = try std.fmt.allocPrint(allocator, "| Total | {d} |\n", .{self.total_benchmarks});
        defer allocator.free(total_row);
        try buf.appendSlice(allocator, total_row);

        const imp_row = try std.fmt.allocPrint(allocator, "| Improved | {d} |\n", .{self.improvements});
        defer allocator.free(imp_row);
        try buf.appendSlice(allocator, imp_row);

        const stable_row = try std.fmt.allocPrint(allocator, "| Stable | {d} |\n", .{self.stable});
        defer allocator.free(stable_row);
        try buf.appendSlice(allocator, stable_row);

        const reg_row = try std.fmt.allocPrint(allocator, "| Regressed | {d} |\n", .{self.regressions});
        defer allocator.free(reg_row);
        try buf.appendSlice(allocator, reg_row);

        const nobase_row = try std.fmt.allocPrint(allocator, "| No Baseline | {d} |\n", .{self.no_baseline});
        defer allocator.free(nobase_row);
        try buf.appendSlice(allocator, nobase_row);
        try buf.appendSlice(allocator, "\n");

        // Regressions section
        if (self.regressions > 0) {
            try buf.appendSlice(allocator, "## :warning: Regressions\n\n");
            try buf.appendSlice(allocator, "| Benchmark | Baseline | Current | Change |\n");
            try buf.appendSlice(allocator, "|-----------|----------|---------|--------|\n");
            for (self.results) |result| {
                if (result.status == .regressed) {
                    const row = try std.fmt.allocPrint(allocator, "| {s} | {d:.2} | {d:.2} | {d:+.2}% |\n", .{
                        result.benchmark_name,
                        result.baseline_value,
                        result.current_value,
                        result.change_percent,
                    });
                    defer allocator.free(row);
                    try buf.appendSlice(allocator, row);
                }
            }
            try buf.appendSlice(allocator, "\n");
        }

        // Improvements section
        if (self.improvements > 0) {
            try buf.appendSlice(allocator, "## :rocket: Improvements\n\n");
            try buf.appendSlice(allocator, "| Benchmark | Baseline | Current | Change |\n");
            try buf.appendSlice(allocator, "|-----------|----------|---------|--------|\n");
            for (self.results) |result| {
                if (result.status == .improved) {
                    const row = try std.fmt.allocPrint(allocator, "| {s} | {d:.2} | {d:.2} | {d:+.2}% |\n", .{
                        result.benchmark_name,
                        result.baseline_value,
                        result.current_value,
                        result.change_percent,
                    });
                    defer allocator.free(row);
                    try buf.appendSlice(allocator, row);
                }
            }
            try buf.appendSlice(allocator, "\n");
        }

        // All results
        try buf.appendSlice(allocator, "## All Results\n\n");
        try buf.appendSlice(allocator, "| Status | Benchmark | Baseline | Current | Change |\n");
        try buf.appendSlice(allocator, "|--------|-----------|----------|---------|--------|\n");
        for (self.results) |result| {
            const status_icon = switch (result.status) {
                .improved => ":arrow_up:",
                .stable => ":heavy_minus_sign:",
                .regressed => ":arrow_down:",
                .no_baseline => ":grey_question:",
            };
            const row = try std.fmt.allocPrint(allocator, "| {s} | {s} | {d:.2} | {d:.2} | {d:+.2}% |\n", .{
                status_icon,
                result.benchmark_name,
                result.baseline_value,
                result.current_value,
                result.change_percent,
            });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
        }

        return buf.toOwnedSlice(allocator);
    }

    /// Free allocated memory
    pub fn deinit(self: *RegressionReport, allocator: std.mem.Allocator) void {
        for (self.results) |*result| {
            var r = result.*;
            r.deinit(allocator);
        }
        allocator.free(self.results);
        if (self.branch) |b| allocator.free(b);
        if (self.commit) |c| allocator.free(c);
    }
};

/// Compare all current results against their baselines
pub fn compareAll(
    store: *BaselineStore,
    current_results: []const BenchmarkResult,
    allocator: std.mem.Allocator,
) !RegressionReport {
    return compareAllWithConfig(store, current_results, allocator, .{});
}

/// Compare all current results against their baselines with custom configuration
pub fn compareAllWithConfig(
    store: *BaselineStore,
    current_results: []const BenchmarkResult,
    allocator: std.mem.Allocator,
    config: ComparisonConfig,
) !RegressionReport {
    var results = std.ArrayListUnmanaged(ComparisonResult){};
    errdefer {
        for (results.items) |*r| r.deinit(allocator);
        results.deinit(allocator);
    }

    var improvements: u32 = 0;
    var regressions: u32 = 0;
    var stable: u32 = 0;
    var no_baseline: u32 = 0;

    // Track branch/commit from first result
    var branch: ?[]const u8 = null;
    var commit: ?[]const u8 = null;

    for (current_results) |current| {
        // Filter by category if configured
        if (config.include_categories) |include| {
            if (current.category) |cat| {
                var found = false;
                for (include) |inc| {
                    if (std.mem.eql(u8, cat, inc)) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;
            } else {
                continue; // No category, skip
            }
        }

        if (config.exclude_categories) |exclude| {
            if (current.category) |cat| {
                var skip = false;
                for (exclude) |exc| {
                    if (std.mem.eql(u8, cat, exc)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;
            }
        }

        // Track branch/commit
        if (branch == null and current.git_branch != null) {
            branch = try allocator.dupe(u8, current.git_branch.?);
        }
        if (commit == null and current.git_commit != null) {
            commit = try allocator.dupe(u8, current.git_commit.?);
        }

        // Load baseline
        var baseline_opt = store.loadBaselineForBranch(current.name, current.git_branch) catch null;

        // Fallback to main if configured
        if (baseline_opt == null and config.fallback_to_main) {
            if (current.git_branch) |b| {
                if (!std.mem.eql(u8, b, "main") and !std.mem.eql(u8, b, "master")) {
                    baseline_opt = store.loadBaselineForBranch(current.name, "main") catch null;
                }
            }
        }

        var comparison_result: ComparisonResult = undefined;

        if (baseline_opt) |baseline| {
            defer {
                var b = baseline;
                b.deinit(store.allocator);
            }

            // Calculate change
            const change = calculateChange(baseline.value, current.value, current.metric);

            // Determine status
            const status: ComparisonResult.Status = if (change < -config.regression_threshold)
                .regressed
            else if (change > config.improvement_threshold)
                .improved
            else
                .stable;

            switch (status) {
                .improved => improvements += 1,
                .stable => stable += 1,
                .regressed => regressions += 1,
                .no_baseline => no_baseline += 1,
            }

            comparison_result = .{
                .benchmark_name = try allocator.dupe(u8, current.name),
                .category = if (current.category) |c| try allocator.dupe(u8, c) else null,
                .metric = try allocator.dupe(u8, current.metric),
                .baseline_value = baseline.value,
                .current_value = current.value,
                .change_percent = change,
                .status = status,
                .baseline_timestamp = baseline.timestamp,
                .current_timestamp = current.timestamp,
                .baseline_commit = if (baseline.git_commit) |bc| try allocator.dupe(u8, bc) else null,
                .current_commit = if (current.git_commit) |cc| try allocator.dupe(u8, cc) else null,
            };
        } else {
            no_baseline += 1;

            comparison_result = .{
                .benchmark_name = try allocator.dupe(u8, current.name),
                .category = if (current.category) |c| try allocator.dupe(u8, c) else null,
                .metric = try allocator.dupe(u8, current.metric),
                .baseline_value = 0,
                .current_value = current.value,
                .change_percent = 0,
                .status = .no_baseline,
                .baseline_timestamp = 0,
                .current_timestamp = current.timestamp,
                .baseline_commit = null,
                .current_commit = if (current.git_commit) |cc| try allocator.dupe(u8, cc) else null,
            };
        }

        try results.append(allocator, comparison_result);
    }

    return .{
        .total_benchmarks = @intCast(results.items.len),
        .improvements = improvements,
        .regressions = regressions,
        .stable = stable,
        .no_baseline = no_baseline,
        .results = try results.toOwnedSlice(allocator),
        .config = config,
        .timestamp = std.time.timestamp(),
        .branch = branch,
        .commit = commit,
    };
}

/// Calculate percentage change between baseline and current value
fn calculateChange(baseline: f64, current: f64, metric: []const u8) f64 {
    if (baseline == 0) return 0;

    const change = ((current - baseline) / baseline) * 100.0;

    // For latency metrics, invert the sign (lower is better)
    if (std.mem.indexOf(u8, metric, "latency") != null or
        std.mem.indexOf(u8, metric, "_ns") != null or
        std.mem.indexOf(u8, metric, "_ms") != null or
        std.mem.indexOf(u8, metric, "time") != null)
    {
        return -change;
    }

    return change;
}

/// Compare a single benchmark result against its baseline
pub fn compareSingle(
    store: *BaselineStore,
    current: BenchmarkResult,
    allocator: std.mem.Allocator,
    config: ComparisonConfig,
) !ComparisonResult {
    const results = [_]BenchmarkResult{current};
    var report = try compareAllWithConfig(store, &results, allocator, config);
    defer {
        if (report.branch) |b| allocator.free(b);
        if (report.commit) |c| allocator.free(c);
    }

    if (report.results.len == 0) {
        return error.NoResults;
    }

    // Transfer ownership of the single result
    const result = report.results[0];
    allocator.free(report.results);
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "ComparisonResult status symbols" {
    try std.testing.expectEqualStrings("+", ComparisonResult.Status.improved.symbol());
    try std.testing.expectEqualStrings("=", ComparisonResult.Status.stable.symbol());
    try std.testing.expectEqualStrings("-", ComparisonResult.Status.regressed.symbol());
    try std.testing.expectEqualStrings("?", ComparisonResult.Status.no_baseline.symbol());
}

test "calculateChange throughput metrics" {
    // 10% improvement in throughput
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), calculateChange(100, 110, "ops_per_sec"), 0.01);
    // 20% regression in throughput
    try std.testing.expectApproxEqAbs(@as(f64, -20.0), calculateChange(100, 80, "ops_per_sec"), 0.01);
}

test "calculateChange latency metrics" {
    // 10% faster (lower latency = improvement)
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), calculateChange(100, 90, "latency_ns"), 0.01);
    // 20% slower (higher latency = regression)
    try std.testing.expectApproxEqAbs(@as(f64, -20.0), calculateChange(100, 120, "latency_ns"), 0.01);
}

test "RegressionReport hasRegressions" {
    var report = RegressionReport{
        .total_benchmarks = 5,
        .improvements = 1,
        .regressions = 0,
        .stable = 3,
        .no_baseline = 1,
        .results = &.{},
        .config = .{},
        .timestamp = 0,
        .branch = null,
        .commit = null,
    };

    try std.testing.expect(!report.hasRegressions());
    try std.testing.expect(report.passed());

    report.regressions = 1;
    try std.testing.expect(report.hasRegressions());
    try std.testing.expect(!report.passed());
}

test "RegressionReport JSON output" {
    const allocator = std.testing.allocator;

    var results_arr = [_]ComparisonResult{
        .{
            .benchmark_name = "test_bench",
            .category = null,
            .metric = "ops_per_sec",
            .baseline_value = 1000,
            .current_value = 1100,
            .change_percent = 10.0,
            .status = .improved,
            .baseline_timestamp = 1706000000,
            .current_timestamp = 1706100000,
            .baseline_commit = null,
            .current_commit = null,
        },
    };

    var report = RegressionReport{
        .total_benchmarks = 1,
        .improvements = 1,
        .regressions = 0,
        .stable = 0,
        .no_baseline = 0,
        .results = &results_arr,
        .config = .{},
        .timestamp = 1706100000,
        .branch = null,
        .commit = null,
    };

    const json = try report.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"passed\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"improvements\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "test_bench") != null);
}
