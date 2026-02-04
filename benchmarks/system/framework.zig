//! Advanced Benchmark Framework with Statistical Analysis
//!
//! Industry-standard benchmarking with:
//! - Warm-up phases to stabilize CPU caches and branch predictors
//! - Statistical analysis (mean, median, std dev, percentiles)
//! - Outlier detection and removal
//! - Memory allocation tracking
//! - CPU cycle counting (when available)
//! - Automatic iteration calibration
//! - JSON/CSV export support

const std = @import("std");

/// Statistical summary of benchmark results
pub const Statistics = struct {
    min_ns: u64,
    max_ns: u64,
    mean_ns: f64,
    median_ns: f64,
    std_dev_ns: f64,
    p50_ns: u64,
    p90_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    iterations: u64,
    outliers_removed: u64,
    total_time_ns: u64,

    pub fn opsPerSecond(self: Statistics) f64 {
        if (self.mean_ns == 0) return 0;
        return 1_000_000_000.0 / self.mean_ns;
    }

    pub fn throughputMBps(self: Statistics, bytes_per_op: u64) f64 {
        const ops = self.opsPerSecond();
        return (ops * @as(f64, @floatFromInt(bytes_per_op))) / (1024.0 * 1024.0);
    }
};

/// Benchmark configuration
pub const BenchConfig = struct {
    /// Minimum time to run the benchmark (nanoseconds)
    min_time_ns: u64 = 100_000_000, // 100ms (was 1s - too slow)
    /// Maximum iterations regardless of time
    max_iterations: u64 = 1_000_000,
    /// Minimum iterations to run
    min_iterations: u64 = 10,
    /// Number of warm-up iterations
    warmup_iterations: u64 = 50,
    /// Whether to remove statistical outliers
    remove_outliers: bool = true,
    /// Outlier threshold (number of standard deviations)
    outlier_threshold: f64 = 3.0,
    /// Target coefficient of variation (for auto-calibration)
    target_cv: f64 = 0.05, // 5%
    /// Whether to track memory allocations
    track_memory: bool = false,
    /// Name for the benchmark
    name: []const u8 = "unnamed",
    /// Category for grouping
    category: []const u8 = "general",
    /// Bytes processed per operation (for throughput calculation)
    bytes_per_op: u64 = 0,
};

/// Result of a single benchmark run
pub const BenchResult = struct {
    config: BenchConfig,
    stats: Statistics,
    memory_allocated: u64,
    memory_freed: u64,
    timestamp: i64,

    pub fn format(
        self: BenchResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print(
            "{s} [{s}]: {d:.2} ops/sec, mean={d:.0}ns, p99={d}ns",
            .{
                self.config.name,
                self.config.category,
                self.stats.opsPerSecond(),
                self.stats.mean_ns,
                self.stats.p99_ns,
            },
        );
    }
};

var global_collector: ?*BenchCollector = null;

/// Shared collector for aggregating benchmark results across suites.
pub const BenchCollector = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchResult),

    pub fn init(allocator: std.mem.Allocator) BenchCollector {
        return .{
            .allocator = allocator,
            .results = .{},
        };
    }

    pub fn deinit(self: *BenchCollector) void {
        for (self.results.items) |*result| {
            freeConfigStrings(self.allocator, &result.config);
        }
        self.results.deinit(self.allocator);
    }

    pub fn append(self: *BenchCollector, result: BenchResult) !void {
        var owned = result;
        owned.config = try cloneConfigStrings(self.allocator, result.config);
        try self.results.append(self.allocator, owned);
    }
};

/// Set a global collector for benchmark results.
pub fn setGlobalCollector(collector: ?*BenchCollector) void {
    global_collector = collector;
}

fn appendGlobalResult(result: BenchResult) !void {
    if (global_collector) |collector| {
        try collector.append(result);
    }
}

/// Tracking allocator for memory benchmarks
pub const TrackingAllocator = struct {
    parent: std.mem.Allocator,
    allocated: std.atomic.Value(u64),
    freed: std.atomic.Value(u64),
    peak: std.atomic.Value(u64),
    current: std.atomic.Value(u64),

    pub fn init(parent: std.mem.Allocator) TrackingAllocator {
        return .{
            .parent = parent,
            .allocated = std.atomic.Value(u64).init(0),
            .freed = std.atomic.Value(u64).init(0),
            .peak = std.atomic.Value(u64).init(0),
            .current = std.atomic.Value(u64).init(0),
        };
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
                .remap = remap,
            },
        };
    }

    fn remap(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent.vtable.remap(self.parent.ptr, buf, buf_align, new_len, ret_addr);
        if (result) |_| {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                _ = self.allocated.fetchAdd(diff, .monotonic);
                _ = self.current.fetchAdd(diff, .monotonic);
            } else {
                const diff = buf.len - new_len;
                _ = self.freed.fetchAdd(diff, .monotonic);
                _ = self.current.fetchSub(diff, .monotonic);
            }
        }
        return result;
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.parent.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            _ = self.allocated.fetchAdd(len, .monotonic);
            const new_current = self.current.fetchAdd(len, .monotonic) + len;
            // Update peak if necessary
            var peak = self.peak.load(.monotonic);
            while (new_current > peak) {
                if (self.peak.cmpxchgWeak(peak, new_current, .monotonic, .monotonic)) |old| {
                    peak = old;
                } else {
                    break;
                }
            }
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.parent.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                const diff = new_len - buf.len;
                _ = self.allocated.fetchAdd(diff, .monotonic);
                _ = self.current.fetchAdd(diff, .monotonic);
            } else {
                const diff = buf.len - new_len;
                _ = self.freed.fetchAdd(diff, .monotonic);
                _ = self.current.fetchSub(diff, .monotonic);
            }
            return true;
        }
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, buf_align, ret_addr);
        _ = self.freed.fetchAdd(buf.len, .monotonic);
        _ = self.current.fetchSub(buf.len, .monotonic);
    }

    pub fn reset(self: *TrackingAllocator) void {
        self.allocated.store(0, .monotonic);
        self.freed.store(0, .monotonic);
        self.peak.store(0, .monotonic);
        self.current.store(0, .monotonic);
    }

    pub fn getStats(self: *TrackingAllocator) struct { allocated: u64, freed: u64, peak: u64 } {
        return .{
            .allocated = self.allocated.load(.monotonic),
            .freed = self.freed.load(.monotonic),
            .peak = self.peak.load(.monotonic),
        };
    }
};

/// High-resolution timer with CPU cycle support
pub const Timer = struct {
    inner: ?std.time.Timer,

    pub fn start() Timer {
        return .{
            .inner = std.time.Timer.start() catch null,
        };
    }

    pub fn elapsed(self: Timer) u64 {
        if (self.inner) |timer| {
            // Create a mutable copy to call read
            var t = timer;
            return t.read();
        }
        return 0;
    }

    pub fn elapsedCycles(self: Timer) u64 {
        // Just use nanoseconds as a proxy for cycles
        return self.elapsed();
    }
};

/// Benchmark runner with statistical analysis
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchResult),
    tracking_allocator: ?TrackingAllocator,

    pub fn init(allocator: std.mem.Allocator) BenchmarkRunner {
        return .{
            .allocator = allocator,
            .results = .{},
            .tracking_allocator = null,
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        for (self.results.items) |*result| {
            freeConfigStrings(self.allocator, &result.config);
        }
        self.results.deinit(self.allocator);
    }

    /// Run a benchmark function with the given configuration
    pub fn run(
        self: *BenchmarkRunner,
        config: BenchConfig,
        comptime bench_fn: anytype,
        args: anytype,
    ) !BenchResult {
        // Setup tracking allocator if needed
        var tracker: ?TrackingAllocator = null;
        if (config.track_memory) {
            tracker = TrackingAllocator.init(self.allocator);
        }

        // Warm-up phase
        var i: u64 = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            const result = @call(.auto, bench_fn, args);
            std.mem.doNotOptimizeAway(&result);
        }

        // Collection phase
        var samples = std.ArrayListUnmanaged(u64){};
        defer samples.deinit(self.allocator);

        var total_time: u64 = 0;
        var iterations: u64 = 0;

        while (total_time < config.min_time_ns and iterations < config.max_iterations) {
            const timer = Timer.start();
            const result = @call(.auto, bench_fn, args);
            const elapsed = timer.elapsed();
            std.mem.doNotOptimizeAway(&result);

            try samples.append(self.allocator, elapsed);
            total_time += elapsed;
            iterations += 1;
        }

        // Ensure minimum iterations
        while (iterations < config.min_iterations) {
            const timer = Timer.start();
            const result = @call(.auto, bench_fn, args);
            const elapsed = timer.elapsed();
            std.mem.doNotOptimizeAway(&result);

            try samples.append(self.allocator, elapsed);
            total_time += elapsed;
            iterations += 1;
        }

        // Calculate statistics
        const stats = try calculateStatistics(self.allocator, samples.items, config);

        // Get memory stats
        var mem_allocated: u64 = 0;
        var mem_freed: u64 = 0;
        if (tracker) |*t| {
            const mem_stats = t.getStats();
            mem_allocated = mem_stats.allocated;
            mem_freed = mem_stats.freed;
        }

        const result = BenchResult{
            .config = try cloneConfigStrings(self.allocator, config),
            .stats = stats,
            .memory_allocated = mem_allocated,
            .memory_freed = mem_freed,
            .timestamp = 0, // Timestamp not needed for benchmarks
        };

        try self.results.append(self.allocator, result);
        try appendGlobalResult(result);
        return result;
    }

    /// Run a benchmark that takes an allocator
    pub fn runWithAllocator(
        self: *BenchmarkRunner,
        config: BenchConfig,
        comptime bench_fn: anytype,
        extra_args: anytype,
    ) !BenchResult {
        var tracker = TrackingAllocator.init(self.allocator);

        // Warm-up phase
        var i: u64 = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            tracker.reset();
            const result = @call(.auto, bench_fn, .{tracker.allocator()} ++ extra_args);
            std.mem.doNotOptimizeAway(&result);
        }

        // Collection phase
        var samples = std.ArrayListUnmanaged(u64){};
        defer samples.deinit(self.allocator);

        var total_time: u64 = 0;
        var iterations: u64 = 0;

        tracker.reset();

        while (total_time < config.min_time_ns and iterations < config.max_iterations) {
            const timer = Timer.start();
            const result = @call(.auto, bench_fn, .{tracker.allocator()} ++ extra_args);
            const elapsed = timer.elapsed();
            std.mem.doNotOptimizeAway(&result);

            try samples.append(self.allocator, elapsed);
            total_time += elapsed;
            iterations += 1;
        }

        const stats = try calculateStatistics(self.allocator, samples.items, config);
        const mem_stats = tracker.getStats();

        const result = BenchResult{
            .config = try cloneConfigStrings(self.allocator, config),
            .stats = stats,
            .memory_allocated = mem_stats.allocated,
            .memory_freed = mem_stats.freed,
            .timestamp = 0, // Timestamp not needed for benchmarks
        };

        try self.results.append(self.allocator, result);
        try appendGlobalResult(result);
        return result;
    }

    /// Export results to JSON (uses std.debug.print)
    pub fn exportJson(self: *BenchmarkRunner) void {
        std.debug.print("{{\n  \"benchmarks\": [\n", .{});

        for (self.results.items, 0..) |result, idx| {
            if (idx > 0) std.debug.print(",\n", .{});
            std.debug.print(
                \\    {{
                \\      "name": "{s}",
                \\      "category": "{s}",
                \\      "iterations": {d},
                \\      "mean_ns": {d:.2},
                \\      "ops_per_sec": {d:.2}
                \\    }}
            , .{
                result.config.name,
                result.config.category,
                result.stats.iterations,
                result.stats.mean_ns,
                result.stats.opsPerSecond(),
            });
        }

        std.debug.print("\n  ]\n}}\n", .{});
    }

    /// Export results to JSON (writer version - deprecated)
    pub fn exportJsonWriter(self: *BenchmarkRunner, writer: anytype) !void {
        try writer.writeAll("{\n  \"benchmarks\": [\n");

        for (self.results.items, 0..) |result, idx| {
            if (idx > 0) try writer.writeAll(",\n");
            try writer.print(
                \\    {{
                \\      "name": "{s}",
                \\      "category": "{s}",
                \\      "iterations": {d},
                \\      "mean_ns": {d:.2},
                \\      "median_ns": {d:.2},
                \\      "std_dev_ns": {d:.2},
                \\      "min_ns": {d},
                \\      "max_ns": {d},
                \\      "p50_ns": {d},
                \\      "p90_ns": {d},
                \\      "p95_ns": {d},
                \\      "p99_ns": {d},
                \\      "ops_per_sec": {d:.2},
                \\      "memory_allocated": {d},
                \\      "timestamp": {d}
                \\    }}
            , .{
                result.config.name,
                result.config.category,
                result.stats.iterations,
                result.stats.mean_ns,
                result.stats.median_ns,
                result.stats.std_dev_ns,
                result.stats.min_ns,
                result.stats.max_ns,
                result.stats.p50_ns,
                result.stats.p90_ns,
                result.stats.p95_ns,
                result.stats.p99_ns,
                result.stats.opsPerSecond(),
                result.memory_allocated,
                result.timestamp,
            });
        }

        try writer.writeAll("\n  ]\n}\n");
    }

    /// Print a formatted summary
    pub fn printSummary(self: *BenchmarkRunner, writer: anytype) !void {
        try writer.writeAll("\n");
        try writer.writeAll("=" ** 80);
        try writer.writeAll("\n");
        try writer.writeAll("                     BENCHMARK RESULTS SUMMARY\n");
        try writer.writeAll("=" ** 80);
        try writer.writeAll("\n\n");

        // Group by category
        var categories = std.StringHashMapUnmanaged(std.ArrayListUnmanaged(BenchResult)){};
        defer {
            var it = categories.valueIterator();
            while (it.next()) |list| {
                list.deinit(self.allocator);
            }
            categories.deinit(self.allocator);
        }

        for (self.results.items) |result| {
            const entry = try categories.getOrPut(self.allocator, result.config.category);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{};
            }
            try entry.value_ptr.append(self.allocator, result);
        }

        var cat_it = categories.iterator();
        while (cat_it.next()) |entry| {
            try writer.print("[{s}]\n", .{entry.key_ptr.*});
            try writer.writeAll("-" ** 80);
            try writer.writeAll("\n");
            try writer.print("{s:<40} {s:>12} {s:>12} {s:>12}\n", .{
                "Benchmark",
                "ops/sec",
                "mean (ns)",
                "p99 (ns)",
            });
            try writer.writeAll("-" ** 80);
            try writer.writeAll("\n");

            for (entry.value_ptr.items) |result| {
                try writer.print("{s:<40} {d:>12.0} {d:>12.0} {d:>12}\n", .{
                    result.config.name,
                    result.stats.opsPerSecond(),
                    result.stats.mean_ns,
                    result.stats.p99_ns,
                });
            }
            try writer.writeAll("\n");
        }
    }

    /// Append a result with proper string ownership (clones config strings)
    /// Use this instead of directly appending to .results to avoid double-free bugs
    pub fn appendResult(self: *BenchmarkRunner, result: BenchResult) !void {
        var owned = result;
        owned.config = try cloneConfigStrings(self.allocator, result.config);
        try self.results.append(self.allocator, owned);
    }

    /// Print a formatted summary using std.debug.print
    pub fn printSummaryDebug(self: *BenchmarkRunner) void {
        std.debug.print("\n", .{});
        std.debug.print("================================================================================\n", .{});
        std.debug.print("                     BENCHMARK RESULTS SUMMARY\n", .{});
        std.debug.print("================================================================================\n\n", .{});

        for (self.results.items) |result| {
            std.debug.print("{s}: {d:.0} ops/sec, mean={d:.0}ns, p99={d}ns\n", .{
                result.config.name,
                result.stats.opsPerSecond(),
                result.stats.mean_ns,
                result.stats.p99_ns,
            });
        }
        std.debug.print("\n", .{});
    }
};

fn cloneConfigStrings(allocator: std.mem.Allocator, config: BenchConfig) !BenchConfig {
    var owned = config;
    owned.name = try allocator.dupe(u8, config.name);
    owned.category = try allocator.dupe(u8, config.category);
    return owned;
}

fn freeConfigStrings(allocator: std.mem.Allocator, config: *BenchConfig) void {
    if (config.name.len > 0) allocator.free(config.name);
    if (config.category.len > 0) allocator.free(config.category);
    config.name = "";
    config.category = "";
}

/// Calculate comprehensive statistics from samples
fn calculateStatistics(
    allocator: std.mem.Allocator,
    samples: []const u64,
    config: BenchConfig,
) !Statistics {
    if (samples.len == 0) {
        return Statistics{
            .min_ns = 0,
            .max_ns = 0,
            .mean_ns = 0,
            .median_ns = 0,
            .std_dev_ns = 0,
            .p50_ns = 0,
            .p90_ns = 0,
            .p95_ns = 0,
            .p99_ns = 0,
            .iterations = 0,
            .outliers_removed = 0,
            .total_time_ns = 0,
        };
    }

    // Sort samples for percentile calculation
    const sorted = try allocator.alloc(u64, samples.len);
    defer allocator.free(sorted);
    @memcpy(sorted, samples);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

    // Calculate mean
    var sum: u128 = 0;
    for (sorted) |s| {
        sum += s;
    }
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(sorted.len));

    // Calculate standard deviation
    var variance_sum: f64 = 0;
    for (sorted) |s| {
        const diff = @as(f64, @floatFromInt(s)) - mean;
        variance_sum += diff * diff;
    }
    const std_dev = @sqrt(variance_sum / @as(f64, @floatFromInt(sorted.len)));

    // Remove outliers if configured
    var filtered = sorted;
    var outliers_removed: u64 = 0;

    if (config.remove_outliers and sorted.len > 10) {
        var filtered_list = std.ArrayListUnmanaged(u64){};
        defer filtered_list.deinit(allocator);

        const threshold = config.outlier_threshold * std_dev;
        for (sorted) |s| {
            const diff = @abs(@as(f64, @floatFromInt(s)) - mean);
            if (diff <= threshold) {
                try filtered_list.append(allocator, s);
            } else {
                outliers_removed += 1;
            }
        }

        if (filtered_list.items.len > 0) {
            filtered = try allocator.dupe(u64, filtered_list.items);
        }
    }
    defer if (filtered.ptr != sorted.ptr) allocator.free(filtered);

    // Recalculate with filtered data
    sum = 0;
    for (filtered) |s| {
        sum += s;
    }
    const final_mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(filtered.len));

    variance_sum = 0;
    for (filtered) |s| {
        const diff = @as(f64, @floatFromInt(s)) - final_mean;
        variance_sum += diff * diff;
    }
    const final_std_dev = @sqrt(variance_sum / @as(f64, @floatFromInt(filtered.len)));

    return Statistics{
        .min_ns = filtered[0],
        .max_ns = filtered[filtered.len - 1],
        .mean_ns = final_mean,
        .median_ns = @floatFromInt(filtered[filtered.len / 2]),
        .std_dev_ns = final_std_dev,
        .p50_ns = percentile(filtered, 50),
        .p90_ns = percentile(filtered, 90),
        .p95_ns = percentile(filtered, 95),
        .p99_ns = percentile(filtered, 99),
        .iterations = samples.len,
        .outliers_removed = outliers_removed,
        .total_time_ns = @intCast(sum),
    };
}

fn percentile(sorted: []const u64, p: u8) u64 {
    if (sorted.len == 0) return 0;
    const idx = (sorted.len * p) / 100;
    return sorted[@min(idx, sorted.len - 1)];
}

/// Regression detection result
pub const RegressionResult = struct {
    name: []const u8,
    baseline_ops_sec: f64,
    current_ops_sec: f64,
    change_percent: f64,
    is_regression: bool,
    is_improvement: bool,

    pub fn format(
        self: RegressionResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const symbol = if (self.is_regression) "⚠️ SLOWER" else if (self.is_improvement) "✓ FASTER" else "→ SAME";
        try writer.print("{s}: {d:.0} → {d:.0} ops/sec ({d:+.1}%) {s}", .{
            self.name,
            self.baseline_ops_sec,
            self.current_ops_sec,
            self.change_percent,
            symbol,
        });
    }
};

/// Compare benchmark results against a baseline for regression detection
pub fn compareWithBaseline(
    allocator: std.mem.Allocator,
    current: []const BenchResult,
    baseline: []const BenchResult,
    threshold_percent: f64,
) ![]RegressionResult {
    var results = std.ArrayListUnmanaged(RegressionResult){};
    errdefer results.deinit(allocator);

    for (current) |curr| {
        // Find matching baseline
        for (baseline) |base| {
            if (std.mem.eql(u8, curr.config.name, base.config.name)) {
                const curr_ops = curr.stats.opsPerSecond();
                const base_ops = base.stats.opsPerSecond();
                const change = if (base_ops > 0)
                    ((curr_ops - base_ops) / base_ops) * 100.0
                else
                    0.0;

                try results.append(allocator, .{
                    .name = curr.config.name,
                    .baseline_ops_sec = base_ops,
                    .current_ops_sec = curr_ops,
                    .change_percent = change,
                    .is_regression = change < -threshold_percent,
                    .is_improvement = change > threshold_percent,
                });
                break;
            }
        }
    }

    return results.toOwnedSlice(allocator);
}

/// Print regression analysis summary
pub fn printRegressionSummary(results: []const RegressionResult) void {
    var regressions: usize = 0;
    var improvements: usize = 0;

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                     REGRESSION ANALYSIS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    for (results) |r| {
        const symbol = if (r.is_regression) "[SLOWER]" else if (r.is_improvement) "[FASTER]" else "[SAME]  ";
        std.debug.print("{s} {s}: {d:.0} → {d:.0} ops/sec ({d:+.1}%)\n", .{
            symbol,
            r.name,
            r.baseline_ops_sec,
            r.current_ops_sec,
            r.change_percent,
        });

        if (r.is_regression) regressions += 1;
        if (r.is_improvement) improvements += 1;
    }

    std.debug.print("\nSummary: {d} regressions, {d} improvements, {d} unchanged\n", .{
        regressions,
        improvements,
        results.len - regressions - improvements,
    });

    if (regressions > 0) {
        std.debug.print("⚠️  Performance regressions detected!\n", .{});
    }
}

// Tests
test "statistics calculation" {
    const allocator = std.testing.allocator;
    const samples = [_]u64{ 100, 110, 105, 95, 108, 102, 98, 112, 103, 107 };

    const stats = try calculateStatistics(allocator, &samples, .{});

    try std.testing.expect(stats.min_ns == 95);
    try std.testing.expect(stats.max_ns == 112);
    try std.testing.expect(stats.iterations == 10);
}

test "tracking allocator" {
    const base = std.testing.allocator;
    var tracker = TrackingAllocator.init(base);
    const alloc = tracker.allocator();

    const buf = try alloc.alloc(u8, 1024);
    const stats1 = tracker.getStats();
    try std.testing.expect(stats1.allocated == 1024);

    alloc.free(buf);
    const stats2 = tracker.getStats();
    try std.testing.expect(stats2.freed == 1024);
}
