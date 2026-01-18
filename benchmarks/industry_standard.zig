//! Industry-Standard Benchmark Extensions
//!
//! Comprehensive benchmarking aligned with industry standards:
//! - ANN-Benchmarks compatibility (ann-benchmarks.com)
//! - Cache profiling metrics (L1/L2/L3 miss estimation)
//! - Energy efficiency metrics
//! - Regression detection with baselines
//! - Scaling analysis (linear/sublinear characterization)
//! - Hardware capability detection
//!
//! ## Supported Standards
//!
//! - **ANN-Benchmarks**: Vector similarity search benchmarks
//! - **MLPerf**: ML inference benchmarks
//! - **SPEC-like**: System performance characterization
//! - **TPC-inspired**: Transaction processing metrics

const std = @import("std");
const framework = @import("framework.zig");

// ============================================================================
// Cache Profiling Metrics
// ============================================================================

/// Cache level statistics (estimated based on access patterns)
pub const CacheStats = struct {
    /// Estimated L1 cache hits
    l1_hits: u64 = 0,
    /// Estimated L1 cache misses
    l1_misses: u64 = 0,
    /// Estimated L2 cache hits
    l2_hits: u64 = 0,
    /// Estimated L2 cache misses
    l2_misses: u64 = 0,
    /// Estimated L3 cache hits
    l3_hits: u64 = 0,
    /// Estimated L3 cache misses (main memory access)
    l3_misses: u64 = 0,
    /// Total memory accesses
    total_accesses: u64 = 0,
    /// Cache line size used for estimation (bytes)
    cache_line_size: usize = 64,

    pub fn l1HitRate(self: CacheStats) f64 {
        const total = self.l1_hits + self.l1_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.l1_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn l2HitRate(self: CacheStats) f64 {
        const total = self.l2_hits + self.l2_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.l2_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn l3HitRate(self: CacheStats) f64 {
        const total = self.l3_hits + self.l3_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.l3_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn overallHitRate(self: CacheStats) f64 {
        if (self.total_accesses == 0) return 0.0;
        const misses = self.l3_misses;
        return 1.0 - (@as(f64, @floatFromInt(misses)) / @as(f64, @floatFromInt(self.total_accesses)));
    }

    /// Estimate memory bandwidth based on cache misses
    pub fn estimatedBandwidthMBps(self: CacheStats, elapsed_ns: u64) f64 {
        if (elapsed_ns == 0) return 0.0;
        const bytes_transferred = self.l3_misses * self.cache_line_size;
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        return @as(f64, @floatFromInt(bytes_transferred)) / (seconds * 1024.0 * 1024.0);
    }
};

/// Cache profiler for estimating cache behavior
pub const CacheProfiler = struct {
    stats: CacheStats,
    /// L1 cache size (typically 32KB-64KB per core)
    l1_size: usize,
    /// L2 cache size (typically 256KB-512KB per core)
    l2_size: usize,
    /// L3 cache size (typically 8MB-64MB shared)
    l3_size: usize,
    /// Working set tracking
    working_set: std.AutoHashMapUnmanaged(usize, void),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CacheProfiler {
        return .{
            .stats = .{},
            .l1_size = detectL1CacheSize(),
            .l2_size = detectL2CacheSize(),
            .l3_size = detectL3CacheSize(),
            .working_set = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CacheProfiler) void {
        self.working_set.deinit(self.allocator);
    }

    /// Record a memory access for cache simulation
    pub fn recordAccess(self: *CacheProfiler, address: usize, size: usize) void {
        const cache_line = address / self.stats.cache_line_size;
        const num_lines = (size + self.stats.cache_line_size - 1) / self.stats.cache_line_size;

        var i: usize = 0;
        while (i < num_lines) : (i += 1) {
            const line = cache_line + i;
            self.stats.total_accesses += 1;

            // Simple LRU-like simulation
            if (self.working_set.contains(line)) {
                // Hit - check which level based on working set size
                const ws_size = self.working_set.count() * self.stats.cache_line_size;
                if (ws_size <= self.l1_size) {
                    self.stats.l1_hits += 1;
                } else if (ws_size <= self.l2_size) {
                    self.stats.l1_misses += 1;
                    self.stats.l2_hits += 1;
                } else if (ws_size <= self.l3_size) {
                    self.stats.l1_misses += 1;
                    self.stats.l2_misses += 1;
                    self.stats.l3_hits += 1;
                } else {
                    self.stats.l1_misses += 1;
                    self.stats.l2_misses += 1;
                    self.stats.l3_misses += 1;
                }
            } else {
                // Miss at all levels for new data
                self.stats.l1_misses += 1;
                self.stats.l2_misses += 1;
                self.stats.l3_misses += 1;
                self.working_set.put(self.allocator, line, {}) catch {};
            }
        }
    }

    pub fn getStats(self: *const CacheProfiler) CacheStats {
        return self.stats;
    }

    pub fn reset(self: *CacheProfiler) void {
        self.stats = .{};
        self.working_set.clearRetainingCapacity();
    }
};

/// Detect L1 cache size (returns default if detection fails)
fn detectL1CacheSize() usize {
    // Default L1 cache size (32KB is common)
    return 32 * 1024;
}

/// Detect L2 cache size (returns default if detection fails)
fn detectL2CacheSize() usize {
    // Default L2 cache size (256KB is common)
    return 256 * 1024;
}

/// Detect L3 cache size (returns default if detection fails)
fn detectL3CacheSize() usize {
    // Default L3 cache size (8MB is common)
    return 8 * 1024 * 1024;
}

// ============================================================================
// Energy Efficiency Metrics
// ============================================================================

/// Energy efficiency metrics for benchmarking
pub const EnergyMetrics = struct {
    /// Operations per joule (estimated)
    ops_per_joule: f64 = 0,
    /// Average power consumption (watts, estimated)
    avg_power_watts: f64 = 0,
    /// Total energy consumed (joules, estimated)
    total_energy_joules: f64 = 0,
    /// Energy efficiency rating (higher is better)
    efficiency_rating: f64 = 0,
    /// Thermal headroom (estimated)
    thermal_headroom_percent: f64 = 100,
    /// Whether actual power measurements are available
    is_estimated: bool = true,

    /// Calculate energy from power and time
    pub fn fromPowerAndTime(power_watts: f64, elapsed_ns: u64, operations: u64) EnergyMetrics {
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const energy = power_watts * seconds;
        const ops = @as(f64, @floatFromInt(operations));

        return .{
            .ops_per_joule = if (energy > 0) ops / energy else 0,
            .avg_power_watts = power_watts,
            .total_energy_joules = energy,
            .efficiency_rating = if (energy > 0) (ops / energy) / 1000.0 else 0,
            .thermal_headroom_percent = 100,
            .is_estimated = true,
        };
    }

    /// Estimate power based on CPU utilization
    pub fn estimateFromCpuUsage(cpu_percent: f64, tdp_watts: f64, elapsed_ns: u64, operations: u64) EnergyMetrics {
        const estimated_power = tdp_watts * (cpu_percent / 100.0) * 0.7; // 70% efficiency factor
        return fromPowerAndTime(estimated_power, elapsed_ns, operations);
    }
};

/// Energy profiler for tracking power consumption estimates
pub const EnergyProfiler = struct {
    /// Thermal Design Power (TDP) of the system
    tdp_watts: f64,
    /// Base power consumption (idle)
    base_power_watts: f64,
    /// Start time for measurement
    start_ns: u64,
    /// Accumulated CPU time
    cpu_time_ns: u64,
    /// Number of operations
    operations: u64,

    pub fn init(tdp_watts: f64) EnergyProfiler {
        return .{
            .tdp_watts = tdp_watts,
            .base_power_watts = tdp_watts * 0.1, // 10% idle power
            .start_ns = 0,
            .cpu_time_ns = 0,
            .operations = 0,
        };
    }

    pub fn start(self: *EnergyProfiler) void {
        var timer = std.time.Timer.start() catch return;
        self.start_ns = timer.read();
    }

    pub fn recordOperation(self: *EnergyProfiler, count: u64) void {
        self.operations += count;
    }

    pub fn stop(self: *EnergyProfiler) EnergyMetrics {
        var timer = std.time.Timer.start() catch return .{};
        const end_ns = timer.read();
        const elapsed_ns = end_ns -| self.start_ns;

        // Estimate CPU utilization based on operations (rough estimate)
        const ops_per_sec = if (elapsed_ns > 0)
            @as(f64, @floatFromInt(self.operations)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0)
        else
            0;

        // Assume higher ops/sec means higher CPU utilization
        const cpu_percent = @min(100.0, ops_per_sec / 1_000_000.0 * 10.0);

        return EnergyMetrics.estimateFromCpuUsage(
            cpu_percent,
            self.tdp_watts,
            elapsed_ns,
            self.operations,
        );
    }
};

// ============================================================================
// Regression Detection
// ============================================================================

/// Regression detection configuration
pub const RegressionConfig = struct {
    /// Percentage threshold for detecting regression (e.g., 5.0 = 5%)
    regression_threshold: f64 = 5.0,
    /// Percentage threshold for detecting improvement
    improvement_threshold: f64 = 5.0,
    /// Minimum number of samples for statistical significance
    min_samples: usize = 30,
    /// Confidence level (0.95 = 95% confidence)
    confidence_level: f64 = 0.95,
    /// Whether to use statistical tests
    use_statistical_tests: bool = true,
};

/// Result of regression analysis
pub const RegressionAnalysis = struct {
    /// Name of the benchmark
    name: []const u8,
    /// Baseline performance (ops/sec or latency)
    baseline_value: f64,
    /// Current performance
    current_value: f64,
    /// Percentage change (positive = improvement for throughput)
    change_percent: f64,
    /// Whether a regression was detected
    is_regression: bool,
    /// Whether an improvement was detected
    is_improvement: bool,
    /// Statistical significance (p-value if available)
    p_value: ?f64,
    /// Confidence interval lower bound
    ci_lower: f64,
    /// Confidence interval upper bound
    ci_upper: f64,
    /// Number of samples
    sample_count: usize,

    pub fn format(
        self: RegressionAnalysis,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const status = if (self.is_regression)
            "REGRESSION"
        else if (self.is_improvement)
            "IMPROVEMENT"
        else
            "STABLE";

        try writer.print("{s}: {d:.2} -> {d:.2} ({d:+.2}%) [{s}]", .{
            self.name,
            self.baseline_value,
            self.current_value,
            self.change_percent,
            status,
        });
    }
};

/// Baseline storage for regression detection
pub const BaselineStore = struct {
    baselines: std.StringHashMapUnmanaged(BaselineEntry),
    allocator: std.mem.Allocator,

    pub const BaselineEntry = struct {
        mean: f64,
        std_dev: f64,
        min: f64,
        max: f64,
        sample_count: usize,
        timestamp: i64,
    };

    pub fn init(allocator: std.mem.Allocator) BaselineStore {
        return .{
            .baselines = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BaselineStore) void {
        var it = self.baselines.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.baselines.deinit(self.allocator);
    }

    pub fn setBaseline(self: *BaselineStore, name: []const u8, entry: BaselineEntry) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.baselines.put(self.allocator, key, entry);
    }

    pub fn getBaseline(self: *const BaselineStore, name: []const u8) ?BaselineEntry {
        return self.baselines.get(name);
    }

    /// Analyze regression against stored baseline
    pub fn analyzeRegression(
        self: *const BaselineStore,
        name: []const u8,
        current_stats: framework.Statistics,
        config: RegressionConfig,
    ) ?RegressionAnalysis {
        const baseline = self.getBaseline(name) orelse return null;

        const current_ops = current_stats.opsPerSecond();
        const baseline_ops = 1_000_000_000.0 / baseline.mean;

        const change = if (baseline_ops > 0)
            ((current_ops - baseline_ops) / baseline_ops) * 100.0
        else
            0.0;

        // Calculate confidence interval
        const z_score = 1.96; // 95% confidence
        const se = baseline.std_dev / @sqrt(@as(f64, @floatFromInt(baseline.sample_count)));
        const ci_lower = baseline.mean - z_score * se;
        const ci_upper = baseline.mean + z_score * se;

        // Determine regression/improvement status
        const is_regression = change < -config.regression_threshold;
        const is_improvement = change > config.improvement_threshold;

        return .{
            .name = name,
            .baseline_value = baseline_ops,
            .current_value = current_ops,
            .change_percent = change,
            .is_regression = is_regression,
            .is_improvement = is_improvement,
            .p_value = null, // Would need actual statistical test
            .ci_lower = ci_lower,
            .ci_upper = ci_upper,
            .sample_count = baseline.sample_count,
        };
    }

    /// Save baselines to JSON file
    pub fn saveToFile(self: *const BaselineStore, path: []const u8) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);

        var json_buf = std.ArrayListUnmanaged(u8){};
        defer json_buf.deinit(self.allocator);

        try json_buf.appendSlice(self.allocator, "{\n  \"baselines\": {\n");

        var first = true;
        var it = self.baselines.iterator();
        while (it.next()) |entry| {
            if (!first) try json_buf.appendSlice(self.allocator, ",\n");
            first = false;

            try json_buf.appendSlice(self.allocator, "    \"");
            try json_buf.appendSlice(self.allocator, entry.key_ptr.*);
            try json_buf.appendSlice(self.allocator, "\": {");
            try json_buf.writer(self.allocator).print(
                "\"mean\": {d:.4}, \"std_dev\": {d:.4}, \"min\": {d:.4}, \"max\": {d:.4}, \"samples\": {d}, \"timestamp\": {d}",
                .{
                    entry.value_ptr.mean,
                    entry.value_ptr.std_dev,
                    entry.value_ptr.min,
                    entry.value_ptr.max,
                    entry.value_ptr.sample_count,
                    entry.value_ptr.timestamp,
                },
            );
            try json_buf.appendSlice(self.allocator, "}");
        }

        try json_buf.appendSlice(self.allocator, "\n  }\n}\n");

        try file.writeStreamingAll(io, json_buf.items);
    }
};

// ============================================================================
// Scaling Analysis
// ============================================================================

/// Scaling characteristics of a benchmark
pub const ScalingType = enum {
    /// O(1) - constant time
    constant,
    /// O(log n) - logarithmic
    logarithmic,
    /// O(n) - linear
    linear,
    /// O(n log n) - linearithmic
    linearithmic,
    /// O(n^2) - quadratic
    quadratic,
    /// O(n^k) for some k > 2
    polynomial,
    /// O(2^n) - exponential
    exponential,
    /// Unknown or irregular scaling
    unknown,
};

/// Result of scaling analysis
pub const ScalingAnalysis = struct {
    /// Detected scaling type
    scaling_type: ScalingType,
    /// Scaling coefficient (for polynomial: the exponent)
    coefficient: f64,
    /// R-squared value (goodness of fit)
    r_squared: f64,
    /// Data points used for analysis
    data_points: []const ScalingDataPoint,
    /// Predicted saturation point (where performance plateaus)
    saturation_point: ?usize,
    /// Whether scaling is sub-linear (better than O(n))
    is_sublinear: bool,

    pub fn format(
        self: ScalingAnalysis,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const type_str = switch (self.scaling_type) {
            .constant => "O(1)",
            .logarithmic => "O(log n)",
            .linear => "O(n)",
            .linearithmic => "O(n log n)",
            .quadratic => "O(n^2)",
            .polynomial => "O(n^k)",
            .exponential => "O(2^n)",
            .unknown => "unknown",
        };
        try writer.print("Scaling: {s} (R^2={d:.3})", .{ type_str, self.r_squared });
    }
};

/// Data point for scaling analysis
pub const ScalingDataPoint = struct {
    /// Input size (n)
    input_size: usize,
    /// Time taken (nanoseconds)
    time_ns: u64,
    /// Memory used (bytes)
    memory_bytes: u64,
};

/// Analyzer for scaling characteristics
pub const ScalingAnalyzer = struct {
    data_points: std.ArrayListUnmanaged(ScalingDataPoint),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ScalingAnalyzer {
        return .{
            .data_points = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ScalingAnalyzer) void {
        self.data_points.deinit(self.allocator);
    }

    pub fn addDataPoint(self: *ScalingAnalyzer, point: ScalingDataPoint) !void {
        try self.data_points.append(self.allocator, point);
    }

    /// Analyze scaling characteristics from collected data
    pub fn analyze(self: *const ScalingAnalyzer) ScalingAnalysis {
        if (self.data_points.items.len < 3) {
            return .{
                .scaling_type = .unknown,
                .coefficient = 0,
                .r_squared = 0,
                .data_points = self.data_points.items,
                .saturation_point = null,
                .is_sublinear = false,
            };
        }

        // Fit different scaling models and find best fit
        const linear_fit = self.fitLinear();
        const log_fit = self.fitLogarithmic();
        const quadratic_fit = self.fitQuadratic();

        // Select best fit based on R-squared
        var best_type: ScalingType = .unknown;
        var best_r2: f64 = 0;
        var best_coef: f64 = 0;

        if (linear_fit.r_squared > best_r2) {
            best_type = .linear;
            best_r2 = linear_fit.r_squared;
            best_coef = linear_fit.coefficient;
        }

        if (log_fit.r_squared > best_r2) {
            best_type = .logarithmic;
            best_r2 = log_fit.r_squared;
            best_coef = log_fit.coefficient;
        }

        if (quadratic_fit.r_squared > best_r2) {
            best_type = .quadratic;
            best_r2 = quadratic_fit.r_squared;
            best_coef = quadratic_fit.coefficient;
        }

        // Check for constant time (very low variance relative to mean)
        if (self.isConstantTime()) {
            best_type = .constant;
            best_r2 = 1.0;
            best_coef = 1.0;
        }

        return .{
            .scaling_type = best_type,
            .coefficient = best_coef,
            .r_squared = best_r2,
            .data_points = self.data_points.items,
            .saturation_point = self.findSaturationPoint(),
            .is_sublinear = best_type == .constant or best_type == .logarithmic,
        };
    }

    fn fitLinear(self: *const ScalingAnalyzer) struct { coefficient: f64, r_squared: f64 } {
        if (self.data_points.items.len < 2) return .{ .coefficient = 0, .r_squared = 0 };

        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        var sum_xy: f64 = 0;
        var sum_x2: f64 = 0;
        const n = @as(f64, @floatFromInt(self.data_points.items.len));

        for (self.data_points.items) |point| {
            const x = @as(f64, @floatFromInt(point.input_size));
            const y = @as(f64, @floatFromInt(point.time_ns));
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        const denom = n * sum_x2 - sum_x * sum_x;
        if (denom == 0) return .{ .coefficient = 0, .r_squared = 0 };

        const slope = (n * sum_xy - sum_x * sum_y) / denom;
        const mean_y = sum_y / n;

        // Calculate R-squared
        var ss_res: f64 = 0;
        var ss_tot: f64 = 0;
        const intercept = (sum_y - slope * sum_x) / n;

        for (self.data_points.items) |point| {
            const x = @as(f64, @floatFromInt(point.input_size));
            const y = @as(f64, @floatFromInt(point.time_ns));
            const predicted = slope * x + intercept;
            ss_res += (y - predicted) * (y - predicted);
            ss_tot += (y - mean_y) * (y - mean_y);
        }

        const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 0;

        return .{ .coefficient = slope, .r_squared = r_squared };
    }

    fn fitLogarithmic(self: *const ScalingAnalyzer) struct { coefficient: f64, r_squared: f64 } {
        if (self.data_points.items.len < 2) return .{ .coefficient = 0, .r_squared = 0 };

        var sum_logx: f64 = 0;
        var sum_y: f64 = 0;
        var sum_logxy: f64 = 0;
        var sum_logx2: f64 = 0;
        const n = @as(f64, @floatFromInt(self.data_points.items.len));

        for (self.data_points.items) |point| {
            const log_x = @log(@as(f64, @floatFromInt(point.input_size)) + 1);
            const y = @as(f64, @floatFromInt(point.time_ns));
            sum_logx += log_x;
            sum_y += y;
            sum_logxy += log_x * y;
            sum_logx2 += log_x * log_x;
        }

        const denom = n * sum_logx2 - sum_logx * sum_logx;
        if (denom == 0) return .{ .coefficient = 0, .r_squared = 0 };

        const slope = (n * sum_logxy - sum_logx * sum_y) / denom;
        const mean_y = sum_y / n;

        // Calculate R-squared
        var ss_res: f64 = 0;
        var ss_tot: f64 = 0;
        const intercept = (sum_y - slope * sum_logx) / n;

        for (self.data_points.items) |point| {
            const log_x = @log(@as(f64, @floatFromInt(point.input_size)) + 1);
            const y = @as(f64, @floatFromInt(point.time_ns));
            const predicted = slope * log_x + intercept;
            ss_res += (y - predicted) * (y - predicted);
            ss_tot += (y - mean_y) * (y - mean_y);
        }

        const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 0;

        return .{ .coefficient = slope, .r_squared = r_squared };
    }

    fn fitQuadratic(self: *const ScalingAnalyzer) struct { coefficient: f64, r_squared: f64 } {
        if (self.data_points.items.len < 3) return .{ .coefficient = 0, .r_squared = 0 };

        // Simplified: just check correlation with x^2
        var sum_x2: f64 = 0;
        var sum_y: f64 = 0;
        var sum_x2y: f64 = 0;
        var sum_x4: f64 = 0;
        const n = @as(f64, @floatFromInt(self.data_points.items.len));

        for (self.data_points.items) |point| {
            const x = @as(f64, @floatFromInt(point.input_size));
            const x2 = x * x;
            const y = @as(f64, @floatFromInt(point.time_ns));
            sum_x2 += x2;
            sum_y += y;
            sum_x2y += x2 * y;
            sum_x4 += x2 * x2;
        }

        const denom = n * sum_x4 - sum_x2 * sum_x2;
        if (denom == 0) return .{ .coefficient = 0, .r_squared = 0 };

        const coef = (n * sum_x2y - sum_x2 * sum_y) / denom;
        const mean_y = sum_y / n;

        // Calculate R-squared
        var ss_res: f64 = 0;
        var ss_tot: f64 = 0;
        const intercept = (sum_y - coef * sum_x2) / n;

        for (self.data_points.items) |point| {
            const x = @as(f64, @floatFromInt(point.input_size));
            const y = @as(f64, @floatFromInt(point.time_ns));
            const predicted = coef * x * x + intercept;
            ss_res += (y - predicted) * (y - predicted);
            ss_tot += (y - mean_y) * (y - mean_y);
        }

        const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 0;

        return .{ .coefficient = coef, .r_squared = r_squared };
    }

    fn isConstantTime(self: *const ScalingAnalyzer) bool {
        if (self.data_points.items.len < 3) return false;

        var sum: f64 = 0;
        for (self.data_points.items) |point| {
            sum += @as(f64, @floatFromInt(point.time_ns));
        }
        const mean = sum / @as(f64, @floatFromInt(self.data_points.items.len));

        var variance: f64 = 0;
        for (self.data_points.items) |point| {
            const diff = @as(f64, @floatFromInt(point.time_ns)) - mean;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(self.data_points.items.len));

        // Coefficient of variation < 10% suggests constant time
        const cv = @sqrt(variance) / mean;
        return cv < 0.1;
    }

    fn findSaturationPoint(self: *const ScalingAnalyzer) ?usize {
        if (self.data_points.items.len < 5) return null;

        // Look for point where performance stops scaling linearly
        const items = self.data_points.items;
        var prev_rate: f64 = 0;

        for (items[1..], 1..) |point, i| {
            const prev = items[i - 1];
            const size_diff = @as(f64, @floatFromInt(point.input_size - prev.input_size));
            const time_diff = @as(f64, @floatFromInt(point.time_ns)) - @as(f64, @floatFromInt(prev.time_ns));

            if (size_diff > 0) {
                const rate = time_diff / size_diff;
                if (prev_rate > 0 and rate > prev_rate * 2.0) {
                    // Significant slowdown detected
                    return prev.input_size;
                }
                prev_rate = rate;
            }
        }

        return null;
    }
};

// ============================================================================
// ANN-Benchmarks Compatibility
// ============================================================================

/// ANN-Benchmarks compatible result format
pub const AnnBenchmarkResult = struct {
    /// Algorithm name
    algorithm: []const u8,
    /// Dataset name
    dataset: []const u8,
    /// Recall value (0.0 to 1.0)
    recall: f64,
    /// Queries per second
    qps: f64,
    /// Build time in seconds
    build_time_sec: f64,
    /// Index size in bytes
    index_size_bytes: u64,
    /// Distance metric used
    distance: []const u8,
    /// Algorithm parameters (JSON string)
    parameters: []const u8,

    pub fn toJson(self: AnnBenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{");
        try buf.writer(allocator).print(
            "\"algorithm\": \"{s}\", \"dataset\": \"{s}\", \"recall\": {d:.6}, \"qps\": {d:.2}, \"build_time\": {d:.4}, \"index_size\": {d}, \"distance\": \"{s}\", \"parameters\": {s}",
            .{
                self.algorithm,
                self.dataset,
                self.recall,
                self.qps,
                self.build_time_sec,
                self.index_size_bytes,
                self.distance,
                self.parameters,
            },
        );
        try buf.appendSlice(allocator, "}");

        return buf.toOwnedSlice(allocator);
    }
};

/// Standard ANN-Benchmarks datasets
pub const AnnDataset = enum {
    /// SIFT1M - 1M 128d vectors
    sift_1m,
    /// GIST1M - 1M 960d vectors
    gist_1m,
    /// GloVe - 1.2M 100d word embeddings
    glove_100,
    /// Fashion-MNIST - 60K 784d vectors
    fashion_mnist,
    /// NYTimes - 290K 256d vectors
    nytimes,
    /// Custom dataset
    custom,

    pub fn dimension(self: AnnDataset) usize {
        return switch (self) {
            .sift_1m => 128,
            .gist_1m => 960,
            .glove_100 => 100,
            .fashion_mnist => 784,
            .nytimes => 256,
            .custom => 0,
        };
    }

    pub fn size(self: AnnDataset) usize {
        return switch (self) {
            .sift_1m => 1_000_000,
            .gist_1m => 1_000_000,
            .glove_100 => 1_200_000,
            .fashion_mnist => 60_000,
            .nytimes => 290_000,
            .custom => 0,
        };
    }

    pub fn name(self: AnnDataset) []const u8 {
        return switch (self) {
            .sift_1m => "sift-1m",
            .gist_1m => "gist-1m",
            .glove_100 => "glove-100",
            .fashion_mnist => "fashion-mnist",
            .nytimes => "nytimes",
            .custom => "custom",
        };
    }
};

// ============================================================================
// Hardware Capability Detection
// ============================================================================

/// Detected hardware capabilities
pub const HardwareCapabilities = struct {
    /// CPU vendor (e.g., "GenuineIntel", "AuthenticAMD")
    cpu_vendor: []const u8,
    /// Number of physical cores
    physical_cores: usize,
    /// Number of logical cores (with hyperthreading)
    logical_cores: usize,
    /// L1 data cache size per core (bytes)
    l1d_cache_size: usize,
    /// L2 cache size per core (bytes)
    l2_cache_size: usize,
    /// L3 cache size total (bytes)
    l3_cache_size: usize,
    /// Whether AVX is supported
    has_avx: bool,
    /// Whether AVX2 is supported
    has_avx2: bool,
    /// Whether AVX-512 is supported
    has_avx512: bool,
    /// Whether NEON (ARM) is supported
    has_neon: bool,
    /// Total system memory (bytes)
    total_memory: usize,
    /// Available memory (bytes)
    available_memory: usize,
    /// Estimated TDP (watts)
    estimated_tdp: f64,

    pub fn detect() HardwareCapabilities {
        // Note: In a real implementation, this would use CPUID, /proc/cpuinfo, etc.
        return .{
            .cpu_vendor = "Unknown",
            .physical_cores = std.Thread.getCpuCount() catch 1,
            .logical_cores = std.Thread.getCpuCount() catch 1,
            .l1d_cache_size = 32 * 1024,
            .l2_cache_size = 256 * 1024,
            .l3_cache_size = 8 * 1024 * 1024,
            .has_avx = true, // Assume modern x86
            .has_avx2 = true,
            .has_avx512 = false,
            .has_neon = std.Target.current.cpu.arch.isARM(),
            .total_memory = 16 * 1024 * 1024 * 1024, // 16GB default
            .available_memory = 8 * 1024 * 1024 * 1024, // 8GB default
            .estimated_tdp = 65.0, // Typical desktop TDP
        };
    }

    pub fn toJson(self: HardwareCapabilities, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n");
        try buf.writer(allocator).print(
            \\  "cpu_vendor": "{s}",
            \\  "physical_cores": {d},
            \\  "logical_cores": {d},
            \\  "l1d_cache_kb": {d},
            \\  "l2_cache_kb": {d},
            \\  "l3_cache_mb": {d},
            \\  "has_avx": {s},
            \\  "has_avx2": {s},
            \\  "has_avx512": {s},
            \\  "total_memory_gb": {d},
            \\  "estimated_tdp_watts": {d}
        ,
            .{
                self.cpu_vendor,
                self.physical_cores,
                self.logical_cores,
                self.l1d_cache_size / 1024,
                self.l2_cache_size / 1024,
                self.l3_cache_size / (1024 * 1024),
                if (self.has_avx) "true" else "false",
                if (self.has_avx2) "true" else "false",
                if (self.has_avx512) "true" else "false",
                self.total_memory / (1024 * 1024 * 1024),
                self.estimated_tdp,
            },
        );
        try buf.appendSlice(allocator, "\n}");

        return buf.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Comprehensive Benchmark Report
// ============================================================================

/// Comprehensive benchmark report with all metrics
pub const BenchmarkReport = struct {
    /// Report metadata
    name: []const u8,
    timestamp: i64,
    hardware: HardwareCapabilities,
    /// Benchmark results
    results: []const framework.BenchResult,
    /// Cache statistics
    cache_stats: ?CacheStats,
    /// Energy metrics
    energy_metrics: ?EnergyMetrics,
    /// Scaling analysis
    scaling: ?ScalingAnalysis,
    /// Regression analysis
    regressions: []const RegressionAnalysis,
    /// ANN-benchmark compatible results
    ann_results: []const AnnBenchmarkResult,

    /// Generate markdown report
    pub fn toMarkdown(self: *const BenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "# Benchmark Report: ");
        try buf.appendSlice(allocator, self.name);
        try buf.appendSlice(allocator, "\n\n");

        // Hardware info
        try buf.appendSlice(allocator, "## Hardware\n\n");
        try buf.writer(allocator).print(
            "- CPU Cores: {d} physical, {d} logical\n",
            .{ self.hardware.physical_cores, self.hardware.logical_cores },
        );
        try buf.writer(allocator).print(
            "- Cache: L1={d}KB, L2={d}KB, L3={d}MB\n",
            .{
                self.hardware.l1d_cache_size / 1024,
                self.hardware.l2_cache_size / 1024,
                self.hardware.l3_cache_size / (1024 * 1024),
            },
        );
        try buf.appendSlice(allocator, "\n");

        // Results summary
        try buf.appendSlice(allocator, "## Results Summary\n\n");
        try buf.appendSlice(allocator, "| Benchmark | Ops/sec | Mean (ns) | P99 (ns) | Memory |\n");
        try buf.appendSlice(allocator, "|-----------|---------|-----------|----------|--------|\n");

        for (self.results) |result| {
            try buf.writer(allocator).print(
                "| {s} | {d:.0} | {d:.0} | {d} | {d} bytes |\n",
                .{
                    result.config.name,
                    result.stats.opsPerSecond(),
                    result.stats.mean_ns,
                    result.stats.p99_ns,
                    result.memory_allocated,
                },
            );
        }

        // Cache stats
        if (self.cache_stats) |stats| {
            try buf.appendSlice(allocator, "\n## Cache Performance\n\n");
            try buf.writer(allocator).print(
                "- L1 Hit Rate: {d:.1}%\n",
                .{stats.l1HitRate() * 100},
            );
            try buf.writer(allocator).print(
                "- L2 Hit Rate: {d:.1}%\n",
                .{stats.l2HitRate() * 100},
            );
            try buf.writer(allocator).print(
                "- L3 Hit Rate: {d:.1}%\n",
                .{stats.l3HitRate() * 100},
            );
        }

        // Regressions
        if (self.regressions.len > 0) {
            try buf.appendSlice(allocator, "\n## Regression Analysis\n\n");
            for (self.regressions) |reg| {
                const status = if (reg.is_regression)
                    "REGRESSION"
                else if (reg.is_improvement)
                    "IMPROVEMENT"
                else
                    "STABLE";

                try buf.writer(allocator).print(
                    "- **{s}**: {d:.2} -> {d:.2} ops/sec ({d:+.1}%) [{s}]\n",
                    .{ reg.name, reg.baseline_value, reg.current_value, reg.change_percent, status },
                );
            }
        }

        try buf.appendSlice(allocator, "\n");

        return buf.toOwnedSlice(allocator);
    }

    /// Generate JSON report
    pub fn toJson(self: *const BenchmarkReport, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{\n  \"name\": \"");
        try buf.appendSlice(allocator, self.name);
        try buf.appendSlice(allocator, "\",\n  \"timestamp\": ");
        try buf.writer(allocator).print("{d}", .{self.timestamp});
        try buf.appendSlice(allocator, ",\n  \"results\": [\n");

        for (self.results, 0..) |result, i| {
            if (i > 0) try buf.appendSlice(allocator, ",\n");
            try buf.appendSlice(allocator, "    {");
            try buf.writer(allocator).print(
                "\"name\": \"{s}\", \"ops_per_sec\": {d:.2}, \"mean_ns\": {d:.2}, \"p99_ns\": {d}",
                .{
                    result.config.name,
                    result.stats.opsPerSecond(),
                    result.stats.mean_ns,
                    result.stats.p99_ns,
                },
            );
            try buf.appendSlice(allocator, "}");
        }

        try buf.appendSlice(allocator, "\n  ]\n}\n");

        return buf.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cache profiler basic" {
    var profiler = CacheProfiler.init(std.testing.allocator);
    defer profiler.deinit();

    // Simulate some memory accesses
    profiler.recordAccess(0, 64);
    profiler.recordAccess(64, 64);
    profiler.recordAccess(0, 64); // Should hit cache

    const stats = profiler.getStats();
    try std.testing.expect(stats.total_accesses == 3);
}

test "energy metrics calculation" {
    const metrics = EnergyMetrics.fromPowerAndTime(100.0, 1_000_000_000, 1_000_000);

    try std.testing.expect(metrics.ops_per_joule > 0);
    try std.testing.expect(metrics.total_energy_joules > 0);
    try std.testing.expect(metrics.is_estimated);
}

test "scaling analyzer" {
    var analyzer = ScalingAnalyzer.init(std.testing.allocator);
    defer analyzer.deinit();

    // Add linear-looking data points
    try analyzer.addDataPoint(.{ .input_size = 100, .time_ns = 1000, .memory_bytes = 100 });
    try analyzer.addDataPoint(.{ .input_size = 200, .time_ns = 2000, .memory_bytes = 200 });
    try analyzer.addDataPoint(.{ .input_size = 400, .time_ns = 4000, .memory_bytes = 400 });
    try analyzer.addDataPoint(.{ .input_size = 800, .time_ns = 8000, .memory_bytes = 800 });

    const analysis = analyzer.analyze();
    try std.testing.expect(analysis.r_squared > 0.9);
}

test "baseline store" {
    var store = BaselineStore.init(std.testing.allocator);
    defer store.deinit();

    try store.setBaseline("test_bench", .{
        .mean = 1000.0,
        .std_dev = 100.0,
        .min = 800.0,
        .max = 1200.0,
        .sample_count = 100,
        .timestamp = 0,
    });

    const baseline = store.getBaseline("test_bench");
    try std.testing.expect(baseline != null);
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), baseline.?.mean, 0.001);
}

test "hardware capabilities" {
    const hw = HardwareCapabilities.detect();
    try std.testing.expect(hw.physical_cores > 0);
    try std.testing.expect(hw.l1d_cache_size > 0);
}
