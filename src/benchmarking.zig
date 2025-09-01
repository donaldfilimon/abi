//! Benchmarking Infrastructure
//!
//! This module provides comprehensive benchmarking capabilities including:
//! - Automated benchmark execution
//! - Statistical analysis of results
//! - Performance regression detection
//! - Multi-threaded benchmarking
//! - Integration with performance profiler

const std = @import("std");
const builtin = @import("builtin");
const performance_profiler = @import("performance_profiler.zig");
const memory_tracker = @import("memory_tracker.zig");

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    /// Number of iterations to run
    iterations: usize = 1000,
    /// Warm-up iterations before measurement
    warmup_iterations: usize = 100,
    /// Minimum execution time per benchmark (nanoseconds)
    min_execution_time_ns: u64 = 1_000_000_000, // 1 second
    /// Maximum execution time per benchmark (nanoseconds)
    max_execution_time_ns: u64 = 300_000_000_000, // 5 minutes
    /// Statistical confidence level (0.95 = 95%)
    confidence_level: f64 = 0.95,
    /// Enable performance profiling during benchmarks
    enable_profiling: bool = true,
    /// Enable memory tracking during benchmarks
    enable_memory_tracking: bool = true,
    /// Number of threads for multi-threaded benchmarks
    thread_count: usize = 1,
    /// Enable CPU affinity setting
    enable_cpu_affinity: bool = false,
};

/// Benchmark result
pub const BenchmarkResult = struct {
    /// Benchmark name
    name: []const u8,
    /// Execution time statistics
    timing: TimingStats,
    /// Memory usage statistics
    memory: MemoryStats,
    /// Performance counter results
    counters: std.StringHashMapUnmanaged(PerformanceCounterResult),
    /// Raw execution times for statistical analysis
    raw_times: []u64,
    /// Benchmark metadata
    metadata: BenchmarkMetadata,
    /// Performance profile (if enabled)
    profile: ?[]u8,

    /// Calculate operations per second
    pub fn operationsPerSecond(self: BenchmarkResult) f64 {
        if (self.timing.mean == 0) return 0.0;
        return 1_000_000_000.0 / self.timing.mean;
    }

    /// Get memory usage per operation
    pub fn memoryPerOperation(self: BenchmarkResult) f64 {
        const ops_per_sec = self.operationsPerSecond();
        if (ops_per_sec == 0) return 0.0;
        return @as(f64, @floatFromInt(self.memory.peak_usage)) / ops_per_sec;
    }
};

/// Timing statistics
pub const TimingStats = struct {
    /// Mean execution time (nanoseconds)
    mean: f64 = 0.0,
    /// Median execution time (nanoseconds)
    median: f64 = 0.0,
    /// Standard deviation (nanoseconds)
    std_dev: f64 = 0.0,
    /// Minimum execution time (nanoseconds)
    min: u64 = std.math.maxInt(u64),
    /// Maximum execution time (nanoseconds)
    max: u64 = 0,
    /// 95th percentile (nanoseconds)
    percentile_95: f64 = 0.0,
    /// 99th percentile (nanoseconds)
    percentile_99: f64 = 0.0,
    /// Confidence interval half-width
    confidence_interval: f64 = 0.0,
};

/// Memory statistics for benchmark
pub const MemoryStats = struct {
    /// Peak memory usage (bytes)
    peak_usage: usize = 0,
    /// Average memory usage (bytes)
    average_usage: usize = 0,
    /// Memory allocations during benchmark
    allocations: u64 = 0,
    /// Memory deallocations during benchmark
    deallocations: u64 = 0,
    /// Memory efficiency (1.0 = no waste)
    efficiency: f64 = 0.0,
};

/// Performance counter result
pub const PerformanceCounterResult = struct {
    /// Counter name
    name: []const u8,
    /// Start value
    start_value: u64,
    /// End value
    end_value: u64,
    /// Delta value
    delta: u64,
    /// Rate (delta per second)
    rate_per_second: f64,
};

/// Benchmark metadata
pub const BenchmarkMetadata = struct {
    /// Benchmark start timestamp
    start_time: u64,
    /// Benchmark end timestamp
    end_time: u64,
    /// Total iterations executed
    total_iterations: usize,
    /// Warm-up iterations
    warmup_iterations: usize,
    /// Benchmark configuration used
    config: BenchmarkConfig,
    /// System information
    system_info: SystemInfo,
};

/// System information
pub const SystemInfo = struct {
    /// CPU model
    cpu_model: []const u8,
    /// Number of CPU cores
    cpu_cores: usize,
    /// Total memory (bytes)
    total_memory: usize,
    /// Operating system
    os: []const u8,
    /// Architecture
    arch: []const u8,
    /// Zig version
    zig_version: []const u8,
};

/// Benchmark function type
pub const BenchmarkFn = *const fn (*BenchmarkRunner, std.mem.Allocator) anyerror!void;

/// Benchmark suite
pub const BenchmarkSuite = struct {
    /// Suite name
    name: []const u8,
    /// Benchmarks in this suite
    benchmarks: std.ArrayListUnmanaged(Benchmark),
    /// Suite configuration
    config: BenchmarkConfig,

    /// Initialize benchmark suite
    pub fn init(allocator: std.mem.Allocator, name: []const u8, config: BenchmarkConfig) !*BenchmarkSuite {
        const suite = try allocator.create(BenchmarkSuite);
        errdefer allocator.destroy(suite);

        suite.* = .{
            .name = try allocator.dupe(u8, name),
            .benchmarks = try std.ArrayListUnmanaged(Benchmark).initCapacity(allocator, 16),
            .config = config,
        };

        return suite;
    }

    /// Deinitialize benchmark suite
    pub fn deinit(self: *BenchmarkSuite, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

        for (self.benchmarks.items) |*benchmark| {
            benchmark.deinit(allocator);
        }
        self.benchmarks.deinit(allocator);

        allocator.destroy(self);
    }

    /// Add benchmark to suite
    pub fn addBenchmark(self: *BenchmarkSuite, allocator: std.mem.Allocator, benchmark: Benchmark) !void {
        try self.benchmarks.append(allocator, benchmark);
    }

    /// Run all benchmarks in suite
    pub fn run(self: *BenchmarkSuite, allocator: std.mem.Allocator) ![]BenchmarkResult {
        var results = std.ArrayListUnmanaged(BenchmarkResult){};
        errdefer results.deinit(allocator);

        std.log.info("Running benchmark suite: {s}", .{self.name});
        std.log.info("Benchmarks: {d}", .{self.benchmarks.items.len});

        for (self.benchmarks.items) |benchmark| {
            std.log.info("  Running benchmark: {s}", .{benchmark.name});

            const result = try benchmark.run(allocator, self.config);
            try results.append(allocator, result);

            // Log immediate results
            std.log.info("    Operations/sec: {d:.2}", .{result.operationsPerSecond()});
            std.log.info("    Mean time: {d:.2} ns", .{result.timing.mean});
            std.log.info("    Memory peak: {d} bytes", .{result.memory.peak_usage});
        }

        return try results.toOwnedSlice(allocator);
    }
};

/// Individual benchmark
pub const Benchmark = struct {
    /// Benchmark name
    name: []const u8,
    /// Benchmark function
    function: BenchmarkFn,
    /// Benchmark description
    description: []const u8,
    /// Setup function (optional)
    setup_fn: ?*const fn (std.mem.Allocator) anyerror!void,
    /// Teardown function (optional)
    teardown_fn: ?*const fn (std.mem.Allocator) anyerror!void,

    /// Initialize benchmark
    pub fn init(
        name: []const u8,
        function: BenchmarkFn,
        description: []const u8,
        setup_fn: ?*const fn (std.mem.Allocator) anyerror!void,
        teardown_fn: ?*const fn (std.mem.Allocator) anyerror!void,
    ) Benchmark {
        return .{
            .name = name,
            .function = function,
            .description = description,
            .setup_fn = setup_fn,
            .teardown_fn = teardown_fn,
        };
    }

    /// Deinitialize benchmark
    pub fn deinit(self: *Benchmark, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
        // Name, description, and functions are typically string literals or managed elsewhere
    }

    /// Run benchmark
    pub fn run(self: *Benchmark, allocator: std.mem.Allocator, config: BenchmarkConfig) !BenchmarkResult {
        // Initialize benchmark runner
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();

        // Setup
        if (self.setup_fn) |setup| {
            try setup(allocator);
        }
        defer {
            if (self.teardown_fn) |teardown| {
                teardown(allocator) catch {};
            }
        }

        // Start profiling if enabled
        if (config.enable_profiling) {
            try runner.performance_profiler.?.startSession(self.name);
        }

        const start_time = std.time.nanoTimestamp();

        // Warm-up phase
        for (0..config.warmup_iterations) |_| {
            try self.function(&runner, allocator);
        }

        // Measurement phase
        var raw_times = std.ArrayListUnmanaged(u64){};
        defer raw_times.deinit(allocator);

        var total_iterations: usize = 0;
        var elapsed_time: u64 = 0;

        while (elapsed_time < config.min_execution_time_ns and total_iterations < config.iterations) {
            const iteration_start = std.time.nanoTimestamp();

            try self.function(&runner, allocator);

            const iteration_end = std.time.nanoTimestamp();
            const iteration_time = iteration_end - iteration_start;

            try raw_times.append(allocator, iteration_time);
            elapsed_time += iteration_time;
            total_iterations += 1;

            // Break if we've exceeded maximum time
            if (elapsed_time >= config.max_execution_time_ns) {
                break;
            }
        }

        const end_time = std.time.nanoTimestamp();

        // Get performance profile if enabled
        const profile = if (config.enable_profiling)
            try runner.performance_profiler.?.endSession()
        else
            null;

        // Calculate timing statistics
        const timing_stats = try calculateTimingStats(raw_times.items);

        // Get memory statistics
        const memory_stats = if (config.enable_memory_tracking and runner.memory_profiler != null)
            try getMemoryStats(runner.memory_profiler.?)
        else
            MemoryStats{};

        // Get performance counters
        const counters = if (runner.performance_profiler) |profiler|
            try getPerformanceCounters(allocator, profiler)
        else
            std.StringHashMapUnmanaged(PerformanceCounterResult){};

        // Get system information
        const system_info = try getSystemInfo(allocator);

        const metadata = BenchmarkMetadata{
            .start_time = start_time,
            .end_time = end_time,
            .total_iterations = total_iterations,
            .warmup_iterations = config.warmup_iterations,
            .config = config,
            .system_info = system_info,
        };

        return BenchmarkResult{
            .name = try allocator.dupe(u8, self.name),
            .timing = timing_stats,
            .memory = memory_stats,
            .counters = counters,
            .raw_times = try allocator.dupe(u64, raw_times.items),
            .metadata = metadata,
            .profile = profile,
        };
    }
};

/// Benchmark runner for executing benchmark functions
pub const BenchmarkRunner = struct {
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Performance profiler
    performance_profiler: ?*performance_profiler.PerformanceProfiler,
    /// Memory profiler
    memory_profiler: ?*memory_tracker.MemoryProfiler,
    /// Iteration counter
    iteration_count: usize = 0,

    /// Initialize benchmark runner
    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !*BenchmarkRunner {
        const runner = try allocator.create(BenchmarkRunner);
        errdefer allocator.destroy(runner);

        runner.* = .{
            .config = config,
            .performance_profiler = if (config.enable_profiling)
                try performance_profiler.PerformanceProfiler.init(allocator, performance_profiler.utils.developmentConfig())
            else
                null,
            .memory_profiler = if (config.enable_memory_tracking)
                try memory_tracker.MemoryProfiler.init(allocator, memory_tracker.utils.developmentConfig())
            else
                null,
        };

        // Set up profiler integration
        if (runner.performance_profiler) |perf_profiler| {
            if (runner.memory_profiler) |mem_profiler| {
                perf_profiler.setMemoryTracker(mem_profiler);
            }
        }

        return runner;
    }

    /// Deinitialize benchmark runner
    pub fn deinit(self: *BenchmarkRunner) void {
        if (self.performance_profiler) |profiler| {
            profiler.deinit();
        }
        if (self.memory_profiler) |profiler| {
            profiler.deinit();
        }
    }

    /// Start iteration (called automatically by benchmark framework)
    pub fn startIteration(self: *BenchmarkRunner) void {
        self.iteration_count += 1;

        // Start performance monitoring
        if (self.performance_profiler) |profiler| {
            // Create a scope for this iteration
            _ = profiler.createScope("benchmark_iteration");
        }
    }

    /// End iteration (called automatically by benchmark framework)
    pub fn endIteration() void {
        // Performance monitoring is handled by the scope
    }
};

/// Statistical analysis functions
pub const stats = struct {
    /// Calculate timing statistics from raw data
    pub fn calculateTimingStats(raw_times: []const u64) !TimingStats {
        if (raw_times.len == 0) {
            return TimingStats{};
        }

        // Calculate mean
        var sum: u64 = 0;
        var min_time = raw_times[0];
        var max_time = raw_times[0];

        for (raw_times) |time| {
            sum += time;
            if (time < min_time) min_time = time;
            if (time > max_time) max_time = time;
        }

        const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(raw_times.len));

        // Calculate median
        const sorted_times = try allocator.dupe(u64, raw_times);
        defer allocator.free(sorted_times);

        std.sort.insertion(u64, sorted_times, {}, comptime std.sort.asc(u64));

        const median = if (sorted_times.len % 2 == 0) blk: {
            const mid = sorted_times.len / 2;
            break :blk @as(f64, @floatFromInt(sorted_times[mid - 1] + sorted_times[mid])) / 2.0;
        } else blk: {
            const mid = sorted_times.len / 2;
            break :blk @as(f64, @floatFromInt(sorted_times[mid]));
        };

        // Calculate standard deviation
        var variance_sum: f64 = 0.0;
        for (raw_times) |time| {
            const diff = @as(f64, @floatFromInt(time)) - mean;
            variance_sum += diff * diff;
        }

        const std_dev = std.math.sqrt(variance_sum / @as(f64, @floatFromInt(raw_times.len)));

        // Calculate percentiles
        const percentile_95 = try calculatePercentile(sorted_times, 0.95);
        const percentile_99 = try calculatePercentile(sorted_times, 0.99);

        // Calculate confidence interval (simplified)
        const confidence_interval = std_dev * 1.96 / std.math.sqrt(@as(f64, @floatFromInt(raw_times.len))); // 95% CI

        return TimingStats{
            .mean = mean,
            .median = median,
            .std_dev = std_dev,
            .min = min_time,
            .max = max_time,
            .percentile_95 = percentile_95,
            .percentile_99 = percentile_99,
            .confidence_interval = confidence_interval,
        };
    }

    /// Calculate percentile from sorted data
    pub fn calculatePercentile(sorted_data: []const u64, percentile: f64) !f64 {
        if (sorted_data.len == 0) return 0.0;
        if (percentile < 0.0 or percentile > 1.0) return error.InvalidPercentile;

        const index = percentile * @as(f64, @floatFromInt(sorted_data.len - 1));
        const lower = @as(usize, @intFromFloat(std.math.floor(index)));
        const upper = @as(usize, @intFromFloat(std.math.ceil(index)));

        if (lower == upper) {
            return @as(f64, @floatFromInt(sorted_data[lower]));
        }

        const weight = index - @as(f64, @floatFromInt(lower));
        const lower_val = @as(f64, @floatFromInt(sorted_data[lower]));
        const upper_val = @as(f64, @floatFromInt(sorted_data[upper]));

        return lower_val * (1.0 - weight) + upper_val * weight;
    }
};

/// Utility functions
pub const utils = struct {
    /// Create a standard benchmark configuration
    pub fn standardConfig() BenchmarkConfig {
        return .{
            .iterations = 1000,
            .warmup_iterations = 100,
            .min_execution_time_ns = 1_000_000_000, // 1 second
            .max_execution_time_ns = 30_000_000_000, // 30 seconds
            .confidence_level = 0.95,
            .enable_profiling = true,
            .enable_memory_tracking = true,
            .thread_count = 1,
            .enable_cpu_affinity = false,
        };
    }

    /// Create a quick benchmark configuration
    pub fn quickConfig() BenchmarkConfig {
        return .{
            .iterations = 100,
            .warmup_iterations = 10,
            .min_execution_time_ns = 100_000_000, // 100ms
            .max_execution_time_ns = 1_000_000_000, // 1 second
            .confidence_level = 0.90,
            .enable_profiling = false,
            .enable_memory_tracking = false,
            .thread_count = 1,
            .enable_cpu_affinity = false,
        };
    }

    /// Create a thorough benchmark configuration
    pub fn thoroughConfig() BenchmarkConfig {
        return .{
            .iterations = 10000,
            .warmup_iterations = 1000,
            .min_execution_time_ns = 10_000_000_000, // 10 seconds
            .max_execution_time_ns = 300_000_000_000, // 5 minutes
            .confidence_level = 0.99,
            .enable_profiling = true,
            .enable_memory_tracking = true,
            .thread_count = std.Thread.getCpuCount() catch 4,
            .enable_cpu_affinity = true,
        };
    }

    /// Generate benchmark report
    pub fn generateReport(allocator: std.mem.Allocator, results: []const BenchmarkResult) ![]u8 {
        var report = std.ArrayListUnmanaged(u8){};
        errdefer report.deinit(allocator);

        const writer = report.writer(allocator);

        try writer.print("=== Benchmark Report ===\n", .{});
        try writer.print("Generated: {d}\n", .{std.time.nanoTimestamp()});
        try writer.print("Total benchmarks: {d}\n\n", .{results.len});

        for (results, 0..) |result, i| {
            try writer.print("Benchmark {d}: {s}\n", .{ i + 1, result.name });
            try writer.print("  Iterations: {d}\n", .{result.metadata.total_iterations});
            try writer.print("  Mean time: {d:.2} ns\n", .{result.timing.mean});
            try writer.print("  Median time: {d:.2} ns\n", .{result.timing.median});
            try writer.print("  Std dev: {d:.2} ns\n", .{result.timing.std_dev});
            try writer.print("  Min time: {d} ns\n", .{result.timing.min});
            try writer.print("  Max time: {d} ns\n", .{result.timing.max});
            try writer.print("  95th percentile: {d:.2} ns\n", .{result.timing.percentile_95});
            try writer.print("  Operations/sec: {d:.2}\n", .{result.operationsPerSecond()});
            try writer.print("  Memory peak: {d} bytes\n", .{result.memory.peak_usage});
            try writer.print("  Memory per op: {d:.2} bytes\n", .{result.memoryPerOperation()});

            if (result.counters.count() > 0) {
                try writer.print("  Performance counters:\n", .{});
                var counter_iter = result.counters.iterator();
                while (counter_iter.next()) |entry| {
                    try writer.print("    {s}: {d} ({d:.2}/sec)\n", .{
                        entry.value_ptr.name,
                        entry.value_ptr.delta,
                        entry.value_ptr.rate_per_second,
                    });
                }
            }

            try writer.print("\n", .{});
        }

        // System information
        if (results.len > 0) {
            try writer.print("System Information:\n", .{});
            try writer.print("  CPU: {s}\n", .{results[0].metadata.system_info.cpu_model});
            try writer.print("  Cores: {d}\n", .{results[0].metadata.system_info.cpu_cores});
            try writer.print("  Memory: {d} GB\n", .{results[0].metadata.system_info.total_memory / (1024 * 1024 * 1024)});
            try writer.print("  OS: {s}\n", .{results[0].metadata.system_info.os});
            try writer.print("  Architecture: {s}\n", .{results[0].metadata.system_info.arch});
            try writer.print("  Zig Version: {s}\n", .{results[0].metadata.system_info.zig_version});
        }

        try writer.print("\n=== End Report ===\n", .{});

        return try report.toOwnedSlice(allocator);
    }

    /// Compare benchmark results for regression detection
    pub fn compareResults(baseline: BenchmarkResult, current: BenchmarkResult, threshold_percent: f64) BenchmarkComparison {
        const mean_diff_percent = ((current.timing.mean - baseline.timing.mean) / baseline.timing.mean) * 100.0;
        const ops_per_sec_diff_percent = ((current.operationsPerSecond() - baseline.operationsPerSecond()) / baseline.operationsPerSecond()) * 100.0;

        const has_regression = mean_diff_percent > threshold_percent;
        const has_improvement = mean_diff_percent < -threshold_percent;

        return .{
            .mean_time_difference_percent = mean_diff_percent,
            .ops_per_sec_difference_percent = ops_per_sec_diff_percent,
            .has_regression = has_regression,
            .has_improvement = has_improvement,
            .baseline_result = baseline,
            .current_result = current,
            .threshold_percent = threshold_percent,
        };
    }
};

/// Benchmark comparison result
pub const BenchmarkComparison = struct {
    /// Mean time difference (percentage)
    mean_time_difference_percent: f64,
    /// Operations per second difference (percentage)
    ops_per_sec_difference_percent: f64,
    /// Whether this is a regression
    has_regression: bool,
    /// Whether this is an improvement
    has_improvement: bool,
    /// Baseline result
    baseline_result: BenchmarkResult,
    /// Current result
    current_result: BenchmarkResult,
    /// Threshold percentage used
    threshold_percent: f64,
};

/// Helper functions
fn calculateTimingStats(raw_times: []const u64) !TimingStats {
    return stats.calculateTimingStats(raw_times);
}

fn getMemoryStats(profiler: *memory_tracker.MemoryProfiler) !MemoryStats {
    const stats_data = profiler.getStats();
    return MemoryStats{
        .peak_usage = stats_data.peak_usage,
        .average_usage = stats_data.currentUsage(),
        .allocations = stats_data.total_allocation_count,
        .deallocations = stats_data.total_deallocation_count,
        .efficiency = stats_data.efficiency(),
    };
}

fn getPerformanceCounters(profiler: *performance_profiler.PerformanceProfiler) !std.StringHashMapUnmanaged(PerformanceCounterResult) {
    const counters = std.StringHashMapUnmanaged(PerformanceCounterResult){};

    // Get counter values from profiler
    // (This would need to be implemented based on the actual profiler API)
    _ = profiler;

    return counters;
}

fn getSystemInfo(allocator: std.mem.Allocator) !SystemInfo {
    return SystemInfo{
        .cpu_model = try allocator.dupe(u8, "Unknown CPU"),
        .cpu_cores = std.Thread.getCpuCount() catch 1,
        .total_memory = 8 * 1024 * 1024 * 1024, // 8GB default
        .os = try allocator.dupe(u8, @tagName(builtin.os.tag)),
        .arch = try allocator.dupe(u8, @tagName(builtin.cpu.arch)),
        .zig_version = try allocator.dupe(u8, "0.16.0-dev"),
    };
}
