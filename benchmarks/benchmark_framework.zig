//! Standardized Benchmark Framework for ABI
//!
//! This framework provides consistent benchmarking methodologies across all test suites:
//! - Standardized timing and measurement
//! - Statistical analysis with confidence intervals
//! - Memory usage tracking
//! - Performance regression detection
//! - Export capabilities for CI/CD integration

const std = @import("std");
const builtin = @import("builtin");
const utils = @import("abi").utils;

const separator_line = "================================================================================";
const subsection_line = "------------------------------------------------------------";

/// Benchmark configuration with standardized settings
pub const BenchmarkConfig = struct {
    /// Number of warmup iterations to perform before measurement
    warmup_iterations: u32 = 100,

    /// Number of measurement iterations
    measurement_iterations: u32 = 1000,

    /// Number of samples to collect for statistical analysis
    samples: u32 = 10,

    /// Minimum measurement time in nanoseconds
    min_measurement_time_ns: u64 = 1_000_000, // 1ms

    /// Maximum measurement time in nanoseconds
    max_measurement_time_ns: u64 = 30_000_000_000, // 30s

    /// Enable memory usage tracking
    enable_memory_tracking: bool = true,

    /// Enable detailed statistics
    enable_detailed_stats: bool = true,

    /// Output format for results
    output_format: OutputFormat = .console,

    /// Export results to file
    export_results: bool = false,

    /// Export file path
    export_path: []const u8 = "benchmark_results.json",

    pub const OutputFormat = enum {
        console,
        json,
        csv,
        markdown,
    };
};

/// Statistical analysis of benchmark results
pub const BenchmarkStats = struct {
    mean_ns: f64,
    median_ns: f64,
    min_ns: u64,
    max_ns: u64,
    std_deviation_ns: f64,
    variance_ns: f64,
    confidence_interval_95: struct { lower: f64, upper: f64 },
    throughput_ops_per_sec: f64,
    memory_usage_bytes: usize,
    memory_peak_bytes: usize,

    pub fn format(self: BenchmarkStats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("Mean: {d:.2}ns, Median: {d:.2}ns, StdDev: {d:.2}ns\n" ++
            "Range: {d}ns - {d}ns, Throughput: {d:.0} ops/sec\n" ++
            "Memory: {} bytes (peak: {} bytes)\n" ++
            "95% CI: [{d:.2}ns, {d:.2}ns]", .{ self.mean_ns, self.median_ns, self.std_deviation_ns, self.min_ns, self.max_ns, self.throughput_ops_per_sec, self.memory_usage_bytes, self.memory_peak_bytes, self.confidence_interval_95.lower, self.confidence_interval_95.upper });
    }
};

/// Individual benchmark result
pub const BenchmarkResult = struct {
    name: []const u8,
    category: []const u8,
    stats: BenchmarkStats,
    success: bool,
    error_message: ?[]const u8,
    timestamp: i64,
    platform_info: PlatformInfo,

    pub fn deinit(self: *BenchmarkResult, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.category);
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }

    pub fn toJson(self: BenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        var json = std.ArrayList(u8){};
        try json.ensureUnusedCapacity(allocator, 1024);
        defer json.deinit(allocator);

        try json.appendSlice(allocator, "{\n");

        // Format all fields directly
        const json_content = try std.fmt.allocPrint(allocator,
            \\  "name": "{s}",
            \\  "category": "{s}",
            \\  "success": {},
            \\  "timestamp": {},
            \\  "mean_time_ns": {d:.2},
            \\  "throughput_ops_per_sec": {d:.0},
            \\  "memory_usage_bytes": {},
            \\  "std_deviation_ns": {d:.2},
            \\  "confidence_interval_95": {{
            \\    "lower": {d:.2},
            \\    "upper": {d:.2}
            \\  }},
            \\  "platform": {{
            \\    "os": "{s}",
            \\    "arch": "{s}",
            \\    "zig_version": "{s}"
            \\  }}
            \\
        , .{
            self.name,
            self.category,
            self.success,
            self.timestamp,
            self.stats.mean_ns,
            self.stats.throughput_ops_per_sec,
            self.stats.memory_usage_bytes,
            self.stats.std_deviation_ns,
            self.stats.confidence_interval_95.lower,
            self.stats.confidence_interval_95.upper,
            self.platform_info.os,
            self.platform_info.arch,
            self.platform_info.zig_version,
        });
        defer allocator.free(json_content);
        try json.appendSlice(allocator, json_content);

        try json.appendSlice(allocator, "}");

        return try json.toOwnedSlice(allocator);
    }
};

/// Platform information for benchmarking context
pub const PlatformInfo = struct {
    os: []const u8,
    arch: []const u8,
    zig_version: []const u8,
    cpu_count: u32,

    pub fn init() PlatformInfo {
        return .{
            .os = @tagName(builtin.target.os.tag),
            .arch = @tagName(builtin.cpu.arch),
            .zig_version = @import("builtin").zig_version_string,
            .cpu_count = @as(u32, @intCast(std.Thread.getCpuCount() catch 1)),
        };
    }
};

/// Main benchmark suite coordinator
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: std.ArrayList(BenchmarkResult),
    platform_info: PlatformInfo,

    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !*BenchmarkSuite {
        const self = try allocator.create(BenchmarkSuite);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .results = std.ArrayList(BenchmarkResult){},
            .platform_info = PlatformInfo.init(),
        };
        return self;
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        for (self.results.items) |*result| {
            result.deinit(self.allocator);
        }
        self.results.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Run a benchmark with automatic timing and statistical analysis
    pub fn runBenchmark(
        self: *BenchmarkSuite,
        name: []const u8,
        category: []const u8,
        benchmark_fn: anytype,
        context: anytype,
    ) !void {
        std.log.info("Running benchmark: {s} ({s})", .{ name, category });

        var memory_before: usize = 0;
        var memory_after: usize = 0;
        var memory_peak: usize = 0;

        if (self.config.enable_memory_tracking) {
            memory_before = getMemoryUsage();
        }

        // Warmup phase
        for (0..self.config.warmup_iterations) |_| {
            _ = benchmark_fn(context) catch |err| {
                std.log.warn("Benchmark warmup failed: {}", .{err});
            };
        }

        // Force garbage collection/compaction if supported by allocator
        // Note: rawResize was removed in newer Zig versions

        // Measurement phase
        var measurements = try self.allocator.alloc(u64, self.config.samples);
        defer self.allocator.free(measurements);

        for (0..self.config.samples) |sample_idx| {
            var total_time: u64 = 0;
            var iterations: u32 = 0;

            while (total_time < self.config.min_measurement_time_ns and
                iterations < self.config.measurement_iterations)
            {
                const start_time = std.time.nanoTimestamp;
                _ = benchmark_fn(context) catch |err| {
                    const result = BenchmarkResult{
                        .name = name,
                        .category = category,
                        .stats = undefined,
                        .success = false,
                        .error_message = try std.fmt.allocPrint(self.allocator, "Benchmark failed: {}", .{err}),
                        .timestamp = 0,
                        .platform_info = self.platform_info,
                    };
                    try self.results.append(self.allocator, result);
                    return;
                };
                const end_time = std.time.nanoTimestamp;

                total_time += @as(u64, @intCast(end_time - start_time));
                iterations += 1;

                if (total_time >= self.config.max_measurement_time_ns) break;
            }

            const avg_time_per_iteration = if (iterations > 0) total_time / iterations else 0;
            measurements[sample_idx] = avg_time_per_iteration;

            if (self.config.enable_memory_tracking) {
                const current_memory = getMemoryUsage();
                memory_peak = @max(memory_peak, current_memory);
            }
        }

        if (self.config.enable_memory_tracking) {
            memory_after = getMemoryUsage();
        }

        // Calculate statistics
        const stats = try calculateStats(
            self.allocator,
            measurements,
            memory_after - memory_before,
            memory_peak,
        );

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const category_copy = try self.allocator.dupe(u8, category);
        errdefer self.allocator.free(category_copy);

        const result = BenchmarkResult{
            .name = name_copy,
            .category = category_copy,
            .stats = stats,
            .success = true,
            .error_message = null,
            .timestamp = 0,
            .platform_info = self.platform_info,
        };

        try self.results.append(self.allocator, result);

        // Log result
        std.log.info("✅ {s}: {d:.0} ops/sec ({d:.2}ns avg)", .{ name, stats.throughput_ops_per_sec, stats.mean_ns });
    }

    pub fn runBenchmarkFmt(
        self: *BenchmarkSuite,
        comptime fmt: []const u8,
        args: anytype,
        category: []const u8,
        benchmark_fn: anytype,
        context: anytype,
    ) !void {
        const formatted_name = try std.fmt.allocPrint(self.allocator, fmt, args);
        defer self.allocator.free(formatted_name);
        try self.runBenchmark(formatted_name, category, benchmark_fn, context);
    }

    /// Print comprehensive results report
    pub fn printReport(self: *BenchmarkSuite) !void {
        switch (self.config.output_format) {
            .console => try self.printConsoleReport(),
            .json => try self.printJsonReport(),
            .csv => try self.printCsvReport(),
            .markdown => try self.printMarkdownReport(),
        }
    }

    fn printConsoleReport(self: *BenchmarkSuite) !void {
        std.log.info("\n{s}", .{separator_line});
        std.log.info("🚀 BENCHMARK RESULTS REPORT", .{});
        std.log.info("{s}", .{separator_line});
        std.log.info("Platform: {s} {s} (Zig {s})", .{ self.platform_info.os, self.platform_info.arch, self.platform_info.zig_version });
        std.log.info("CPU Cores: {}", .{self.platform_info.cpu_count});
        std.log.info("Total Benchmarks: {}", .{self.results.items.len});
        std.log.info("{s}", .{separator_line});

        // Group by category
        var categories = std.StringHashMap(std.ArrayList(*BenchmarkResult)).init(self.allocator);
        defer {
            var it = categories.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit(self.allocator);
            }
            categories.deinit();
        }

        for (self.results.items) |*result| {
            const entry = try categories.getOrPut(result.category);
            if (!entry.found_existing) {
                entry.value_ptr.* = std.ArrayList(*BenchmarkResult){};
            }
            try entry.value_ptr.append(self.allocator, result);
        }

        var it = categories.iterator();
        while (it.next()) |entry| {
            std.log.info("\n📊 Category: {s}", .{entry.key_ptr.*});
            std.log.info("{s}", .{subsection_line});

            for (entry.value_ptr.items) |result| {
                std.log.info("  {s:<30} {d:>8.0} ops/sec  {d:>8.2}ns avg", .{ result.name, result.stats.throughput_ops_per_sec, result.stats.mean_ns });

                if (self.config.enable_detailed_stats) {
                    std.log.info("    └─ {any}", .{result.stats});
                }
            }
        }

        // Summary statistics
        try self.printSummaryStats();
    }

    fn printSummaryStats(self: *BenchmarkSuite) !void {
        if (self.results.items.len == 0) return;

        var total_throughput: f64 = 0;
        var fastest_benchmark: ?*BenchmarkResult = null;
        var slowest_benchmark: ?*BenchmarkResult = null;
        var fastest_throughput: f64 = 0;
        var slowest_throughput: f64 = std.math.inf(f64);

        for (self.results.items) |*result| {
            if (!result.success) continue;

            total_throughput += result.stats.throughput_ops_per_sec;

            if (result.stats.throughput_ops_per_sec > fastest_throughput) {
                fastest_throughput = result.stats.throughput_ops_per_sec;
                fastest_benchmark = result;
            }

            if (result.stats.throughput_ops_per_sec < slowest_throughput) {
                slowest_throughput = result.stats.throughput_ops_per_sec;
                slowest_benchmark = result;
            }
        }

        std.log.info("\n🏆 PERFORMANCE SUMMARY", .{});
        std.log.info("{s}", .{subsection_line});
        std.log.info("Average Throughput: {d:.0} ops/sec", .{total_throughput / @as(f64, @floatFromInt(self.results.items.len))});

        if (fastest_benchmark) |fastest| {
            std.log.info("Fastest Benchmark: {s} ({d:.0} ops/sec)", .{ fastest.name, fastest.stats.throughput_ops_per_sec });
        }

        if (slowest_benchmark) |slowest| {
            std.log.info("Slowest Benchmark: {s} ({d:.0} ops/sec)", .{ slowest.name, slowest.stats.throughput_ops_per_sec });
        }

        std.log.info("Performance Ratio: {d:.1}x", .{if (slowest_throughput > 0) fastest_throughput / slowest_throughput else 0});
    }

    fn printJsonReport(self: *BenchmarkSuite) !void {
        var json = std.ArrayList(u8){};
        try json.ensureUnusedCapacity(self.allocator, 4096);
        defer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{\n");

        // Format platform info directly into the array
        const platform_json = try std.fmt.allocPrint(self.allocator, "  \"platform\": {{\n    \"os\": \"{s}\",\n    \"arch\": \"{s}\",\n    \"zig_version\": \"{s}\",\n    \"cpu_count\": {}\n  }},\n", .{
            self.platform_info.os,
            self.platform_info.arch,
            self.platform_info.zig_version,
            self.platform_info.cpu_count,
        });
        defer self.allocator.free(platform_json);
        try json.appendSlice(self.allocator, platform_json);

        try json.appendSlice(self.allocator, "  \"benchmarks\": [\n");

        for (self.results.items, 0..) |result, i| {
            const result_json = try result.toJson(self.allocator);
            defer self.allocator.free(result_json);

            try json.appendSlice(self.allocator, result_json);
            if (i < self.results.items.len - 1) {
                try json.appendSlice(self.allocator, ",\n");
            }
        }

        try json.appendSlice(self.allocator, "\n  ]\n");
        try json.appendSlice(self.allocator, "}\n");

        if (self.config.export_results) {
            try std.fs.cwd().writeFile(.{ .sub_path = self.config.export_path, .data = json.items });
        } else {
            std.log.info("{s}", .{json.items});
        }
    }

    fn printCsvReport(self: *BenchmarkSuite) !void {
        var csv = std.ArrayList(u8){};
        try csv.ensureUnusedCapacity(self.allocator, 2048);
        defer csv.deinit(self.allocator);

        // Header
        try csv.appendSlice(self.allocator, "name,category,mean_time_ns,throughput_ops_per_sec,memory_usage_bytes,std_deviation_ns,success\n");

        // Data rows
        for (self.results.items) |result| {
            const csv_row = try std.fmt.allocPrint(self.allocator, "{s},{s},{d:.2},{d:.0},{},{d:.2},{}\n", .{ result.name, result.category, result.stats.mean_ns, result.stats.throughput_ops_per_sec, result.stats.memory_usage_bytes, result.stats.std_deviation_ns, result.success });
            defer self.allocator.free(csv_row);
            try csv.appendSlice(self.allocator, csv_row);
        }

        if (self.config.export_results) {
            try std.fs.cwd().writeFile(.{ .sub_path = self.config.export_path, .data = csv.items });
        } else {
            std.log.info("{s}", .{csv.items});
        }
    }

    fn printMarkdownReport(self: *BenchmarkSuite) !void {
        var md = std.ArrayList(u8){};
        try md.ensureUnusedCapacity(self.allocator, 4096);
        defer md.deinit(self.allocator);

        try md.appendSlice(self.allocator, "# Benchmark Results\n\n");

        const platform_md = try std.fmt.allocPrint(self.allocator, "**Platform:** {s} {s} (Zig {s})\n**CPU Cores:** {}\n", .{ self.platform_info.os, self.platform_info.arch, self.platform_info.zig_version, self.platform_info.cpu_count });
        defer self.allocator.free(platform_md);
        try md.appendSlice(self.allocator, platform_md);

        try md.appendSlice(self.allocator, "\n## Results\n\n");
        try md.appendSlice(self.allocator, "| Name | Category | Throughput (ops/sec) | Mean Time (ns) | Memory (bytes) |\n");
        try md.appendSlice(self.allocator, "|------|----------|---------------------|----------------|----------------|\n");

        for (self.results.items) |result| {
            const row_md = try std.fmt.allocPrint(self.allocator, "| {s} | {s} | {d:.0} | {d:.2} | {} |\n", .{ result.name, result.category, result.stats.throughput_ops_per_sec, result.stats.mean_ns, result.stats.memory_usage_bytes });
            defer self.allocator.free(row_md);
            try md.appendSlice(self.allocator, row_md);
        }

        if (self.config.export_results) {
            try std.fs.cwd().writeFile(.{ .sub_path = self.config.export_path, .data = md.items });
        } else {
            std.log.info("{s}", .{md.items});
        }
    }

    /// Export results for CI/CD integration
    pub fn exportForCI(self: *BenchmarkSuite, format: BenchmarkConfig.OutputFormat, path: []const u8) !void {
        const original_format = self.config.output_format;
        const original_export = self.config.export_results;
        const original_path = self.config.export_path;

        self.config.output_format = format;
        self.config.export_results = true;
        self.config.export_path = path;

        try self.printReport();

        // Restore original settings
        self.config.output_format = original_format;
        self.config.export_results = original_export;
        self.config.export_path = original_path;
    }
};

/// Calculate statistical measures from measurement data
fn calculateStats(
    allocator: std.mem.Allocator,
    measurements: []const u64,
    memory_usage: usize,
    memory_peak: usize,
) !BenchmarkStats {
    if (measurements.len == 0) {
        return BenchmarkStats{
            .mean_ns = 0,
            .median_ns = 0,
            .min_ns = 0,
            .max_ns = 0,
            .std_deviation_ns = 0,
            .variance_ns = 0,
            .confidence_interval_95 = .{ .lower = 0, .upper = 0 },
            .throughput_ops_per_sec = 0,
            .memory_usage_bytes = memory_usage,
            .memory_peak_bytes = memory_peak,
        };
    }

    // Calculate mean
    var sum: u64 = 0;
    for (measurements) |measurement| {
        sum += measurement;
    }
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(measurements.len));

    // Calculate min/max
    var min: u64 = std.math.maxInt(u64);
    var max: u64 = 0;
    for (measurements) |measurement| {
        min = @min(min, measurement);
        max = @max(max, measurement);
    }

    // Calculate variance and standard deviation
    var variance_sum: f64 = 0;
    for (measurements) |measurement| {
        const diff = @as(f64, @floatFromInt(measurement)) - mean;
        variance_sum += diff * diff;
    }
    const variance = variance_sum / @as(f64, @floatFromInt(measurements.len));
    const std_dev = std.math.sqrt(variance);

    // Calculate median
    const sorted_measurements = try allocator.alloc(u64, measurements.len);
    defer allocator.free(sorted_measurements);
    @memcpy(sorted_measurements, measurements);
    std.mem.sort(u64, sorted_measurements, {}, std.sort.asc(u64));

    const median = if (sorted_measurements.len % 2 == 0) blk: {
        const mid = sorted_measurements.len / 2;
        break :blk @as(f64, @floatFromInt(sorted_measurements[mid - 1] + sorted_measurements[mid])) / 2.0;
    } else @as(f64, @floatFromInt(sorted_measurements[sorted_measurements.len / 2]));

    // Calculate 95% confidence interval (t-distribution approximation)
    const t_value_95 = 1.96; // Approximation for large samples
    const margin_of_error = t_value_95 * (std_dev / std.math.sqrt(@as(f64, @floatFromInt(measurements.len))));

    // Calculate throughput
    const throughput = if (mean > 0) 1_000_000_000.0 / mean else 0;

    return BenchmarkStats{
        .mean_ns = mean,
        .median_ns = median,
        .min_ns = min,
        .max_ns = max,
        .std_deviation_ns = std_dev,
        .variance_ns = variance,
        .confidence_interval_95 = .{
            .lower = mean - margin_of_error,
            .upper = mean + margin_of_error,
        },
        .throughput_ops_per_sec = throughput,
        .memory_usage_bytes = memory_usage,
        .memory_peak_bytes = memory_peak,
    };
}

/// Get current memory usage (platform-specific implementation)
fn getMemoryUsage() usize {
    // This is a simplified implementation
    // In a real implementation, you would use platform-specific APIs
    // like GetProcessMemoryInfo on Windows or /proc/self/status on Linux
    return 0;
}

/// Benchmark utilities for common operations
pub const BenchmarkUtils = struct {
    /// Create test data for vector operations
    pub fn createTestVectors(allocator: std.mem.Allocator, size: usize) !struct { a: []f32, b: []f32, result: []f32 } {
        const a = try allocator.alloc(f32, size);
        const b = try allocator.alloc(f32, size);
        const result = try allocator.alloc(f32, size);

        // Initialize with test data
        for (a, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }

        for (b, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i * 7) % 100)) * 0.02;
        }

        return .{ .a = a, .b = b, .result = result };
    }

    /// Create test text data
    pub fn createTestText(allocator: std.mem.Allocator, size: usize) ![]u8 {
        const text = try allocator.alloc(u8, size);
        for (text, 0..) |*byte, i| {
            byte.* = @as(u8, @intCast((i % 26) + 'a'));
        }
        return text;
    }

    /// Create test matrix data
    pub fn createTestMatrices(allocator: std.mem.Allocator, rows: usize, cols: usize) !struct { a: []f32, b: []f32, result: []f32 } {
        const a = try allocator.alloc(f32, rows * cols);
        const b = try allocator.alloc(f32, cols * rows);
        const result = try allocator.alloc(f32, rows * rows);

        // Initialize matrices
        for (a, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }

        for (b, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i * 7) % 100)) * 0.02;
        }

        return .{ .a = a, .b = b, .result = result };
    }
};
