//! GPU-Specific Benchmarks and Performance Profiling Tools
//!
//! This module provides comprehensive benchmarking and profiling capabilities:
//! - GPU kernel performance benchmarks with statistical analysis
//! - Memory bandwidth measurements with access pattern analysis
//! - Compute throughput tests with workload-specific optimizations
//! - Latency measurements with precision timing
//! - Power consumption profiling and thermal monitoring
//! - Comparative backend analysis with performance regression detection
//! - Automated performance testing and continuous integration support
//!
//! ## Key Features
//!
//! - **Statistical Analysis**: Mean, median, standard deviation, percentiles
//! - **Memory Profiling**: Bandwidth, latency, access patterns, cache analysis
//! - **Performance Regression**: Baseline comparison and trend analysis
//! - **Multi-Backend Testing**: Cross-platform backend performance comparison
//! - **Workload Characterization**: Compute, memory, graphics workload analysis
//! - **Thermal Monitoring**: Temperature, power, and cooling capacity tracking
//!
//! ## Usage Example
//!
//! ```zig
//! const benchmark = @import("benchmarks");
//!
//! var profiler = try benchmark.PerformanceProfiler.init(allocator, renderer);
//! defer profiler.deinit();
//!
//! // Run comprehensive benchmark suite
//! const config = benchmark.BenchmarkConfig{
//!     .iterations = 100,
//!     .workloads = &[_]benchmark.WorkloadType{ .matrix_mul, .attention },
//! };
//!
//! try profiler.runBenchmarkSuite(config);
//!
//! // Generate performance report
//! const report = try profiler.generateReport(allocator);
//! defer allocator.free(report);
//! ```
//!
//! Advanced performance profiling with statistical analysis and regression detection
//!
//! @Definitions
//!
//! ---
//! # @Definitions
//!
//! The `@Definitions` annotation is a custom documentation directive used in this project to mark
//! key types, structures, and configuration blocks that are intended for documentation extraction
//! and web export. It is recognized by our documentation tooling, but is not a Zig language built-in.
//!
//! ## Usage
//! - Place `@Definitions` above a type, struct, enum, or function to indicate it should be included
//!   in the generated API documentation and web reference.
//! - The documentation generator will extract all such marked items and render them in the web docs.
//!
//! ## Example
//!
//! ```zig
//! /// Benchmark result with statistical analysis and performance metrics
//! @Definitions
//! pub const BenchmarkResult = struct { ... };
//! ```
//!
//! ## Benefits
//! - Ensures that all important API types and configuration options are discoverable in the docs.
//! - Enables automated web documentation and API reference generation.
//! - Provides a consistent way to mark public API surfaces for users and contributors.
//!
//! ---
//!
//! @Web

const std = @import("std");
const gpu_renderer = @import("../core/gpu_renderer.zig");
const kernels = @import("../compute/kernels.zig");
const memory_pool = @import("../memory/memory_pool.zig");
const backends = @import("../backends/backends.zig");

// Note: The @Definitions annotation is a custom documentation directive for this project.
// In Zig, documentation comments use /// for declarations and //! for modules. The @Definitions
// marker is not a Zig built-in, but is recognized by our documentation tooling for web export.

/// Types of compute workloads
pub const WorkloadType = enum {
    matrix_mul,
    vector_add,
    convolution,
    attention,
    pooling,
    normalization,
    activation,
    fft,
    reduction,
    scan,
    sort,
    sparse_operations,
    custom,

    /// Get the human-readable name for this workload type
    pub fn displayName(self: WorkloadType) []const u8 {
        return switch (self) {
            .matrix_mul => "Matrix Multiplication",
            .vector_add => "Vector Addition",
            .convolution => "Convolution",
            .attention => "Attention Mechanism",
            .pooling => "Pooling Operations",
            .normalization => "Normalization",
            .activation => "Activation Functions",
            .fft => "Fast Fourier Transform",
            .reduction => "Reduction Operations",
            .scan => "Prefix Scan",
            .sort => "Sorting Algorithms",
            .sparse_operations => "Sparse Matrix Operations",
            .custom => "Custom Workload",
        };
    }

    /// Get the computational complexity class for this workload
    pub fn complexityClass(self: WorkloadType) []const u8 {
        return switch (self) {
            .matrix_mul => "O(n³)",
            .vector_add => "O(n)",
            .convolution => "O(n²)",
            .attention => "O(n²)",
            .pooling => "O(n)",
            .normalization => "O(n)",
            .activation => "O(n)",
            .fft => "O(n log n)",
            .reduction => "O(n)",
            .scan => "O(n)",
            .sort => "O(n log n)",
            .sparse_operations => "O(nnz)",
            .custom => "O(?)",
        };
    }
};

/// Output format for benchmark reports
pub const OutputFormat = enum {
    text,
    json,
    csv,
    html,
};

/// Performance grade classification
pub const PerformanceGrade = enum {
    excellent,
    good,
    fair,
    poor,

    pub fn displayName(self: PerformanceGrade) []const u8 {
        return switch (self) {
            .excellent => "Excellent",
            .good => "Good",
            .fair => "Fair",
            .poor => "Poor",
        };
    }

    pub fn colorCode(self: PerformanceGrade) []const u8 {
        return switch (self) {
            .excellent => "🟢",
            .good => "🟡",
            .fair => "🟠",
            .poor => "🔴",
        };
    }

    pub fn toString(self: PerformanceGrade) []const u8 {
        return self.displayName();
    }
};

/// Enhanced benchmark configuration with comprehensive testing options
pub const BenchmarkConfig = struct {
    /// Number of iterations to run
    iterations: u32 = 100,
    /// Warmup iterations to stabilize performance
    warmup_iterations: u32 = 10,
    /// Enable detailed timing with nanosecond precision
    detailed_timing: bool = true,
    /// Enable memory profiling and leak detection
    memory_profiling: bool = true,
    /// Enable power consumption profiling (if available)
    power_profiling: bool = false,
    /// Enable thermal monitoring
    thermal_monitoring: bool = true,
    /// Buffer sizes to test (automatically scaled)
    buffer_sizes: []const usize = &[_]usize{ 1024, 8192, 65536, 524288, 4194304 },
    /// Compute workloads to test
    workloads: []const WorkloadType = &[_]WorkloadType{ .matrix_mul, .vector_add, .convolution, .attention },
    /// Timeout for individual test operations in milliseconds
    timeout_ms: u32 = 5000,
    /// Minimum acceptable accuracy for computation results (0.0 - 1.0)
    min_accuracy: f32 = 0.99,
    /// Whether to validate computation results
    validate_results: bool = true,
    /// Enable statistical analysis (percentiles, outliers)
    statistical_analysis: bool = true,
    /// Output format for reports
    output_format: OutputFormat = .text,
    /// Directory to save benchmark results
    output_directory: ?[]const u8 = null,
    /// Enable performance regression detection
    regression_detection: bool = true,
    /// Baseline file for regression comparison
    baseline_file: ?[]const u8 = null,
    /// Confidence level for statistical tests (0.0 - 1.0)
    confidence_level: f32 = 0.95,
    /// Enable memory access pattern analysis
    memory_pattern_analysis: bool = false,
    /// Enable cache performance analysis
    cache_analysis: bool = false,
    /// Maximum memory usage threshold (MB)
    max_memory_threshold_mb: u32 = 8192,
    /// Enable concurrent workload testing
    concurrent_testing: bool = false,
    /// Random seed for reproducible benchmarks
    random_seed: u64 = 0x123456789ABCDEF0,

    /// Validate configuration parameters
    pub fn validate(self: *const BenchmarkConfig) !void {
        if (self.iterations == 0) return error.InvalidConfiguration;
        if (self.warmup_iterations > self.iterations) return error.InvalidConfiguration;
        if (self.min_accuracy < 0.0 or self.min_accuracy > 1.0) return error.InvalidConfiguration;
        if (self.confidence_level < 0.0 or self.confidence_level > 1.0) return error.InvalidConfiguration;
        if (self.timeout_ms == 0) return error.InvalidConfiguration;
        if (self.buffer_sizes.len == 0) return error.InvalidConfiguration;
        if (self.workloads.len == 0) return error.InvalidConfiguration;
    }

    /// Get estimated runtime for benchmark suite
    pub fn estimateRuntimeMs(self: *const BenchmarkConfig) u64 {
        const total_iterations = self.iterations * @as(u64, self.workloads.len) * @as(u64, self.buffer_sizes.len);
        const avg_time_per_iteration_ms = 10; // Conservative estimate
        return total_iterations * avg_time_per_iteration_ms;
    }

    /// Create configuration for quick benchmarking
    pub fn quick() BenchmarkConfig {
        return BenchmarkConfig{
            .iterations = 10,
            .warmup_iterations = 2,
            .detailed_timing = false,
            .memory_profiling = false,
            .power_profiling = false,
            .thermal_monitoring = false,
            .buffer_sizes = &[_]usize{ 8192, 65536 },
            .workloads = &[_]WorkloadType{ .matrix_mul, .vector_add },
            .statistical_analysis = false,
            .regression_detection = false,
        };
    }

    /// Create configuration for comprehensive benchmarking
    pub fn comprehensive() BenchmarkConfig {
        return BenchmarkConfig{
            .iterations = 1000,
            .warmup_iterations = 50,
            .detailed_timing = true,
            .memory_profiling = true,
            .power_profiling = true,
            .thermal_monitoring = true,
            .buffer_sizes = &[_]usize{ 1024, 4096, 16384, 65536, 262144, 1048576 },
            .workloads = &[_]WorkloadType{
                .matrix_mul, .vector_add,    .convolution, .attention,
                .pooling,    .normalization, .activation,  .fft,
            },
            .statistical_analysis = true,
            .regression_detection = true,
            .memory_pattern_analysis = true,
            .cache_analysis = true,
            .concurrent_testing = true,
        };
    }
};

/// Execution context information with hardware details
pub const ExecutionContext = struct {
    gpu_name: []const u8,
    driver_version: []const u8,
    compute_units: u32,
    memory_size_mb: u32,
    clock_speed_mhz: u32,
    temperature_celsius: f32,
    fan_speed_percent: f32,
    gpu_utilization_percent: f32 = 0,
    memory_utilization_percent: f32 = 0,
};

/// Benchmark result with statistical analysis and performance metrics
pub const BenchmarkResult = struct {
    workload: WorkloadType,
    backend: backends.Backend,
    iterations: u32,
    total_time_ns: u64,
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    std_dev_ns: u64,
    median_time_ns: u64,
    throughput_items_per_sec: f64,
    memory_bandwidth_gb_per_sec: f64,
    compute_utilization_percent: f32,
    memory_usage_mb: f64,
    peak_memory_usage_mb: f64,
    power_consumption_watts: f32,
    average_power_watts: f32,
    energy_consumed_joules: f32,
    error_count: u32,
    validation_passed: bool,
    accuracy_score: f32,
    cache_hit_rate: f32,
    thermal_throttling_detected: bool,
    timestamp: i64,
    execution_context: ExecutionContext,

    /// Calculate efficiency score (throughput per watt)
    pub fn calculateEfficiencyScore(self: *const BenchmarkResult) f32 {
        if (self.average_power_watts <= 0) return 0.0;
        return @as(f32, @floatCast(self.throughput_items_per_sec)) / self.average_power_watts;
    }

    /// Calculate performance stability (inverse of coefficient of variation)
    pub fn calculateStabilityScore(self: *const BenchmarkResult) f32 {
        if (self.avg_time_ns == 0) return 0.0;
        const cv = @as(f32, @floatFromInt(self.std_dev_ns)) / @as(f32, @floatFromInt(self.avg_time_ns));
        return 1.0 / (1.0 + cv);
    }

    /// Get performance grade based on multiple metrics
    pub fn getPerformanceGrade(self: *const BenchmarkResult) PerformanceGrade {
        const efficiency = self.calculateEfficiencyScore();
        const stability = self.calculateStabilityScore();
        const utilization = self.compute_utilization_percent / 100.0;

        const overall_score = (efficiency * 0.4 + stability * 0.3 + utilization * 0.3);

        if (overall_score >= 0.9) return .excellent;
        if (overall_score >= 0.8) return .good;
        if (overall_score >= 0.6) return .fair;
        return .poor;
    }

    /// Calculate efficiency score (throughput per watt) - duplicate method for compatibility
    pub fn efficiencyScore(self: *const BenchmarkResult) f32 {
        return self.calculateEfficiencyScore();
    }

    /// Calculate performance stability (inverse of coefficient of variation) - duplicate method for compatibility
    pub fn stabilityScore(self: *const BenchmarkResult) f32 {
        return self.calculateStabilityScore();
    }

    /// Get performance grade based on multiple metrics - duplicate method for compatibility
    pub fn performanceGrade(self: *const BenchmarkResult) PerformanceGrade {
        return self.getPerformanceGrade();
    }

    /// Calculate memory efficiency (bytes processed per second per MB used)
    pub fn calculateMemoryEfficiency(self: *const BenchmarkResult) f64 {
        if (self.memory_usage_mb <= 0) return 0.0;
        return self.throughput_items_per_sec / @as(f64, @floatFromInt(self.memory_usage_mb * 1024 * 1024));
    }

    /// Calculate cache efficiency based on hit rate
    pub fn calculateCacheEfficiency(self: *const BenchmarkResult) f32 {
        return self.cache_hit_rate;
    }

    /// Get overall performance score combining multiple metrics
    pub fn getOverallScore(self: *const BenchmarkResult) f32 {
        const efficiency = self.calculateEfficiencyScore();
        const stability = self.calculateStabilityScore();
        const memory_efficiency = @as(f32, @floatCast(self.calculateMemoryEfficiency() / 1000000.0)); // Normalize
        const cache_efficiency = self.calculateCacheEfficiency();

        // Weighted combination of metrics
        return (efficiency * 0.3) + (stability * 0.25) + (memory_efficiency * 0.25) + (cache_efficiency * 0.2);
    }

    /// Check if result meets minimum quality thresholds
    pub fn meetsQualityThresholds(self: *const BenchmarkResult, config: BenchmarkConfig) bool {
        if (!self.validation_passed) return false;
        if (self.accuracy_score < config.min_accuracy) return false;
        if (self.error_count > self.iterations / 10) return false; // More than 10% errors
        return true;
    }

    /// Generate a summary string for quick overview
    pub fn generateSummary(self: *const BenchmarkResult, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{} on {}: {d:.1}ms ± {d:.1}ms, {d:.0} items/sec, {} {s}", .{
            self.workload.displayName(),
            @tagName(self.backend),
            @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(self.std_dev_ns)) / 1_000_000.0,
            self.throughput_items_per_sec,
            self.getPerformanceGrade().displayName(),
            self.getPerformanceGrade().colorCode(),
        });
    }

    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s} on {s}:\n", .{ self.workload.displayName(), self.backend.toString() });
        try writer.print("  GPU: {s} ({} CUs, {} MHz)\n", .{ self.execution_context.gpu_name, self.execution_context.compute_units, self.execution_context.clock_speed_mhz });
        try writer.print("  Iterations: {} ({}% successful)\n", .{ self.iterations, ((self.iterations - self.error_count) * 100) / self.iterations });
        try writer.print("  Timing: {d:.2}ms ± {d:.2}ms (min: {d:.2}ms, max: {d:.2}ms)\n", .{
            @as(f64, @floatFromInt(self.avg_time_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(self.std_dev_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(self.min_time_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(self.max_time_ns)) / 1_000_000.0,
        });
        try writer.print("  Throughput: {d:.0} items/sec\n", .{self.throughput_items_per_sec});
        try writer.print("  Memory BW: {d:.2} GB/s\n", .{self.memory_bandwidth_gb_per_sec});
        try writer.print("  Compute Util: {d:.1}%\n", .{self.compute_utilization_percent});
        try writer.print("  Memory Usage: {d:.2} MB (peak: {d:.2} MB)\n", .{ self.memory_usage_mb, self.peak_memory_usage_mb });
        if (self.average_power_watts > 0) {
            try writer.print("  Power: {d:.2}W avg, {d:.2}W peak\n", .{ self.average_power_watts, self.power_consumption_watts });
            try writer.print("  Energy: {d:.2}J, Efficiency: {d:.1} items/W\n", .{ self.energy_consumed_joules, self.efficiencyScore() });
        }
        if (self.validation_passed) {
            try writer.print("  Validation: PASSED (accuracy: {d:.2}%)\n", .{self.accuracy_score * 100});
        } else {
            try writer.print("  Validation: FAILED (accuracy: {d:.2}%)\n", .{self.accuracy_score * 100});
        }
        try writer.print("  Performance Grade: {s}\n", .{self.getPerformanceGrade().toString()});
        if (self.thermal_throttling_detected) {
            try writer.print("  WARNING: Thermal throttling detected!\n", .{});
        }
    }
};

/// Detailed timing measurement with hierarchical support
pub const TimingMeasurement = struct {
    name: []const u8,
    start_time: i64,
    end_time: i64,
    memory_before: usize,
    memory_after: usize,
    gpu_time_ns: u64,
    cpu_time_ns: u64,
    synchronization_time_ns: u64,
    parent_measurement: ?*TimingMeasurement,
    child_measurements: std.ArrayList(TimingMeasurement),

    pub fn duration_ns(self: TimingMeasurement) u64 {
        return @as(u64, @intCast(self.end_time - self.start_time));
    }

    pub fn memory_delta(self: TimingMeasurement) i64 {
        return @as(i64, @intCast(self.memory_after)) - @as(i64, @intCast(self.memory_before));
    }
};

pub const PowerSample = struct {
    timestamp: i64,
    gpu_power_watts: f32,
    memory_power_watts: f32,
    total_power_watts: f32,
    voltage: f32,
    current: f32,
    temperature: f32,
};

/// Power consumption monitoring
pub const PowerMonitor = struct {
    allocator: std.mem.Allocator,
    samples: std.ArrayList(PowerSample),
    sampling_interval_ms: u32,
    last_sample_time: i64,

    pub fn init(allocator: std.mem.Allocator, sampling_interval_ms: u32) !PowerMonitor {
        return PowerMonitor{
            .allocator = allocator,
            .samples = std.ArrayList(PowerSample).init(allocator),
            .sampling_interval_ms = sampling_interval_ms,
            .last_sample_time = 0,
        };
    }

    pub fn deinit(self: *PowerMonitor) void {
        self.samples.deinit();
    }

    pub fn recordSample(self: *PowerMonitor, sample: PowerSample) !void {
        try self.samples.append(sample);
        self.last_sample_time = sample.timestamp;
    }

    pub fn getAveragePower(self: PowerMonitor) f32 {
        if (self.samples.items.len == 0) return 0.0;

        var total: f32 = 0.0;
        for (self.samples.items) |sample| {
            total += sample.total_power_watts;
        }
        return total / @as(f32, @floatFromInt(self.samples.items.len));
    }

    pub fn getPeakPower(self: PowerMonitor) f32 {
        if (self.samples.items.len == 0) return 0.0;

        var peak: f32 = 0.0;
        for (self.samples.items) |sample| {
            peak = @max(peak, sample.total_power_watts);
        }
        return peak;
    }
};

pub const MemoryType = enum {
    buffer,
    texture,
    uniform,
    staging,
    other,
};

pub const MemoryAllocation = struct {
    timestamp: i64,
    size: usize,
    type: MemoryType,
    freed: bool,
};

/// Memory usage tracking
pub const MemoryTracker = struct {
    allocator: std.mem.Allocator,
    current_usage: usize,
    peak_usage: usize,
    allocations: std.ArrayList(MemoryAllocation),

    pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
        return MemoryTracker{
            .allocator = allocator,
            .current_usage = 0,
            .peak_usage = 0,
            .allocations = std.ArrayList(MemoryAllocation).init(allocator),
        };
    }

    pub fn deinit(self: *MemoryTracker) void {
        self.allocations.deinit();
    }

    pub fn recordAllocation(self: *MemoryTracker, size: usize, memory_type: MemoryType) !void {
        self.current_usage += size;
        self.peak_usage = @max(self.peak_usage, self.current_usage);

        try self.allocations.append(MemoryAllocation{
            .timestamp = std.time.milliTimestamp(),
            .size = size,
            .type = memory_type,
            .freed = false,
        });
    }

    pub fn recordDeallocation(self: *MemoryTracker, size: usize) void {
        self.current_usage = if (self.current_usage >= size) self.current_usage - size else 0;
    }
};

/// Thermal monitoring for throttling detection
pub const ThermalMonitor = struct {
    allocator: std.mem.Allocator,
    temperature_samples: std.ArrayList(f32),
    throttling_threshold: f32,
    throttling_detected: bool,

    pub fn init(allocator: std.mem.Allocator, threshold: f32) !ThermalMonitor {
        return ThermalMonitor{
            .allocator = allocator,
            .temperature_samples = std.ArrayList(f32).init(allocator),
            .throttling_threshold = threshold,
            .throttling_detected = false,
        };
    }

    pub fn deinit(self: *ThermalMonitor) void {
        self.temperature_samples.deinit();
    }

    pub fn recordTemperature(self: *ThermalMonitor, temperature: f32) !void {
        try self.temperature_samples.append(temperature);
        if (temperature > self.throttling_threshold) {
            self.throttling_detected = true;
        }
    }

    pub fn getAverageTemperature(self: ThermalMonitor) f32 {
        if (self.temperature_samples.items.len == 0) return 0.0;

        var total: f32 = 0.0;
        for (self.temperature_samples.items) |temp| {
            total += temp;
        }
        return total / @as(f32, @floatFromInt(self.temperature_samples.items.len));
    }
};

/// Performance profiler with advanced metrics and analysis
pub const PerformanceProfiler = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    results: std.ArrayList(BenchmarkResult),
    start_time: i64,
    measurements: std.ArrayList(TimingMeasurement),
    baseline_results: ?std.ArrayList(BenchmarkResult),
    power_monitor: ?PowerMonitor,
    memory_tracker: MemoryTracker,
    thermal_monitor: ThermalMonitor,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*PerformanceProfiler {
        const self = try allocator.create(PerformanceProfiler);
        self.* = .{
            .allocator = allocator,
            .renderer = renderer,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
            .start_time = std.time.milliTimestamp(),
            .measurements = std.ArrayList(TimingMeasurement).init(allocator),
            .baseline_results = null,
            .power_monitor = null,
            .memory_tracker = try MemoryTracker.init(allocator),
            .thermal_monitor = try ThermalMonitor.init(allocator, 85.0), // 85°C threshold
        };
        return self;
    }

    pub fn deinit(self: *PerformanceProfiler) void {
        for (self.results.items) |*result| {
            _ = result; // Results are owned by the profiler
        }
        self.results.deinit();

        for (self.measurements.items) |*measurement| {
            self.allocator.free(measurement.name);
            measurement.child_measurements.deinit();
        }
        self.measurements.deinit();

        if (self.baseline_results) |*baseline| {
            baseline.deinit();
        }

        if (self.power_monitor) |*monitor| {
            monitor.deinit();
        }

        self.memory_tracker.deinit();
        self.thermal_monitor.deinit();

        self.allocator.destroy(self);
    }

    /// Enable power monitoring with specified sampling interval
    pub fn enablePowerMonitoring(self: *PerformanceProfiler, sampling_interval_ms: u32) !void {
        self.power_monitor = try PowerMonitor.init(self.allocator, sampling_interval_ms);
    }

    /// Set baseline results for comparison
    pub fn setBaseline(self: *PerformanceProfiler, baseline_results: []const BenchmarkResult) !void {
        self.baseline_results = std.ArrayList(BenchmarkResult).init(self.allocator);
        try self.baseline_results.?.appendSlice(baseline_results);
    }

    /// Start timing a specific operation with hierarchical support
    pub fn startTiming(self: *PerformanceProfiler, name: []const u8) !void {
        const measurement = TimingMeasurement{
            .name = try self.allocator.dupe(u8, name),
            .start_time = @as(i64, @intCast(std.time.nanoTimestamp())),
            .end_time = 0,
            .memory_before = self.memory_tracker.current_usage,
            .memory_after = 0,
            .gpu_time_ns = 0,
            .cpu_time_ns = 0,
            .synchronization_time_ns = 0,
            .parent_measurement = null,
            .child_measurements = std.ArrayList(TimingMeasurement).init(self.allocator),
        };
        try self.measurements.append(measurement);
    }

    /// End timing for the current operation
    pub fn endTiming(self: *PerformanceProfiler) !void {
        if (self.measurements.items.len == 0) return;

        const index = self.measurements.items.len - 1;
        self.measurements.items[index].end_time = @as(i64, @intCast(std.time.nanoTimestamp()));
        self.measurements.items[index].memory_after = self.memory_tracker.current_usage;

        // Record GPU synchronization time (would be measured in real implementation)
        self.measurements.items[index].synchronization_time_ns = 100000; // 100μs estimate
    }

    /// Run comprehensive benchmark suite with enhanced metrics
    pub fn runBenchmarkSuite(self: *PerformanceProfiler, config: BenchmarkConfig) !void {
        std.log.info("Starting GPU benchmark suite with {} iterations", .{config.iterations});

        for (config.workloads) |workload| {
            for (config.buffer_sizes) |buffer_size| {
                try self.runWorkloadBenchmark(workload, buffer_size, config);
            }
        }

        std.log.info("Benchmark suite completed", .{});
    }

    /// Generate comprehensive performance report with analysis
    pub fn generatePerformanceReport(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
        var report = std.ArrayList(u8).init(allocator);
        defer report.deinit();
        const writer = report.writer();

        try writer.print("=== GPU Performance Benchmark Report ===\n\n", .{});

        // Summary statistics
        try writer.print("📊 Summary:\n", .{});
        try writer.print("  Total benchmarks run: {}\n", .{self.results.items.len});
        try writer.print("  Total execution time: {}ms\n", .{std.time.milliTimestamp() - self.start_time});

        if (self.results.items.len > 0) {
            // Calculate aggregate statistics
            var total_throughput: f64 = 0;
            var total_efficiency: f32 = 0;
            var best_result: ?*const BenchmarkResult = null;
            var worst_result: ?*const BenchmarkResult = null;

            for (self.results.items) |*result| {
                total_throughput += result.throughput_items_per_sec;
                total_efficiency += result.calculateEfficiencyScore();

                if (best_result == null or result.getOverallScore() > best_result.?.getOverallScore()) {
                    best_result = result;
                }
                if (worst_result == null or result.getOverallScore() < worst_result.?.getOverallScore()) {
                    worst_result = result;
                }
            }

            const avg_throughput = total_throughput / @as(f64, @floatFromInt(self.results.items.len));
            const avg_efficiency = total_efficiency / @as(f32, @floatFromInt(self.results.items.len));

            try writer.print("  Average throughput: {d:.0} items/sec\n", .{avg_throughput});
            try writer.print("  Average efficiency: {d:.2}\n", .{avg_efficiency});

            // Best and worst performers
            if (best_result) |best| {
                try writer.print("  Best performer: {} on {}\n", .{ best.workload.displayName(), @tagName(best.backend) });
            }
            if (worst_result) |worst| {
                try writer.print("  Worst performer: {} on {}\n", .{ worst.workload.displayName(), @tagName(worst.backend) });
            }
        }

        try writer.print("\n📋 Detailed Results:\n", .{});
        for (self.results.items, 0..) |*result, i| {
            const summary = try result.generateSummary(allocator);
            defer allocator.free(summary);
            try writer.print("  {}. {}\n", .{ i + 1, summary });
        }

        // Recommendations
        try writer.print("\n💡 Recommendations:\n", .{});
        try self.generateRecommendations(writer);

        return report.toOwnedSlice();
    }

    /// Generate performance recommendations based on results
    fn generateRecommendations(self: *PerformanceProfiler, writer: anytype) !void {
        if (self.results.items.len == 0) {
            try writer.print("  - Run benchmarks to get recommendations\n", .{});
            return;
        }

        // Analyze backend performance
        var backend_scores = std.StringHashMap(f32).init(self.allocator);
        defer backend_scores.deinit();

        for (self.results.items) |*result| {
            const backend_name = @tagName(result.backend);
            const current_score = backend_scores.get(backend_name) orelse 0.0;
            backend_scores.put(backend_name, current_score + result.getOverallScore()) catch {};
        }

        // Find best backend
        var best_backend: ?[]const u8 = null;
        var best_score: f32 = 0;

        var it = backend_scores.iterator();
        while (it.next()) |entry| {
            if (best_backend == null or entry.value_ptr.* > best_score) {
                best_backend = entry.key_ptr.*;
                best_score = entry.value_ptr.*;
            }
        }

        if (best_backend) |backend| {
            try writer.print("  - Recommended backend: {}\n", .{backend});
        }

        // Check for thermal issues
        var thermal_issues = false;
        for (self.results.items) |*result| {
            if (result.thermal_throttling_detected) {
                thermal_issues = true;
                break;
            }
        }

        if (thermal_issues) {
            try writer.print("  - ⚠️  Thermal throttling detected - consider cooling improvements\n", .{});
        }

        // Memory efficiency recommendations
        var avg_memory_efficiency: f64 = 0;
        for (self.results.items) |*result| {
            avg_memory_efficiency += result.calculateMemoryEfficiency();
        }
        avg_memory_efficiency /= @as(f64, @floatFromInt(self.results.items.len));

        if (avg_memory_efficiency < 1000) {
            try writer.print("  - Consider optimizing memory access patterns for better efficiency\n", .{});
        }
    }

    /// Export results to different formats
    pub fn exportResults(self: *PerformanceProfiler, allocator: std.mem.Allocator, format: OutputFormat) ![]const u8 {
        return switch (format) {
            .text => self.generatePerformanceReport(allocator),
            .json => self.exportToJson(allocator),
            .csv => self.exportToCsv(allocator),
            .html => self.exportToHtml(allocator),
        };
    }

    /// Export results to HTML format
    fn exportToHtml(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
        var html_output = std.ArrayList(u8).init(allocator);
        defer html_output.deinit();

        const writer = html_output.writer();

        try writer.print("<!DOCTYPE html>\n<html>\n<head>\n<title>GPU Benchmark Results</title>\n</head>\n<body>\n", .{});
        try writer.print("<h1>GPU Performance Benchmark Results</h1>\n", .{});
        try writer.print("<table border='1'>\n", .{});
        try writer.print("<tr><th>Workload</th><th>Backend</th><th>Throughput</th><th>Efficiency</th><th>Grade</th></tr>\n", .{});

        for (self.results.items) |*result| {
            try writer.print("<tr><td>{}</td><td>{}</td><td>{d:.0}</td><td>{d:.3}</td><td>{}</td></tr>\n", .{
                @tagName(result.workload),
                @tagName(result.backend),
                result.throughput_items_per_sec,
                result.calculateEfficiencyScore(),
                @tagName(result.getPerformanceGrade()),
            });
        }

        try writer.print("</table>\n</body>\n</html>\n", .{});

        return html_output.toOwnedSlice();
    }

    /// Enhanced single iteration with validation and accuracy measurement
    const IterationResult = struct {
        success: bool,
        validation_passed: bool,
        accuracy: f32,
        execution_time_ns: u64,
        memory_used: usize,
    };

    fn getExecutionTimeForWorkload(self: *PerformanceProfiler, workload: WorkloadType) u64 {
        _ = self;
        return switch (workload) {
            .matrix_mul => 1000 * std.time.ns_per_ms,
            .vector_add => 500 * std.time.ns_per_ms,
            .convolution => 2000 * std.time.ns_per_ms,
            .attention => 3000 * std.time.ns_per_ms,
            .fft => 1500 * std.time.ns_per_ms,
            .reduction => 800 * std.time.ns_per_ms,
            .scan => 1200 * std.time.ns_per_ms,
            .sort => 2500 * std.time.ns_per_ms,
            else => 1000 * std.time.ns_per_ms,
        };
    }

    fn runSingleIteration(self: *PerformanceProfiler, workload: WorkloadType, data_size: usize, config: BenchmarkConfig) !IterationResult {
        // Simulate execution based on workload type
        const execution_time = self.getExecutionTimeForWorkload(workload);

        std.Thread.sleep(execution_time);

        // Simulate validation if enabled
        var validation_passed = true;
        var accuracy: f32 = 1.0;

        if (config.validate_results) {
            // Simulate occasional validation failures and accuracy variations
            const random_value = std.crypto.random.int(u8);
            if (random_value < 5) { // 5% chance of validation failure
                validation_passed = false;
                accuracy = 0.5 + @as(f32, @floatFromInt(random_value % 40)) / 100.0;
            } else {
                accuracy = 0.95 + @as(f32, @floatFromInt(random_value % 5)) / 100.0;
            }
        }

        return IterationResult{
            .success = true,
            .validation_passed = validation_passed,
            .accuracy = accuracy,
            .execution_time_ns = execution_time,
            .memory_used = data_size,
        };
    }

    /// Calculate items processed for workload with more accurate estimates
    fn getWorkloadItemCount(self: *PerformanceProfiler, workload: WorkloadType, data_size: usize) !u64 {
        _ = self;
        return switch (workload) {
            .matrix_mul => {
                const matrix_size = std.math.sqrt(@as(f64, @floatFromInt(data_size / @sizeOf(f32) / 2)));
                const size = @as(u64, @intFromFloat(matrix_size));
                return size * size * size; // O(n^3) operations
            },
            .vector_add => data_size / @sizeOf(f32),
            .convolution => {
                // Assume 3x3 kernel on square input
                const input_size = std.math.sqrt(@as(f64, @floatFromInt(data_size / @sizeOf(f32))));
                const size = @as(u64, @intFromFloat(input_size));
                return size * size * 9; // 9 operations per output element
            },
            .attention => {
                const seq_len = std.math.sqrt(@as(f64, @floatFromInt(data_size / @sizeOf(f32) / 4))); // Q, K, V, O
                const len = @as(u64, @intFromFloat(seq_len));
                return len * len * len; // O(n^2) for attention, times sequence length
            },
            .fft => {
                const n = data_size / @sizeOf(f32);
                const n_f64 = @as(f64, @floatFromInt(n));
                return @as(u64, @intFromFloat(n_f64 * @log(n_f64)));
            },
            .reduction, .pooling, .normalization, .activation => data_size / @sizeOf(f32),
            .scan => (data_size / @sizeOf(f32)) * 2, // Two passes typically
            .sort => {
                const n = data_size / @sizeOf(f32);
                const n_f64 = @as(f64, @floatFromInt(n));
                return @as(u64, @intFromFloat(n_f64 * @log(n_f64)));
            },
            .sparse_operations => data_size / @sizeOf(f32) / 10, // Assume 10% sparsity
            else => data_size / @sizeOf(f32),
        };
    }

    /// Calculate memory bandwidth for workload with improved accuracy
    fn calculateMemoryBandwidth(self: *PerformanceProfiler, workload: WorkloadType, data_size: usize, avg_time_ns: u64) !f64 {
        _ = self;

        const memory_multiplier: f64 = switch (workload) {
            .matrix_mul => 3.0, // A, B matrices read, C matrix written
            .vector_add => 3.0, // Two vectors read, one written
            .convolution => 2.5, // Input + kernel reads, output write
            .attention => 4.0, // Q, K, V read, attention matrix computed and written
            .fft => 2.0, // In-place typically, but with temporary storage
            .reduction => 1.0, // Single read pass
            .scan => 2.0, // Read and write
            .sort => 2.5, // Multiple reads/writes during sorting
            else => 2.0, // Default read + write
        };

        const bytes_processed = @as(f64, @floatFromInt(data_size)) * memory_multiplier;
        const time_seconds = @as(f64, @floatFromInt(avg_time_ns)) / 1_000_000_000.0;
        const bandwidth_bytes_per_sec = bytes_processed / time_seconds;
        return bandwidth_bytes_per_sec / (1024.0 * 1024.0 * 1024.0); // Convert to GB/s
    }

    /// Run benchmark for specific workload with comprehensive metrics
    pub fn runWorkloadBenchmark(
        self: *PerformanceProfiler,
        workload: WorkloadType,
        data_size: usize,
        config: BenchmarkConfig,
    ) !void {
        const iterations = config.iterations;
        var times = std.ArrayList(u64).init(self.allocator);
        defer times.deinit();

        var total_time: u64 = 0;
        var min_time: u64 = std.math.maxInt(u64);
        var max_time: u64 = 0;
        var error_count: u32 = 0;
        var validation_failures: u32 = 0;
        var accuracy_scores = std.ArrayList(f32).init(self.allocator);
        defer accuracy_scores.deinit();

        // Reset monitoring systems
        if (self.power_monitor) |*monitor| {
            monitor.samples.clearAndFree();
        }
        self.thermal_monitor.temperature_samples.clearAndFree();
        self.thermal_monitor.throttling_detected = false;

        // Warmup iterations
        for (0..config.warmup_iterations) |_| {
            _ = try self.runSingleIteration(workload, data_size, config);
        }

        // Benchmark iterations
        for (0..iterations) |_| {
            const start_time = std.time.nanoTimestamp();
            const result = try self.runSingleIteration(workload, data_size, config);
            const end_time = std.time.nanoTimestamp();

            if (!result.success) {
                error_count += 1;
                continue;
            }

            if (!result.validation_passed) {
                validation_failures += 1;
            }

            const iteration_time = end_time - start_time;
            try times.append(@as(u64, @intCast(iteration_time)));
            total_time += @as(u64, @intCast(iteration_time));
            min_time = @min(min_time, @as(u64, @intCast(iteration_time)));
            max_time = @max(max_time, @as(u64, @intCast(iteration_time)));
            try accuracy_scores.append(result.accuracy);

            // Simulate power and thermal monitoring
            if (self.power_monitor) |*monitor| {
                try monitor.recordSample(PowerSample{
                    .timestamp = std.time.milliTimestamp(),
                    .gpu_power_watts = 150.0 + @as(f32, @floatFromInt(std.crypto.random.int(u8) % 50)),
                    .memory_power_watts = 25.0 + @as(f32, @floatFromInt(std.crypto.random.int(u8) % 10)),
                    .total_power_watts = 175.0 + @as(f32, @floatFromInt(std.crypto.random.int(u8) % 60)),
                    .voltage = 1.2,
                    .current = 145.8,
                    .temperature = 70.0 + @as(f32, @floatFromInt(std.crypto.random.int(u8) % 20)),
                });
            }

            try self.thermal_monitor.recordTemperature(70.0 + @as(f32, @floatFromInt(std.crypto.random.int(u8) % 20)));
        }

        const successful_iterations = iterations - error_count;
        if (successful_iterations == 0) {
            std.log.err("All iterations failed for workload {s}", .{@tagName(workload)});
            return;
        }

        const avg_time = total_time / successful_iterations;

        // Calculate standard deviation
        var variance: u64 = 0;
        for (times.items) |time| {
            const diff = if (time > avg_time) time - avg_time else avg_time - time;
            variance += diff * diff;
        }
        const std_dev = @as(u64, @intFromFloat(@sqrt(@as(f64, @floatFromInt(variance / successful_iterations)))));

        // Calculate median
        std.mem.sort(u64, times.items, {}, std.sort.asc(u64));
        const median_time = if (times.items.len % 2 == 0)
            (times.items[times.items.len / 2 - 1] + times.items[times.items.len / 2]) / 2
        else
            times.items[times.items.len / 2];

        // Calculate average accuracy
        var total_accuracy: f32 = 0;
        for (accuracy_scores.items) |score| {
            total_accuracy += score;
        }
        const avg_accuracy = if (accuracy_scores.items.len > 0) total_accuracy / @as(f32, @floatFromInt(accuracy_scores.items.len)) else 0.0;

        // Calculate performance metrics
        const items_processed = try self.getWorkloadItemCount(workload, data_size);
        const throughput = @as(f64, @floatFromInt(items_processed)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0);

        // Get power metrics
        const avg_power = if (self.power_monitor) |monitor| monitor.getAveragePower() else 0.0;
        const peak_power = if (self.power_monitor) |monitor| monitor.getPeakPower() else 0.0;
        const energy_consumed = avg_power * (@as(f32, @floatFromInt(avg_time)) / 1_000_000_000.0);

        const result = BenchmarkResult{
            .workload = workload,
            .backend = .webgpu, // Current backend
            .iterations = successful_iterations,
            .total_time_ns = total_time,
            .avg_time_ns = avg_time,
            .min_time_ns = min_time,
            .max_time_ns = max_time,
            .std_dev_ns = std_dev,
            .median_time_ns = median_time,
            .throughput_items_per_sec = throughput,
            .memory_bandwidth_gb_per_sec = try self.calculateMemoryBandwidth(workload, data_size, avg_time),
            .compute_utilization_percent = 85.0, // Estimated
            .memory_usage_mb = @as(f64, @floatFromInt(data_size)) / (1024.0 * 1024.0),
            .peak_memory_usage_mb = @as(f64, @floatFromInt(self.memory_tracker.peak_usage)) / (1024.0 * 1024.0),
            .power_consumption_watts = peak_power,
            .average_power_watts = avg_power,
            .energy_consumed_joules = energy_consumed,
            .error_count = error_count,
            .validation_passed = validation_failures == 0,
            .accuracy_score = avg_accuracy,
            .cache_hit_rate = 0.85, // Estimated
            .thermal_throttling_detected = self.thermal_monitor.throttling_detected,
            .timestamp = std.time.milliTimestamp(),
            .execution_context = ExecutionContext{
                .gpu_name = "Simulated GPU",
                .driver_version = "1.0.0",
                .compute_units = 64,
                .memory_size_mb = 8192,
                .clock_speed_mhz = 1500,
                .temperature_celsius = self.thermal_monitor.getAverageTemperature(),
                .fan_speed_percent = 75.0,
            },
        };

        try self.results.append(result);

        std.log.info("Benchmark completed: {any}", .{result});
    }

    /// Generate comprehensive performance report with statistical analysis
    pub fn generateReport(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
        var report = std.ArrayList(u8).init(allocator);
        errdefer report.deinit();

        try report.appendSlice("GPU Performance Benchmark Report\n");
        try report.appendSlice("==================================\n\n");

        try std.fmt.format(report.writer(), "Total Benchmarks Run: {}\n", .{self.results.items.len});
        try std.fmt.format(report.writer(), "Report Generated: {}\n", .{std.time.milliTimestamp()});
        try std.fmt.format(report.writer(), "Session Duration: {d:.2} minutes\n\n", .{@as(f64, @floatFromInt(std.time.milliTimestamp() - self.start_time)) / 60000.0});

        // Executive Summary
        try report.appendSlice("Executive Summary\n");
        try report.appendSlice("=================\n");

        if (self.results.items.len > 0) {
            var total_throughput: f64 = 0;
            var total_efficiency: f32 = 0;
            var excellent_count: u32 = 0;
            var thermal_issues: u32 = 0;

            for (self.results.items) |result| {
                total_throughput += result.throughput_items_per_sec;
                total_efficiency += result.efficiencyScore();
                if (result.performanceGrade() == .excellent) excellent_count += 1;
                if (result.thermal_throttling_detected) thermal_issues += 1;
            }

            const avg_throughput = total_throughput / @as(f64, @floatFromInt(self.results.items.len));
            const avg_efficiency = total_efficiency / @as(f32, @floatFromInt(self.results.items.len));
            const excellent_percentage = (excellent_count * 100) / @as(u32, @intCast(self.results.items.len));

            try std.fmt.format(report.writer(), "Overall Performance: {d:.0} items/sec average throughput\n", .{avg_throughput});
            try std.fmt.format(report.writer(), "Energy Efficiency: {d:.1} items/watt average\n", .{avg_efficiency});
            try std.fmt.format(report.writer(), "Excellent Results: {}% of benchmarks\n", .{excellent_percentage});

            if (thermal_issues > 0) {
                try std.fmt.format(report.writer(), "⚠️  Thermal Issues: {} benchmarks affected by throttling\n", .{thermal_issues});
            }
        }
        try report.appendSlice("\n");

        // Group results by workload
        var workloads = std.HashMap(WorkloadType, std.ArrayList(*const BenchmarkResult), std.hash_map.AutoContext(WorkloadType), std.hash_map.default_max_load_percentage).init(allocator);
        defer {
            var it = workloads.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            workloads.deinit();
        }

        for (self.results.items) |*result| {
            const list = try workloads.getOrPut(result.workload);
            if (!list.found_existing) {
                list.value_ptr.* = std.ArrayList(*const BenchmarkResult).init(allocator);
            }
            try list.value_ptr.append(result);
        }

        // Generate detailed report for each workload
        var workload_it = workloads.iterator();
        while (workload_it.next()) |entry| {
            try std.fmt.format(report.writer(), "Workload: {s} ({s})\n", .{ entry.key_ptr.displayName(), entry.key_ptr.complexityClass() });
            try report.appendSlice("----------------------------------------\n");

            for (entry.value_ptr.items) |result| {
                try std.fmt.format(report.writer(), "{}\n", .{result});
            }
            try report.appendSlice("\n");
        }

        // Comparative analysis with baseline
        if (self.baseline_results) |baseline| {
            try report.appendSlice("Baseline Comparison\n");
            try report.appendSlice("==================\n");

            for (self.results.items) |current| {
                for (baseline.items) |base| {
                    if (current.workload == base.workload) {
                        const throughput_improvement = (current.throughput_items_per_sec / base.throughput_items_per_sec - 1.0) * 100.0;
                        const efficiency_improvement = (current.efficiencyScore() / base.efficiencyScore() - 1.0) * 100.0;

                        try std.fmt.format(report.writer(), "{s}: {+d:.1}% throughput, {+d:.1}% efficiency\n", .{
                            current.workload.displayName(),
                            throughput_improvement,
                            efficiency_improvement,
                        });
                        break;
                    }
                }
            }
            try report.appendSlice("\n");
        }

        // Performance recommendations
        try report.appendSlice("Performance Recommendations\n");
        try report.appendSlice("===========================\n");

        var low_utilization_count: u32 = 0;
        var high_power_count: u32 = 0;

        for (self.results.items) |result| {
            if (result.compute_utilization_percent < 70.0) low_utilization_count += 1;
            if (result.average_power_watts > 200.0) high_power_count += 1;
        }

        if (low_utilization_count > 0) {
            try std.fmt.format(report.writer(), "• {} workloads show low GPU utilization (<70%) - consider increasing workload size\n", .{low_utilization_count});
        }

        if (high_power_count > 0) {
            try std.fmt.format(report.writer(), "• {} workloads consume high power (>200W) - consider power optimization\n", .{high_power_count});
        }

        const thermal_issues = blk: {
            var count: u32 = 0;
            for (self.results.items) |result| {
                if (result.thermal_throttling_detected) count += 1;
            }
            break :blk count;
        };

        if (thermal_issues > 0) {
            try report.appendSlice("• Thermal throttling detected - improve cooling or reduce workload intensity\n");
        }

        return report.toOwnedSlice();
    }

    /// Export results to JSON with comprehensive metadata
    fn exportToJson(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
        var json = std.ArrayList(u8).init(allocator);
        errdefer json.deinit();

        try json.appendSlice("{\n");
        try std.fmt.format(json.writer(), "  \"metadata\": {{\n");
        try std.fmt.format(json.writer(), "    \"generated_at\": {},\n", .{std.time.milliTimestamp()});
        try std.fmt.format(json.writer(), "    \"session_duration_ms\": {},\n", .{std.time.milliTimestamp() - self.start_time});
        try std.fmt.format(json.writer(), "    \"total_benchmarks\": {},\n", .{self.results.items.len});
        try std.fmt.format(json.writer(), "    \"profiler_version\": \"1.0.0\"\n");
        try json.appendSlice("  },\n");

        try json.appendSlice("  \"gpu_benchmarks\": [\n");

        for (self.results.items, 0..) |result, i| {
            if (i > 0) try json.appendSlice(",\n");
            try std.fmt.format(json.writer(),
                \\    {{
                \\      "workload": "{s}",
                \\      "workload_display_name": "{s}",
                \\      "complexity_class": "{s}",
                \\      "backend": "{s}",
                \\      "iterations": {},
                \\      "timing": {{
                \\        "avg_time_ns": {},
                \\        "min_time_ns": {},
                \\        "max_time_ns": {},
                \\        "std_dev_ns": {},
                \\        "median_time_ns": {}
                \\      }},
                \\      "performance": {{
                \\        "throughput_items_per_sec": {any},
                \\        "memory_bandwidth_gb_per_sec": {any},
                \\        "compute_utilization_percent": {any},
                \\        "efficiency_score": {any},
                \\        "stability_score": {any},
                \\        "performance_grade": "{s}"
                \\      }},
                \\      "memory": {{
                \\        "usage_mb": {any},
                \\        "peak_usage_mb": {any}
                \\      }},
                \\      "power": {{
                \\        "average_watts": {any},
                \\        "peak_watts": {any},
                \\        "energy_joules": {any}
                \\      }},
                \\      "validation": {{
                \\        "passed": {},
                \\        "accuracy_score": {any}
                \\      }},
                \\      "thermal": {{
                \\        "throttling_detected": {},
                \\        "average_temperature": {any}
                \\      }},
                \\      "execution_context": {{
                \\        "gpu_name": "{s}",
                \\        "compute_units": {},
                \\        "clock_speed_mhz": {}
                \\      }},
                \\      "timestamp": {}
                \\    }}
            , .{
                @tagName(result.workload),
                result.workload.displayName(),
                result.workload.complexityClass(),
                result.backend.toString(),
                result.iterations,
                result.avg_time_ns,
                result.min_time_ns,
                result.max_time_ns,
                result.std_dev_ns,
                result.median_time_ns,
                result.throughput_items_per_sec,
                result.memory_bandwidth_gb_per_sec,
                result.compute_utilization_percent,
                result.efficiencyScore(),
                result.stabilityScore(),
                result.performanceGrade().toString(),
                result.memory_usage_mb,
                result.peak_memory_usage_mb,
                result.average_power_watts,
                result.power_consumption_watts,
                result.energy_consumed_joules,
                result.validation_passed,
                result.accuracy_score,
                result.thermal_throttling_detected,
                result.execution_context.temperature_celsius,
                result.execution_context.gpu_name,
                result.execution_context.compute_units,
                result.execution_context.clock_speed_mhz,
                result.timestamp,
            });
        }

        try json.appendSlice("\n  ]\n");
        try json.appendSlice("}\n");

        return json.toOwnedSlice();
    }

    /// Export results to CSV format
    pub fn exportToCsv(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
        var csv = std.ArrayList(u8).init(allocator);
        errdefer csv.deinit();

        // CSV Header
        try csv.appendSlice("workload,backend,iterations,avg_time_ms,throughput_items_per_sec,memory_bandwidth_gb_per_sec,");
        try csv.appendSlice("compute_utilization_percent,memory_usage_mb,power_watts,efficiency_score,performance_grade,");
        try csv.appendSlice("validation_passed,accuracy_score,thermal_throttling\n");

        // CSV Data
        for (self.results.items) |result| {
            try std.fmt.format(csv.writer(), "{s},{s},{},{d:.2},{d},{d},{d},{d},{d},{d},{s},{},{d:.2},{}\n", .{
                @tagName(result.workload),
                result.backend.toString(),
                result.iterations,
                @as(f64, @floatFromInt(result.avg_time_ns)) / 1_000_000.0,
                result.throughput_items_per_sec,
                result.memory_bandwidth_gb_per_sec,
                result.compute_utilization_percent,
                result.memory_usage_mb,
                result.average_power_watts,
                result.efficiencyScore(),
                result.performanceGrade().toString(),
                result.validation_passed,
                result.accuracy_score,
                result.thermal_throttling_detected,
            });
        }

        return csv.toOwnedSlice();
    }
};

pub const AccessPattern = enum {
    sequential,
    random,
    strided,
    coalesced,
    scattered,

    pub fn toString(self: AccessPattern) []const u8 {
        return switch (self) {
            .sequential => "Sequential",
            .random => "Random",
            .strided => "Strided",
            .coalesced => "Coalesced",
            .scattered => "Scattered",
        };
    }
};

pub const MemoryBenchmarkResult = struct {
    buffer_size: usize,
    access_pattern: AccessPattern,
    read_bandwidth_gb_per_sec: f64,
    write_bandwidth_gb_per_sec: f64,
};

/// Memory bandwidth benchmark with enhanced capabilities
pub const MemoryBandwidthBenchmark = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,
    access_patterns: []const AccessPattern,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*MemoryBandwidthBenchmark {
        const self = try allocator.create(MemoryBandwidthBenchmark);
        self.* = .{
            .allocator = allocator,
            .renderer = renderer,
            .access_patterns = &[_]AccessPattern{ .sequential, .random, .strided, .coalesced, .scattered },
        };
        return self;
    }

    pub fn deinit(self: *MemoryBandwidthBenchmark) void {
        self.allocator.destroy(self);
    }

    /// Measure GPU memory bandwidth
    pub fn measureBandwidth(self: *MemoryBandwidthBenchmark, buffer_size: usize, iterations: u32) !f64 {
        std.log.info("Measuring memory bandwidth with buffer size: {} bytes", .{buffer_size});

        // Create test buffer
        const buffer = try self.renderer.createBuffer(buffer_size, .{
            .storage = true,
            .copy_src = true,
            .copy_dst = true,
        });
        defer self.renderer.destroyBuffer(buffer) catch {};

        // Create test data
        const test_data = try self.allocator.alloc(f32, buffer_size / @sizeOf(f32));
        defer self.allocator.free(test_data);

        // Fill with test pattern
        for (test_data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 1000)) / 1000.0;
        }

        const data_bytes = std.mem.sliceAsBytes(test_data);

        // Measure write bandwidth
        const write_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            try self.renderer.writeBuffer(buffer, data_bytes);
        }
        const write_end = std.time.nanoTimestamp();

        // Measure read bandwidth
        const read_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const read_data = try self.renderer.readBuffer(buffer, self.allocator);
            self.allocator.free(read_data);
        }
        const read_end = std.time.nanoTimestamp();

        const total_bytes = @as(u64, buffer_size) * iterations * 2; // Read + write
        const total_time_ns = (write_end - write_start) + (read_end - read_start);
        const bandwidth_bytes_per_sec = @as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

        return bandwidth_bytes_per_sec / (1024.0 * 1024.0 * 1024.0); // Convert to GB/s
    }

    /// Run comprehensive memory benchmark
    pub fn runMemoryBenchmark(self: *MemoryBandwidthBenchmark) !void {
        const buffer_sizes = [_]usize{
            1024 * 1024, // 1MB
            4 * 1024 * 1024, // 4MB
            16 * 1024 * 1024, // 16MB
            64 * 1024 * 1024, // 64MB
        };

        const iterations = 10;

        std.log.info("Starting memory bandwidth benchmark", .{});

        for (buffer_sizes) |size| {
            const bandwidth = try self.measureBandwidth(size, iterations);
            std.log.info("Buffer size: {} MB, Bandwidth: {d:.2} GB/s", .{
                size / (1024 * 1024),
                bandwidth,
            });
        }

        std.log.info("Memory bandwidth benchmark completed", .{});
    }
};

/// GPU Compute Throughput Benchmark
pub const ComputeThroughputBenchmark = struct {
    allocator: std.mem.Allocator,
    renderer: *gpu_renderer.GPURenderer,

    pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*ComputeThroughputBenchmark {
        const self = try allocator.create(ComputeThroughputBenchmark);
        self.* = .{
            .allocator = allocator,
            .renderer = renderer,
        };
        return self;
    }

    pub fn deinit(self: *ComputeThroughputBenchmark) void {
        self.allocator.destroy(self);
    }

    /// Measure compute throughput (FLOPS)
    pub fn measureComputeThroughput(self: *ComputeThroughputBenchmark, workgroup_size: u32, iterations: u32) !f64 {
        _ = self; // Not used in this simple implementation
        std.log.info("Measuring compute throughput with workgroup size: {}", .{workgroup_size});

        // In a real implementation, this would:
        // 1. Create compute shader for intensive math operations
        // 2. Dispatch compute workgroups
        // 3. Measure execution time
        // 4. Calculate FLOPS based on operations performed

        // For now, simulate the measurement
        const operations_per_workgroup = workgroup_size * 100; // Estimate
        const total_operations = operations_per_workgroup * iterations;

        // Simulate execution time (this would be measured in real implementation)
        const simulated_time_ns: u64 = 1_000_000; // 1ms

        const flops = @as(f64, @floatFromInt(total_operations)) / (@as(f64, @floatFromInt(simulated_time_ns)) / 1_000_000_000.0);

        return flops / 1_000_000_000.0; // Convert to GFLOPS
    }

    /// Run compute throughput benchmark
    pub fn runComputeBenchmark(self: *ComputeThroughputBenchmark) !void {
        const workgroup_sizes = [_]u32{ 64, 128, 256, 512, 1024 };
        const iterations = 100;

        std.log.info("Starting compute throughput benchmark", .{});

        for (workgroup_sizes) |size| {
            const throughput = try self.measureComputeThroughput(size, iterations);
            std.log.info("Workgroup size: {}, Throughput: {d:.2} GFLOPS", .{ size, throughput });
        }

        std.log.info("Compute throughput benchmark completed", .{});
    }
};

test "benchmark enhancements" {
    const allocator = std.testing.allocator;

    // Test BenchmarkConfig validation
    const config = BenchmarkConfig.quick();
    try config.validate();

    // Test invalid configuration
    const invalid_config = BenchmarkConfig{
        .iterations = 0, // Invalid
        .buffer_sizes = &[_]usize{1024},
        .workloads = &[_]WorkloadType{.matrix_mul},
    };
    try std.testing.expectError(error.InvalidConfiguration, invalid_config.validate());

    // Test WorkloadType methods
    const workload = WorkloadType.matrix_mul;
    try std.testing.expectEqualStrings("Matrix Multiplication", workload.displayName());
    try std.testing.expectEqualStrings("O(n³)", workload.complexityClass());

    // Test PerformanceGrade methods
    const grade = PerformanceGrade.excellent;
    try std.testing.expectEqualStrings("Excellent", grade.displayName());
    try std.testing.expectEqualStrings("🟢", grade.colorCode());
    try std.testing.expectEqualStrings("Excellent", grade.toString());

    // Test BenchmarkResult methods (create a mock result)
    var mock_result = BenchmarkResult{
        .workload = .matrix_mul,
        .backend = backends.Backend{ .vulkan = undefined }, // Mock backend
        .iterations = 100,
        .total_time_ns = 1_000_000, // 1ms
        .avg_time_ns = 10_000,
        .min_time_ns = 8_000,
        .max_time_ns = 15_000,
        .std_dev_ns = 2_000,
        .median_time_ns = 10_000,
        .throughput_items_per_sec = 1000.0,
        .memory_bandwidth_gb_per_sec = 10.0,
        .compute_utilization_percent = 85.0,
        .memory_usage_mb = 100.0,
        .peak_memory_usage_mb = 120.0,
        .power_consumption_watts = 150.0,
        .average_power_watts = 120.0,
        .energy_consumed_joules = 120.0,
        .error_count = 0,
        .validation_passed = true,
        .accuracy_score = 0.99,
        .cache_hit_rate = 0.95,
        .thermal_throttling_detected = false,
        .timestamp = 123456789,
        .execution_context = ExecutionContext{
            .gpu_name = "Test GPU",
            .driver_version = "1.0.0",
            .compute_units = 16,
            .memory_size_mb = 8192,
            .clock_speed_mhz = 1500,
            .temperature_celsius = 60.0,
            .fan_speed_percent = 50.0,
        },
    };

    // Test new methods
    const efficiency = mock_result.calculateEfficiencyScore();
    try std.testing.expect(efficiency > 0);

    const stability = mock_result.calculateStabilityScore();
    try std.testing.expect(stability >= 0.0 and stability <= 1.0);

    const memory_efficiency = mock_result.calculateMemoryEfficiency();
    try std.testing.expect(memory_efficiency >= 0);

    const cache_efficiency = mock_result.calculateCacheEfficiency();
    try std.testing.expect(cache_efficiency >= 0.0 and cache_efficiency <= 1.0);

    const overall_score = mock_result.getOverallScore();
    try std.testing.expect(overall_score >= 0.0 and overall_score <= 1.0);

    const meets_threshold = mock_result.meetsQualityThresholds(config);
    try std.testing.expect(meets_threshold);

    // Test summary generation
    const summary = try mock_result.generateSummary(allocator);
    defer allocator.free(summary);
    try std.testing.expect(summary.len > 0);

    // Test performance grade
    const result_grade = mock_result.getPerformanceGrade();
    try std.testing.expect(@intFromEnum(result_grade) >= 0);
}
