const std = @import("std");
const gpu_renderer = @import("gpu_renderer");
const backends = @import("backends");
const math = @import("math");
const crypto = @import("crypto");

fn initArrayList(comptime T: type, allocator: std.mem.Allocator) !std.ArrayList(T) {
    var list = try std.ArrayList(T).initCapacity(allocator, 0);
    errdefer list.deinit(allocator);
    return list;
}

pub const BenchmarkConfig = struct {
    iterations: u32,
    warmup_iterations: u32,
    buffer_sizes: []const usize,
    workloads: []const WorkloadType,
    timeout_ms: u32,
    validate_results: bool,
    validate_accuracy: bool,

    pub fn validate(self: BenchmarkConfig) !void {
        if (self.iterations == 0) return error.InvalidConfiguration;
        if (self.buffer_sizes.len == 0) return error.InvalidConfiguration;
        if (self.workloads.len == 0) return error.InvalidConfiguration;
        if (self.timeout_ms == 0) return error.InvalidConfiguration;
    }
};

pub const WorkloadType = enum {
    matrix_mul,
    vector_add,
    convolution,
    pooling,
    reduction,
    scan,
    sort,
    fft,
    activation,
    normalization,
    attention,
    sparse_operations,

    pub fn displayName(self: WorkloadType) []const u8 {
        return switch (self) {
            .matrix_mul => "Matrix Multiplication",
            .vector_add => "Vector Addition",
            .convolution => "Convolution",
            .pooling => "Pooling",
            .reduction => "Reduction",
            .scan => "Scan",
            .sort => "Sort",
            .fft => "FFT",
            .activation => "Activation",
            .normalization => "Normalization",
            .attention => "Attention",
            .sparse_operations => "Sparse Operations",
        };
    }

    pub fn complexityClass(self: WorkloadType) []const u8 {
        return switch (self) {
            .matrix_mul => "O(n^3)",
            .vector_add => "O(n)",
            .convolution => "O(n^2)",
            .pooling => "O(n)",
            .reduction => "O(n)",
            .scan => "O(n)",
            .sort => "O(n log n)",
            .fft => "O(n log n)",
            .activation => "O(n)",
            .normalization => "O(n)",
            .attention => "O(n^2)",
            .sparse_operations => "O(k)",
        };
    }
};

pub const PerformanceGrade = enum {
    excellent,
    good,
    average,
    poor,

    pub fn displayName(self: PerformanceGrade) []const u8 {
        return switch (self) {
            .excellent => "Excellent",
            .good => "Good",
            .average => "Average",
            .poor => "Poor",
        };
    }

    pub fn colorCode(self: PerformanceGrade) []const u8 {
        return switch (self) {
            .excellent => "🟢",
            .good => "🟡",
            .average => "🟠",
            .poor => "🔴",
        };
    }

    pub fn toString(self: PerformanceGrade) []const u8 {
        return self.displayName();
    }
};

pub const ExecutionContext = struct {
    gpu_name: []const u8,
    driver_version: []const u8,
    compute_units: u32,
    memory_size_mb: u32,
    clock_speed_mhz: u32,
    temperature_celsius: f32,
    fan_speed_percent: f32,
};

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
    timestamp: u64,
    execution_context: ExecutionContext,
};

test "benchmark enhancements" {
    const allocator = std.testing.allocator;

    // Test BenchmarkConfig validation
    const config = BenchmarkConfig{
        .iterations = 10,
        .buffer_sizes = &[_]usize{ 1024 * 1024, 4 * 1024 * 1024 },
        .workloads = &[_]WorkloadType{.matrix_mul},
        .warmup_iterations = 2,
        .timeout_ms = 5000,
        .validate_results = true,
        .validate_accuracy = true,
    };
    try config.validate();

    // Test invalid configuration
    const invalid_config = BenchmarkConfig{
        .iterations = 0,
        .buffer_sizes = &[_]usize{1024},
        .workloads = &[_]WorkloadType{.matrix_mul},
        .warmup_iterations = 2,
        .timeout_ms = 5000,
        .validate_results = true,
        .validate_accuracy = true,
    };
    try std.testing.expectError(error.InvalidConfiguration, invalid_config.validate());

    // Test WorkloadType methods
    const workload = WorkloadType.matrix_mul;
    try std.testing.expectEqualStrings("Matrix Multiplication", workload.displayName());
    try std.testing.expectEqualStrings("O(n^3)", workload.complexityClass());

    // Test PerformanceGrade methods
    const grade = PerformanceGrade.excellent;
    try std.testing.expectEqualStrings("Excellent", grade.displayName());
    try std.testing.expectEqualStrings("🟢", grade.colorCode());
    try std.testing.expectEqualStrings("Excellent", grade.toString());

    // Test BenchmarkResult methods (create a mock result)
    var mock_result = BenchmarkResult{
        .workload = .matrix_mul,
        .backend = backends.Backend{ .vulkan = undefined },
        .iterations = 100,
        .total_time_ns = 1_000_000,
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
