//! Integration Test Fixture Infrastructure
//!
//! Provides reusable test fixtures for cross-module integration testing.
//! Features:
//! - Hardware detection with automatic mock fallback
//! - Consistent test environment setup/teardown
//! - Common assertions and utilities
//! - Resource lifecycle management
//!
//! ## Usage
//!
//! ```zig
//! test "integration test example" {
//!     var fixture = try IntegrationFixture.init(std.testing.allocator, .{
//!         .gpu = true,
//!         .database = true,
//!     });
//!     defer fixture.deinit();
//!
//!     // Test with real or mocked contexts
//!     const gpu = fixture.getGpuContext();
//!     const db = fixture.getDatabaseContext();
//! }
//! ```

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const time = abi.services.shared.time;
const sync = abi.services.shared.sync;
const build_options = @import("build_options");

/// Hardware detection mode
pub const HardwareMode = enum {
    /// Use real hardware when available
    real,
    /// Always use mocks
    mock,
    /// Use real where available, mock otherwise
    hybrid,
};

/// Feature set configuration for fixtures
pub const FeatureSet = struct {
    gpu: bool = false,
    ai: bool = false,
    llm: bool = false,
    database: bool = false,
    network: bool = false,
    web: bool = false,
    observability: bool = false,
    ha: bool = false,
};

/// Test metrics collected during fixture execution
pub const TestMetrics = struct {
    setup_time_ns: u64 = 0,
    teardown_time_ns: u64 = 0,
    operation_count: u64 = 0,
    error_count: u64 = 0,
    memory_allocated: usize = 0,
    memory_freed: usize = 0,

    pub fn recordOperation(self: *TestMetrics) void {
        self.operation_count += 1;
    }

    pub fn recordError(self: *TestMetrics) void {
        self.error_count += 1;
    }
};

/// Accuracy statistics for numerical validation
pub const AccuracyStats = struct {
    max_error: f32 = 0,
    mean_error: f32 = 0,
    rmse: f32 = 0,
    sample_count: usize = 0,

    pub fn update(self: *AccuracyStats, gpu_val: f32, cpu_val: f32) void {
        const error_abs = @abs(gpu_val - cpu_val);
        const error_rel = if (cpu_val != 0) error_abs / @abs(cpu_val) else error_abs;

        self.max_error = @max(self.max_error, error_rel);
        const n = @as(f32, @floatFromInt(self.sample_count));
        const n1 = @as(f32, @floatFromInt(self.sample_count + 1));
        self.mean_error = (self.mean_error * n + error_rel) / n1;
        self.rmse = @sqrt((self.rmse * self.rmse * n + error_rel * error_rel) / n1);
        self.sample_count += 1;
    }
};

/// Integration test fixture providing consistent test environment
pub const IntegrationFixture = struct {
    allocator: std.mem.Allocator,
    features: FeatureSet,
    hardware_mode: HardwareMode,
    metrics: TestMetrics,
    framework: ?*abi.App,
    setup_complete: bool,

    // Hardware availability cache
    gpu_available: bool,
    cuda_available: bool,

    pub const Error = error{
        SetupFailed,
        FeatureNotEnabled,
        HardwareNotAvailable,
    } || std.mem.Allocator.Error || abi.App.Error;

    /// Initialize fixture with requested features
    pub fn init(allocator: std.mem.Allocator, features: FeatureSet) Error!IntegrationFixture {
        return initWithMode(allocator, features, .hybrid);
    }

    /// Initialize with explicit hardware mode
    pub fn initWithMode(
        allocator: std.mem.Allocator,
        features: FeatureSet,
        hardware_mode: HardwareMode,
    ) Error!IntegrationFixture {
        var timer = time.Timer.start() catch null;

        var fixture = IntegrationFixture{
            .allocator = allocator,
            .features = features,
            .hardware_mode = hardware_mode,
            .metrics = .{},
            .framework = null,
            .setup_complete = false,
            .gpu_available = false,
            .cuda_available = false,
        };

        // Detect hardware availability
        fixture.detectHardware();

        // Initialize framework if any features requested
        if (features.gpu or features.ai or features.llm or
            features.database or features.network or features.web or
            features.observability or features.ha)
        {
            const fw = try allocator.create(abi.App);
            // Build config with requested features
            const cfg = abi.Config{
                .gpu = if (features.gpu and build_options.enable_gpu) abi.config.GpuConfig.defaults() else null,
                .ai = if ((features.ai or features.llm) and build_options.enable_ai) abi.config.AiConfig.defaults() else null,
                .database = if (features.database and build_options.enable_database) abi.config.DatabaseConfig.defaults() else null,
                .network = if (features.network and build_options.enable_network) abi.config.NetworkConfig.defaults() else null,
                .observability = if (features.observability and build_options.enable_profiling) abi.config.ObservabilityConfig.defaults() else null,
                .web = if (features.web and build_options.enable_web) abi.config.WebConfig.defaults() else null,
            };
            fw.* = try abi.App.init(allocator, cfg);
            fixture.framework = fw;
        }

        fixture.setup_complete = true;

        if (timer) |*t| {
            fixture.metrics.setup_time_ns = t.read();
        }

        return fixture;
    }

    /// Clean up all resources
    pub fn deinit(self: *IntegrationFixture) void {
        var timer = time.Timer.start() catch null;

        if (self.framework) |fw| {
            fw.deinit();
            self.allocator.destroy(fw);
            self.framework = null;
        }

        self.setup_complete = false;

        if (timer) |*t| {
            self.metrics.teardown_time_ns = t.read();
        }
    }

    /// Detect available hardware
    fn detectHardware(self: *IntegrationFixture) void {
        // GPU detection
        self.gpu_available = build_options.enable_gpu;

        // CUDA detection - simplified check since gpu_backends is a slice
        self.cuda_available = build_options.enable_gpu;
    }

    /// Check if specific hardware is available
    pub fn isHardwareAvailable(self: *const IntegrationFixture, feature: FeatureSet) bool {
        if (self.hardware_mode == .mock) return false;

        if (feature.gpu and !self.gpu_available) return false;
        if (feature.database and !build_options.enable_database) return false;
        if (feature.ai and !build_options.enable_ai) return false;
        if (feature.network and !build_options.enable_network) return false;
        if (feature.web and !build_options.enable_web) return false;

        return true;
    }

    /// Get framework reference (fails if not initialized)
    pub fn getFramework(self: *IntegrationFixture) Error!*abi.App {
        return self.framework orelse error.SetupFailed;
    }

    /// Check if a feature is enabled in this fixture
    pub fn isEnabled(self: *const IntegrationFixture, comptime feature: std.meta.FieldEnum(FeatureSet)) bool {
        return @field(self.features, @tagName(feature));
    }

    /// Get test metrics
    pub fn getMetrics(self: *const IntegrationFixture) TestMetrics {
        return self.metrics;
    }
};

// ============================================================================
// Common Assertion Helpers
// ============================================================================

/// Validate numerical accuracy between GPU and CPU results
pub fn validateAccuracy(
    gpu_result: []const f32,
    cpu_result: []const f32,
    tolerance: f32,
    stats: *AccuracyStats,
) !void {
    if (gpu_result.len != cpu_result.len) {
        return error.LengthMismatch;
    }

    for (gpu_result, cpu_result) |gpu_val, cpu_val| {
        const error_abs = @abs(gpu_val - cpu_val);
        const error_rel = if (cpu_val != 0) error_abs / @abs(cpu_val) else error_abs;

        if (error_rel > tolerance) {
            return error.ToleranceExceeded;
        }

        stats.update(gpu_val, cpu_val);
    }
}

/// Check that all values are finite (no NaN or Inf)
pub fn assertAllFinite(values: []const f32) !void {
    for (values) |v| {
        if (!std.math.isFinite(v)) {
            return error.NonFiniteValue;
        }
    }
}

/// Check that values are within expected range
pub fn assertInRange(values: []const f32, min: f32, max: f32) !void {
    for (values) |v| {
        if (v < min or v > max) {
            return error.OutOfRange;
        }
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

/// Patterns for generating test data
pub const VectorPattern = enum {
    zeros,
    ones,
    sequential,
    alternating,
    random,
    gaussian,
};

/// Generate test vector with specified pattern
pub fn generateTestVector(
    comptime T: type,
    pattern: VectorPattern,
    len: usize,
    allocator: std.mem.Allocator,
) ![]T {
    const vec = try allocator.alloc(T, len);
    errdefer allocator.free(vec);

    switch (pattern) {
        .zeros => @memset(vec, 0),
        .ones => @memset(vec, 1),
        .sequential => {
            for (vec, 0..) |*v, i| {
                v.* = @floatFromInt(i);
            }
        },
        .alternating => {
            for (vec, 0..) |*v, i| {
                v.* = if (i % 2 == 0) @as(T, 1) else @as(T, -1);
            }
        },
        .random, .gaussian => {
            var rng = std.Random.DefaultPrng.init(42);
            for (vec) |*v| {
                v.* = rng.random().float(T) * 2.0 - 1.0;
            }
        },
    }

    return vec;
}

/// Generate random matrix (row-major)
pub fn generateRandomMatrix(
    comptime T: type,
    rows: usize,
    cols: usize,
    seed: u64,
    allocator: std.mem.Allocator,
) ![]T {
    const mat = try allocator.alloc(T, rows * cols);
    errdefer allocator.free(mat);

    var rng = std.Random.DefaultPrng.init(seed);
    for (mat) |*v| {
        v.* = rng.random().float(T) * 2.0 - 1.0;
    }

    return mat;
}

// ============================================================================
// Error Definitions
// ============================================================================

pub const ValidationError = error{
    LengthMismatch,
    ToleranceExceeded,
    NonFiniteValue,
    OutOfRange,
};

// ============================================================================
// Tests
// ============================================================================

test "fixture initialization" {
    var fixture = try IntegrationFixture.init(std.testing.allocator, .{});
    defer fixture.deinit();

    try std.testing.expect(fixture.setup_complete);
}

test "fixture with features" {
    var fixture = try IntegrationFixture.init(std.testing.allocator, .{
        .observability = true,
    });
    defer fixture.deinit();

    try std.testing.expect(fixture.setup_complete);
    try std.testing.expect(fixture.isEnabled(.observability));
}

test "generate test vector patterns" {
    const allocator = std.testing.allocator;

    // Zeros
    const zeros = try generateTestVector(f32, .zeros, 10, allocator);
    defer allocator.free(zeros);
    for (zeros) |v| try std.testing.expectEqual(@as(f32, 0), v);

    // Ones
    const ones = try generateTestVector(f32, .ones, 10, allocator);
    defer allocator.free(ones);
    for (ones) |v| try std.testing.expectEqual(@as(f32, 1), v);

    // Sequential
    const seq = try generateTestVector(f32, .sequential, 5, allocator);
    defer allocator.free(seq);
    try std.testing.expectEqual(@as(f32, 0), seq[0]);
    try std.testing.expectEqual(@as(f32, 4), seq[4]);
}

test "accuracy validation" {
    const gpu = [_]f32{ 1.0, 2.0, 3.0 };
    const cpu = [_]f32{ 1.0, 2.0, 3.0 };

    var stats = AccuracyStats{};
    try validateAccuracy(&gpu, &cpu, 0.01, &stats);

    try std.testing.expectEqual(@as(f32, 0), stats.max_error);
}

test "assert all finite" {
    const good = [_]f32{ 1.0, 2.0, 3.0 };
    try assertAllFinite(&good);

    const bad = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    try std.testing.expectError(error.NonFiniteValue, assertAllFinite(&bad));
}
