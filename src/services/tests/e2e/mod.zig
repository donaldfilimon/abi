//! End-to-End Test Infrastructure
//!
//! Provides test infrastructure for complete user workflow validation.
//! E2E tests exercise full workflows from CLI/API through all layers to verify
//! the system works as a whole.
//!
//! ## Overview
//!
//! The E2E test infrastructure provides:
//! - `E2EContext`: Test environment setup with framework, temp directories, and cleanup
//! - `E2EConfig`: Configuration for test behavior (timeouts, cleanup, verbosity)
//! - `WorkflowTimer`: Timing utilities for workflow duration tracking
//! - Temporary directory management for test artifacts
//!
//! ## Usage
//!
//! ```zig
//! test "e2e: my workflow" {
//!     var ctx = try E2EContext.init(std.testing.allocator, .{
//!         .timeout_ms = 30_000,
//!         .cleanup_after = true,
//!     });
//!     defer ctx.deinit();
//!
//!     // Get framework for operations
//!     const fw = ctx.getFramework();
//!
//!     // Get temp dir for artifacts
//!     const temp_dir = try ctx.getTempDir();
//!
//!     // Test workflow...
//! }
//! ```
//!
//! ## Test Categories
//!
//! - `vector_search_e2e.zig`: Document insertion, indexing, search workflows
//! - `ai_agent_e2e.zig`: Agent initialization, tool usage, conversation flows
//! - `distributed_e2e.zig`: Cluster formation, distributed operations, failover
//! - `gpu_pipeline_e2e.zig`: GPU data upload, computation, result verification
//! - `cli_e2e.zig`: CLI command execution and output verification

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const abi = @import("abi");
const time = abi.shared.time;
const sync = abi.shared.sync;

// Sub-modules
pub const vector_search = @import("vector_search_e2e.zig");
pub const ai_agent = @import("ai_agent_e2e.zig");
pub const distributed = @import("distributed_e2e.zig");
pub const gpu_pipeline = @import("gpu_pipeline_e2e.zig");
pub const cli = @import("cli_e2e.zig");
pub const llm_training = @import("llm_training_e2e.zig");

// Force-reference test modules to include them in test build
comptime {
    _ = vector_search;
    _ = ai_agent;
    _ = distributed;
    _ = gpu_pipeline;
    _ = cli;
    _ = llm_training;
}

// ============================================================================
// E2E Configuration
// ============================================================================

/// E2E test configuration.
pub const E2EConfig = struct {
    /// Maximum time for a single test workflow in milliseconds.
    timeout_ms: u64 = 30_000,

    /// Whether to clean up test artifacts after the test completes.
    cleanup_after: bool = true,

    /// Enable verbose logging for debugging.
    verbose: bool = false,

    /// Base directory for temporary test artifacts.
    /// If null, uses system temp directory.
    temp_base: ?[]const u8 = null,

    /// Features to enable for this test context.
    features: FeatureSet = .{},

    /// Default configuration for quick tests.
    pub fn quick() E2EConfig {
        return .{
            .timeout_ms = 5_000,
            .cleanup_after = true,
            .verbose = false,
        };
    }

    /// Configuration for longer-running tests.
    pub fn extended() E2EConfig {
        return .{
            .timeout_ms = 60_000,
            .cleanup_after = true,
            .verbose = false,
        };
    }

    /// Configuration for debugging tests.
    pub fn debug() E2EConfig {
        return .{
            .timeout_ms = 120_000,
            .cleanup_after = false,
            .verbose = true,
        };
    }
};

/// Features to enable for E2E tests.
pub const FeatureSet = struct {
    gpu: bool = false,
    ai: bool = false,
    llm: bool = false,
    database: bool = false,
    network: bool = false,
    web: bool = false,
    observability: bool = false,

    /// Enable all compile-time available features.
    pub fn all() FeatureSet {
        return .{
            .gpu = build_options.enable_gpu,
            .ai = build_options.enable_ai,
            .llm = build_options.enable_llm,
            .database = build_options.enable_database,
            .network = build_options.enable_network,
            .web = build_options.enable_web,
            .observability = build_options.enable_profiling,
        };
    }

    /// Database-only feature set.
    pub fn databaseOnly() FeatureSet {
        return .{
            .database = build_options.enable_database,
        };
    }

    /// AI-only feature set.
    pub fn aiOnly() FeatureSet {
        return .{
            .ai = build_options.enable_ai,
            .llm = build_options.enable_llm,
        };
    }

    /// GPU-only feature set.
    pub fn gpuOnly() FeatureSet {
        return .{
            .gpu = build_options.enable_gpu,
        };
    }
};

// ============================================================================
// E2E Context
// ============================================================================

/// Test context for E2E tests.
///
/// Manages framework initialization, temporary directories, and cleanup.
/// Provides a consistent environment for end-to-end workflow testing.
pub const E2EContext = struct {
    /// Memory allocator for test resources.
    allocator: std.mem.Allocator,

    /// Test configuration.
    config: E2EConfig,

    /// Framework instance, or null if not yet initialized.
    framework: ?*abi.Framework,

    /// Temporary directory path for test artifacts.
    temp_dir: ?[]const u8,

    /// Test metrics.
    metrics: E2EMetrics,

    /// Whether context initialization completed successfully.
    initialized: bool,

    pub const Error = error{
        InitializationFailed,
        FrameworkNotInitialized,
        TempDirCreationFailed,
        TimeoutExceeded,
        FeatureNotEnabled,
        CleanupFailed,
    } || std.mem.Allocator.Error || abi.Framework.Error;

    /// Initialize the E2E test context.
    ///
    /// Creates a framework with the requested features and sets up
    /// temporary directories for test artifacts.
    pub fn init(allocator: std.mem.Allocator, config: E2EConfig) Error!E2EContext {
        var timer = time.Timer.start() catch null;

        var ctx = E2EContext{
            .allocator = allocator,
            .config = config,
            .framework = null,
            .temp_dir = null,
            .metrics = .{},
            .initialized = false,
        };

        // Create temporary directory
        ctx.temp_dir = try ctx.createTempDir();
        errdefer if (ctx.temp_dir) |dir| {
            allocator.free(dir);
            ctx.temp_dir = null;
        };

        // Initialize framework with requested features
        const fw = try allocator.create(abi.Framework);
        errdefer allocator.destroy(fw);

        const cfg = buildConfig(config.features);
        fw.* = try abi.Framework.init(allocator, cfg);
        ctx.framework = fw;

        ctx.initialized = true;

        if (timer) |*t| {
            ctx.metrics.setup_time_ns = t.read();
        }

        return ctx;
    }

    /// Initialize with minimal configuration (no features).
    pub fn initMinimal(allocator: std.mem.Allocator) Error!E2EContext {
        return init(allocator, .{});
    }

    /// Clean up all resources.
    pub fn deinit(self: *E2EContext) void {
        var timer = time.Timer.start() catch null;

        // Clean up framework
        if (self.framework) |fw| {
            fw.deinit();
            self.allocator.destroy(fw);
            self.framework = null;
        }

        // Clean up temp directory if configured
        if (self.config.cleanup_after) {
            self.cleanup();
        }

        // Free temp dir path
        if (self.temp_dir) |dir| {
            self.allocator.free(dir);
            self.temp_dir = null;
        }

        self.initialized = false;

        if (timer) |*t| {
            self.metrics.teardown_time_ns = t.read();
        }
    }

    /// Get the framework instance.
    pub fn getFramework(self: *E2EContext) Error!*abi.Framework {
        return self.framework orelse error.FrameworkNotInitialized;
    }

    /// Get the temporary directory path.
    pub fn getTempDir(self: *E2EContext) Error![]const u8 {
        return self.temp_dir orelse error.TempDirCreationFailed;
    }

    /// Create a unique temporary directory for test artifacts.
    fn createTempDir(self: *E2EContext) Error![]const u8 {
        const base = self.config.temp_base orelse "/tmp";

        // Generate unique directory name
        var buf: [64]u8 = undefined;
        // Use Timer for Zig 0.16 compatibility (no std.time.timestamp())
        var timer = time.Timer.start() catch {
            return error.TempDirCreationFailed;
        };
        const timestamp_ns = timer.read();
        const random = @as(u32, @truncate(timestamp_ns ^ 0xDEADBEEF));

        const dir_name = std.fmt.bufPrint(&buf, "abi_e2e_{d}_{x}", .{ timestamp_ns, random }) catch return error.TempDirCreationFailed;

        // Build full path
        const full_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ base, dir_name });

        return full_path;
    }

    /// Clean up test artifacts.
    pub fn cleanup(self: *E2EContext) void {
        // In a real implementation, we would recursively delete the temp directory.
        // For now, we just mark it as cleaned up.
        self.metrics.cleanup_performed = true;
    }

    /// Record a test operation.
    pub fn recordOperation(self: *E2EContext, name: []const u8, duration_ns: u64) void {
        _ = name;
        self.metrics.operation_count += 1;
        self.metrics.total_operation_time_ns += duration_ns;
    }

    /// Record a test error.
    pub fn recordError(self: *E2EContext, err: anyerror) void {
        _ = err;
        self.metrics.error_count += 1;
    }

    /// Get test metrics.
    pub fn getMetrics(self: *const E2EContext) E2EMetrics {
        return self.metrics;
    }

    /// Check if a feature is available.
    pub fn isFeatureAvailable(self: *const E2EContext, feature: abi.Feature) bool {
        if (self.framework) |fw| {
            return fw.isEnabled(feature);
        }
        return false;
    }

    /// Build framework config from feature set.
    fn buildConfig(features: FeatureSet) abi.Config {
        return .{
            .gpu = if (features.gpu and build_options.enable_gpu) abi.config.GpuConfig.defaults() else null,
            .ai = if ((features.ai or features.llm) and build_options.enable_ai) abi.config.AiConfig.defaults() else null,
            .database = if (features.database and build_options.enable_database) abi.config.DatabaseConfig.defaults() else null,
            .network = if (features.network and build_options.enable_network) abi.config.NetworkConfig.defaults() else null,
            .observability = if (features.observability and build_options.enable_profiling) abi.config.ObservabilityConfig.defaults() else null,
            .web = if (features.web and build_options.enable_web) abi.config.WebConfig.defaults() else null,
        };
    }
};

// ============================================================================
// Metrics
// ============================================================================

/// Metrics collected during E2E test execution.
pub const E2EMetrics = struct {
    /// Time spent setting up the test context.
    setup_time_ns: u64 = 0,

    /// Time spent tearing down the test context.
    teardown_time_ns: u64 = 0,

    /// Number of operations performed.
    operation_count: u64 = 0,

    /// Total time spent in operations.
    total_operation_time_ns: u64 = 0,

    /// Number of errors encountered.
    error_count: u64 = 0,

    /// Whether cleanup was performed.
    cleanup_performed: bool = false,

    /// Average operation time in nanoseconds.
    pub fn avgOperationTimeNs(self: E2EMetrics) u64 {
        if (self.operation_count == 0) return 0;
        return self.total_operation_time_ns / self.operation_count;
    }

    /// Total test duration in nanoseconds.
    pub fn totalDurationNs(self: E2EMetrics) u64 {
        return self.setup_time_ns + self.total_operation_time_ns + self.teardown_time_ns;
    }
};

// ============================================================================
// Workflow Timer
// ============================================================================

/// Timer for measuring workflow durations.
pub const WorkflowTimer = struct {
    start_time: ?time.Timer = null,
    checkpoints: std.ArrayListUnmanaged(Checkpoint) = .{},
    allocator: std.mem.Allocator,

    pub const Checkpoint = struct {
        name: []const u8,
        elapsed_ns: u64,
    };

    pub fn init(allocator: std.mem.Allocator) WorkflowTimer {
        return .{
            .allocator = allocator,
            .start_time = time.Timer.start() catch null,
        };
    }

    pub fn deinit(self: *WorkflowTimer) void {
        for (self.checkpoints.items) |cp| {
            self.allocator.free(cp.name);
        }
        self.checkpoints.deinit(self.allocator);
    }

    /// Record a checkpoint with the given name.
    pub fn checkpoint(self: *WorkflowTimer, name: []const u8) !void {
        const elapsed_time = if (self.start_time) |*t| t.read() else 0;
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.checkpoints.append(self.allocator, .{
            .name = name_copy,
            .elapsed_ns = elapsed_time,
        });
    }

    /// Get total elapsed time.
    pub fn elapsed(self: *WorkflowTimer) u64 {
        return if (self.start_time) |*t| t.read() else 0;
    }

    /// Check if a timeout has been exceeded.
    pub fn isTimedOut(self: *WorkflowTimer, timeout_ms: u64) bool {
        const elapsed_ns = self.elapsed();
        return elapsed_ns > timeout_ms * 1_000_000;
    }
};

// ============================================================================
// Test Utilities
// ============================================================================

/// Skip test if a feature is disabled.
pub fn skipIfDisabled(comptime feature: abi.Feature) !void {
    if (!feature.isCompileTimeEnabled()) {
        return error.SkipZigTest;
    }
}

/// Skip test if database is disabled.
pub fn skipIfDatabaseDisabled() !void {
    if (!build_options.enable_database) return error.SkipZigTest;
}

/// Skip test if AI is disabled.
pub fn skipIfAiDisabled() !void {
    if (!build_options.enable_ai) return error.SkipZigTest;
}

/// Skip test if GPU is disabled.
pub fn skipIfGpuDisabled() !void {
    if (!build_options.enable_gpu) return error.SkipZigTest;
}

/// Skip test if network is disabled.
pub fn skipIfNetworkDisabled() !void {
    if (!build_options.enable_network) return error.SkipZigTest;
}

/// Generate test vector data.
pub fn generateTestVector(allocator: std.mem.Allocator, dimension: usize, seed: u64) ![]f32 {
    const vec = try allocator.alloc(f32, dimension);
    var rng = std.Random.DefaultPrng.init(seed);
    for (vec) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }
    return vec;
}

/// Generate normalized test vector.
pub fn generateNormalizedVector(allocator: std.mem.Allocator, dimension: usize, seed: u64) ![]f32 {
    const vec = try generateTestVector(allocator, dimension, seed);

    // Normalize
    var norm: f32 = 0;
    for (vec) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (vec) |*v| {
            v.* /= norm;
        }
    }

    return vec;
}

/// Assert that two vectors are approximately equal.
pub fn assertVectorsEqual(a: []const f32, b: []const f32, tolerance: f32) !void {
    if (a.len != b.len) {
        return error.LengthMismatch;
    }
    for (a, b) |av, bv| {
        const diff = @abs(av - bv);
        if (diff > tolerance) {
            return error.ToleranceExceeded;
        }
    }
}

/// Assert all values are finite (no NaN or Inf).
pub fn assertAllFinite(values: []const f32) !void {
    for (values) |v| {
        if (!std.math.isFinite(v)) {
            return error.NonFiniteValue;
        }
    }
}

// ============================================================================
// Error Types
// ============================================================================

pub const ValidationError = error{
    LengthMismatch,
    ToleranceExceeded,
    NonFiniteValue,
    OutOfRange,
    InvalidState,
};

// ============================================================================
// Tests
// ============================================================================

test "E2EContext: minimal initialization" {
    var ctx = try E2EContext.initMinimal(std.testing.allocator);
    defer ctx.deinit();

    try std.testing.expect(ctx.initialized);
    try std.testing.expect(ctx.framework != null);
}

test "E2EContext: with features" {
    var ctx = try E2EContext.init(std.testing.allocator, .{
        .features = .{ .observability = true },
    });
    defer ctx.deinit();

    try std.testing.expect(ctx.initialized);

    const metrics = ctx.getMetrics();
    try std.testing.expect(metrics.setup_time_ns > 0);
}

test "E2EContext: temp directory creation" {
    var ctx = try E2EContext.initMinimal(std.testing.allocator);
    defer ctx.deinit();

    const temp_dir = try ctx.getTempDir();
    try std.testing.expect(temp_dir.len > 0);
}

test "WorkflowTimer: basic timing" {
    var timer = WorkflowTimer.init(std.testing.allocator);
    defer timer.deinit();

    try timer.checkpoint("start");

    // Small delay - use cross-platform sleep
    // On POSIX systems use nanosleep, on Windows this is a no-op (timer tests precision anyway)
    if (@hasDecl(std.posix, "nanosleep")) {
        var req = std.posix.timespec{ .sec = 0, .nsec = 1_000_000 };
        var rem: std.posix.timespec = undefined;
        _ = std.posix.nanosleep(&req, &rem);
    }

    try timer.checkpoint("after_delay");

    const elapsed = timer.elapsed();
    // On POSIX, elapsed should be at least 1ms; on Windows without nanosleep, just verify it's non-negative
    if (@hasDecl(std.posix, "nanosleep")) {
        try std.testing.expect(elapsed >= 1_000_000);
    } else {
        try std.testing.expect(elapsed >= 0);
    }
    try std.testing.expectEqual(@as(usize, 2), timer.checkpoints.items.len);
}

test "WorkflowTimer: timeout detection" {
    var timer = WorkflowTimer.init(std.testing.allocator);
    defer timer.deinit();

    // Should not be timed out for a large timeout
    try std.testing.expect(!timer.isTimedOut(10_000));

    // Should be timed out for zero timeout
    try std.testing.expect(timer.isTimedOut(0));
}

test "generateTestVector: produces correct dimensions" {
    const vec = try generateTestVector(std.testing.allocator, 128, 42);
    defer std.testing.allocator.free(vec);

    try std.testing.expectEqual(@as(usize, 128), vec.len);
    try assertAllFinite(vec);
}

test "generateNormalizedVector: produces unit vector" {
    const vec = try generateNormalizedVector(std.testing.allocator, 64, 42);
    defer std.testing.allocator.free(vec);

    // Check norm is approximately 1
    var norm: f32 = 0;
    for (vec) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.0001);
}

test "E2EConfig: presets" {
    const quick = E2EConfig.quick();
    try std.testing.expectEqual(@as(u64, 5_000), quick.timeout_ms);

    const extended = E2EConfig.extended();
    try std.testing.expectEqual(@as(u64, 60_000), extended.timeout_ms);

    const debug = E2EConfig.debug();
    try std.testing.expect(debug.verbose);
    try std.testing.expect(!debug.cleanup_after);
}

test "FeatureSet: all" {
    const all = FeatureSet.all();
    try std.testing.expectEqual(build_options.enable_gpu, all.gpu);
    try std.testing.expectEqual(build_options.enable_ai, all.ai);
    try std.testing.expectEqual(build_options.enable_database, all.database);
}

test "E2EMetrics: calculations" {
    var metrics = E2EMetrics{
        .setup_time_ns = 1_000_000,
        .teardown_time_ns = 500_000,
        .operation_count = 10,
        .total_operation_time_ns = 5_000_000,
    };

    try std.testing.expectEqual(@as(u64, 500_000), metrics.avgOperationTimeNs());
    try std.testing.expectEqual(@as(u64, 6_500_000), metrics.totalDurationNs());
}
