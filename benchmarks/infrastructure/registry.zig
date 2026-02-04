//! Feature Registry Benchmarks
//!
//! Performance benchmarks for the feature registration and query system:
//! - Registration overhead (comptime vs runtime modes)
//! - Feature query latency (isEnabled checks)
//! - Enable/disable toggle performance
//! - Registry initialization time
//! - Memory overhead per feature

const std = @import("std");
const framework = @import("../system/framework.zig");

/// Registry benchmark configuration
pub const RegistryBenchConfig = struct {
    /// Number of features to register
    feature_counts: []const usize = &.{ 8, 16, 32, 64 },
    /// Number of queries per benchmark iteration
    queries_per_iteration: usize = 10000,
    /// Number of toggle operations per iteration
    toggles_per_iteration: usize = 1000,
    /// Minimum benchmark time
    min_time_ns: u64 = 100_000_000,
    /// Warmup iterations
    warmup_iterations: usize = 100,

    pub const quick = RegistryBenchConfig{
        .feature_counts = &.{ 8, 16 },
        .queries_per_iteration = 1000,
        .toggles_per_iteration = 100,
        .min_time_ns = 50_000_000,
        .warmup_iterations = 50,
    };

    pub const standard = RegistryBenchConfig{
        .feature_counts = &.{ 8, 16, 32 },
        .queries_per_iteration = 5000,
        .toggles_per_iteration = 500,
        .min_time_ns = 100_000_000,
        .warmup_iterations = 100,
    };

    pub const comprehensive = RegistryBenchConfig{
        .feature_counts = &.{ 8, 16, 32, 64 },
        .queries_per_iteration = 10000,
        .toggles_per_iteration = 1000,
        .min_time_ns = 200_000_000,
        .warmup_iterations = 200,
    };
};

// ============================================================================
// Mock Registry Implementation
// ============================================================================

/// Simplified registry for benchmark isolation.
const MockRegistry = struct {
    allocator: std.mem.Allocator,
    features: std.AutoHashMapUnmanaged(u32, FeatureState),
    registration_count: u64,
    query_count: u64,
    toggle_count: u64,

    const FeatureState = struct {
        enabled: bool,
        mode: RegistrationMode,
    };

    const RegistrationMode = enum {
        comptime_only,
        runtime_toggle,
        dynamic,
    };

    pub fn init(allocator: std.mem.Allocator) MockRegistry {
        return .{
            .allocator = allocator,
            .features = .{},
            .registration_count = 0,
            .query_count = 0,
            .toggle_count = 0,
        };
    }

    pub fn deinit(self: *MockRegistry) void {
        self.features.deinit(self.allocator);
    }

    pub fn register(self: *MockRegistry, feature_id: u32, mode: RegistrationMode) !void {
        try self.features.put(self.allocator, feature_id, .{
            .enabled = true,
            .mode = mode,
        });
        self.registration_count += 1;
    }

    pub fn isEnabled(self: *MockRegistry, feature_id: u32) bool {
        self.query_count += 1;
        if (self.features.get(feature_id)) |state| {
            return state.enabled;
        }
        return false;
    }

    pub fn setEnabled(self: *MockRegistry, feature_id: u32, enabled: bool) bool {
        self.toggle_count += 1;
        if (self.features.getPtr(feature_id)) |state| {
            if (state.mode == .runtime_toggle or state.mode == .dynamic) {
                state.enabled = enabled;
                return true;
            }
        }
        return false;
    }

    pub fn clear(self: *MockRegistry) void {
        self.features.clearRetainingCapacity();
        self.registration_count = 0;
        self.query_count = 0;
        self.toggle_count = 0;
    }
};

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark feature registration
fn benchRegistration(allocator: std.mem.Allocator, count: usize) u64 {
    var registry = MockRegistry.init(allocator);
    defer registry.deinit();

    for (0..count) |i| {
        const mode: MockRegistry.RegistrationMode = switch (i % 3) {
            0 => .comptime_only,
            1 => .runtime_toggle,
            else => .dynamic,
        };
        registry.register(@intCast(i), mode) catch {};
    }

    return registry.registration_count;
}

/// Benchmark feature queries (isEnabled checks)
fn benchFeatureQuery(registry: *MockRegistry, queries: usize) u64 {
    var enabled_count: u64 = 0;
    for (0..queries) |i| {
        const feature_id: u32 = @intCast(i % 32); // Query subset of features
        if (registry.isEnabled(feature_id)) {
            enabled_count += 1;
        }
    }
    return enabled_count;
}

/// Benchmark runtime toggle operations
fn benchFeatureToggle(registry: *MockRegistry, toggles: usize) u64 {
    var success_count: u64 = 0;
    for (0..toggles) |i| {
        const feature_id: u32 = @intCast(i % 32);
        const enable = (i % 2) == 0;
        if (registry.setEnabled(feature_id, enable)) {
            success_count += 1;
        }
    }
    return success_count;
}

/// Benchmark mixed workload (queries with occasional toggles)
fn benchMixedWorkload(registry: *MockRegistry, ops: usize) u64 {
    var sum: u64 = 0;
    for (0..ops) |i| {
        const feature_id: u32 = @intCast(i % 32);

        // 95% queries, 5% toggles
        if (i % 20 == 0) {
            if (registry.setEnabled(feature_id, (i % 2) == 0)) {
                sum += 1;
            }
        } else {
            if (registry.isEnabled(feature_id)) {
                sum += 1;
            }
        }
    }
    return sum;
}

// ============================================================================
// Public Benchmark API
// ============================================================================

/// Run all registry benchmarks.
pub fn runAllBenchmarks(allocator: std.mem.Allocator, config: RegistryBenchConfig) !void {
    std.debug.print("\n=== Feature Registry Benchmarks ===\n\n", .{});

    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Benchmark 1: Registration overhead
    std.debug.print("--- Registration Overhead ---\n", .{});
    for (config.feature_counts) |count| {
        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("register_{d}", .{count}),
                .category = "registry",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchRegistration,
            .{ allocator, count },
        );

        const ns_per_feature = result.stats.mean_ns / @as(f64, @floatFromInt(count));
        std.debug.print("  {d} features: {d:.1}ns/feature, total={d:.0}ns\n", .{
            count,
            ns_per_feature,
            result.stats.mean_ns,
        });
    }

    // Benchmark 2: Feature query latency
    std.debug.print("\n--- Feature Query Latency (isEnabled) ---\n", .{});
    for (config.feature_counts) |count| {
        var registry = MockRegistry.init(allocator);
        defer registry.deinit();

        // Pre-register features
        for (0..count) |i| {
            try registry.register(@intCast(i), .runtime_toggle);
        }

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("query_{d}features", .{count}),
                .category = "registry",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchFeatureQuery,
            .{ &registry, config.queries_per_iteration },
        );

        const ns_per_query = result.stats.mean_ns / @as(f64, @floatFromInt(config.queries_per_iteration));
        std.debug.print("  {d} features: {d:.2}ns/query\n", .{ count, ns_per_query });
    }

    // Benchmark 3: Toggle performance
    std.debug.print("\n--- Runtime Toggle Performance ---\n", .{});
    for (config.feature_counts) |count| {
        var registry = MockRegistry.init(allocator);
        defer registry.deinit();

        // Pre-register as runtime_toggle
        for (0..count) |i| {
            try registry.register(@intCast(i), .runtime_toggle);
        }

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("toggle_{d}features", .{count}),
                .category = "registry",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchFeatureToggle,
            .{ &registry, config.toggles_per_iteration },
        );

        const ns_per_toggle = result.stats.mean_ns / @as(f64, @floatFromInt(config.toggles_per_iteration));
        std.debug.print("  {d} features: {d:.2}ns/toggle\n", .{ count, ns_per_toggle });
    }

    // Benchmark 4: Mixed workload
    std.debug.print("\n--- Mixed Workload (95% query, 5% toggle) ---\n", .{});
    {
        var registry = MockRegistry.init(allocator);
        defer registry.deinit();

        for (0..32) |i| {
            try registry.register(@intCast(i), .runtime_toggle);
        }

        const result = try runner.run(
            .{
                .name = "mixed_workload",
                .category = "registry",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchMixedWorkload,
            .{ &registry, config.queries_per_iteration },
        );

        const ns_per_op = result.stats.mean_ns / @as(f64, @floatFromInt(config.queries_per_iteration));
        std.debug.print("  {d:.2}ns/op (p99={d}ns)\n", .{ ns_per_op, result.stats.p99_ns });
    }

    std.debug.print("\n=== Feature Registry Benchmarks Complete ===\n", .{});
}

/// Run quick benchmarks for CI.
pub fn runQuickBenchmarks(allocator: std.mem.Allocator) !void {
    try runAllBenchmarks(allocator, RegistryBenchConfig.quick);
}

// ============================================================================
// Tests
// ============================================================================

test "mock registry basic operations" {
    const allocator = std.testing.allocator;
    var registry = MockRegistry.init(allocator);
    defer registry.deinit();

    // Register
    try registry.register(1, .runtime_toggle);
    try std.testing.expect(registry.isEnabled(1));

    // Toggle
    try std.testing.expect(registry.setEnabled(1, false));
    try std.testing.expect(!registry.isEnabled(1));

    // Not found
    try std.testing.expect(!registry.isEnabled(999));
}

test "comptime_only cannot be toggled" {
    const allocator = std.testing.allocator;
    var registry = MockRegistry.init(allocator);
    defer registry.deinit();

    try registry.register(1, .comptime_only);
    try std.testing.expect(registry.isEnabled(1));

    // Should fail to toggle comptime_only
    try std.testing.expect(!registry.setEnabled(1, false));
    try std.testing.expect(registry.isEnabled(1)); // Still enabled
}

test "benchmark functions compile" {
    const allocator = std.testing.allocator;

    _ = benchRegistration(allocator, 10);

    var registry = MockRegistry.init(allocator);
    defer registry.deinit();

    for (0..10) |i| {
        try registry.register(@intCast(i), .runtime_toggle);
    }

    _ = benchFeatureQuery(&registry, 100);
    _ = benchFeatureToggle(&registry, 50);
    _ = benchMixedWorkload(&registry, 100);
}
