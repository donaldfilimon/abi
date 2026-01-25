//! Stress Test Infrastructure
//!
//! Production-grade stress testing for the ABI framework. This module provides
//! comprehensive stress tests for all major components with configurable
//! intensity profiles.
//!
//! ## Features
//!
//! - **Configurable Profiles**: Light, medium, heavy, and extreme intensity levels
//! - **HA Module Tests**: Backup, PITR, replication under load
//! - **Observability Tests**: High-throughput metrics, tracing, memory stability
//! - **Database Tests**: Vector operations, concurrent access, index stress
//!
//! ## Usage
//!
//! ```bash
//! # Run all stress tests
//! zig test src/tests/stress/mod.zig --summary all
//!
//! # Run specific stress test category
//! zig test src/tests/stress/ha_stress_test.zig --test-filter "ha stress"
//! zig test src/tests/stress/observability_stress_test.zig --test-filter "observability stress"
//! zig test src/tests/stress/database_stress_test.zig --test-filter "database stress"
//!
//! # Run with specific profile (via code modification or environment)
//! # See profiles.zig for available profiles
//! ```
//!
//! ## Test Categories
//!
//! ### HA Stress Tests
//! - Backup under concurrent write load
//! - Backup chain integrity with 1000+ backups
//! - PITR rapid checkpoint creation
//! - Concurrent PITR operations
//! - Replication with many replicas
//! - Failover timing
//!
//! ### Observability Stress Tests
//! - Counter/Gauge high throughput
//! - Histogram with millions of samples
//! - MetricsCollector concurrent registration
//! - Tracer concurrent span creation
//! - Memory stability under long-running collection
//!
//! ### Database Stress Tests
//! - Vector insertion throughput
//! - Batch insertion
//! - Concurrent search and write
//! - Search result quality under load
//! - Rapid updates and deletes
//! - Index operations under pressure
//! - Clustering and quantization

const std = @import("std");
const build_options = @import("build_options");

// Core infrastructure
pub const profiles = @import("profiles.zig");

// Re-export key types
pub const StressProfile = profiles.StressProfile;
pub const StressResult = profiles.StressResult;
pub const LatencyHistogram = profiles.LatencyHistogram;
pub const LatencyStats = profiles.LatencyStats;
pub const Timer = profiles.Timer;

// Utility functions
pub const sleepMs = profiles.sleepMs;
pub const getActiveProfile = profiles.getActiveProfile;
pub const getProfileByName = profiles.getProfileByName;

// Force-reference test modules to include them in test build
comptime {
    // Always include profiles infrastructure
    _ = profiles;

    // HA stress tests
    _ = @import("ha_stress_test.zig");

    // Observability stress tests (when profiling enabled)
    if (build_options.enable_profiling) {
        _ = @import("observability_stress_test.zig");
    }

    // Database stress tests (when database enabled)
    if (build_options.enable_database) {
        _ = @import("database_stress_test.zig");
    }
}

// ============================================================================
// Stress Test Runner
// ============================================================================

/// Run a stress test function with the given profile and collect results
pub fn runStressTest(
    comptime name: []const u8,
    profile: StressProfile,
    allocator: std.mem.Allocator,
    comptime test_fn: fn (StressProfile, std.mem.Allocator) anyerror!void,
) StressResult {
    const timer = Timer.start();

    var passed = true;
    var error_message: ?[]const u8 = null;

    test_fn(profile, allocator) catch |err| {
        passed = false;
        error_message = @errorName(err);
    };

    const duration_ns = timer.read();

    return .{
        .profile_name = name,
        .operations_completed = profile.operations,
        .operations_failed = if (passed) 0 else 1,
        .duration_ns = duration_ns,
        .peak_memory_bytes = 0, // TODO: Track peak memory
        .ops_per_second = @as(f64, @floatFromInt(profile.operations)) / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0),
        .avg_latency_ns = duration_ns / profile.operations,
        .p50_latency_ns = 0,
        .p95_latency_ns = 0,
        .p99_latency_ns = 0,
        .max_latency_ns = 0,
        .passed = passed,
        .error_message = error_message,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "stress: profiles module loads" {
    const profile = StressProfile.medium;
    try std.testing.expectEqualStrings("medium", profile.name);
}

test "stress: all profiles available" {
    _ = StressProfile.light;
    _ = StressProfile.medium;
    _ = StressProfile.heavy;
    _ = StressProfile.extreme;
    _ = StressProfile.quick;
}

test "stress: timer works" {
    const timer = Timer.start();
    sleepMs(1);
    const elapsed = timer.read();
    try std.testing.expect(elapsed > 0);
}

test "stress: latency histogram" {
    const allocator = std.testing.allocator;

    var histogram = LatencyHistogram.init(allocator);
    defer histogram.deinit();

    try histogram.record(100);
    try histogram.record(200);
    try histogram.record(300);

    const stats = histogram.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.count);
    try std.testing.expectEqual(@as(u64, 200), stats.avg);
}

test "stress: get profile by name" {
    try std.testing.expect(getProfileByName("light") != null);
    try std.testing.expect(getProfileByName("medium") != null);
    try std.testing.expect(getProfileByName("heavy") != null);
    try std.testing.expect(getProfileByName("extreme") != null);
    try std.testing.expect(getProfileByName("quick") != null);
    try std.testing.expect(getProfileByName("nonexistent") == null);
}

test "stress: profile scaling" {
    const base = StressProfile.medium;
    const scaled = base.scale(0.5);

    try std.testing.expectEqual(@as(u64, 5000), scaled.operations);
    try std.testing.expectEqual(@as(u32, 8), scaled.concurrent_tasks);
}

test "stress: ops per thread calculation" {
    const profile = StressProfile{
        .name = "test",
        .operations = 1000,
        .concurrent_tasks = 10,
        .duration_seconds = 60,
        .memory_pressure_mb = 256,
    };

    try std.testing.expectEqual(@as(u64, 100), profile.opsPerThread());
}
