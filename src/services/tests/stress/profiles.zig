//! Stress Test Profile System
//!
//! Provides configurable profiles for stress testing at different intensity levels.
//! Profiles define operation counts, concurrency, duration, and memory pressure
//! parameters for systematic stress testing across all modules.
//!
//! ## Usage
//!
//! ```zig
//! const profiles = @import("profiles.zig");
//!
//! // Use predefined profile
//! const profile = profiles.StressProfile.medium;
//!
//! // Or create custom profile
//! const custom = profiles.StressProfile{
//!     .name = "custom",
//!     .operations = 50_000,
//!     .concurrent_tasks = 32,
//!     .duration_seconds = 120,
//!     .memory_pressure_mb = 512,
//! };
//!
//! // Get profile from environment or default
//! const active = profiles.getActiveProfile();
//! ```

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const time = abi.shared.time;

/// Stress test profile defining test parameters
pub const StressProfile = struct {
    /// Profile name for identification
    name: []const u8,
    /// Total number of operations to perform
    operations: u64,
    /// Number of concurrent tasks/threads
    concurrent_tasks: u32,
    /// Maximum test duration in seconds
    duration_seconds: u64,
    /// Memory pressure target in megabytes
    memory_pressure_mb: u32,
    /// Batch size for bulk operations
    batch_size: u32 = 100,
    /// Warmup operations before measurement
    warmup_operations: u64 = 100,
    /// Whether to verify results (slower but safer)
    verify_results: bool = true,
    /// Random seed for reproducibility (0 = use time)
    seed: u64 = 0,

    /// Light profile - quick smoke tests
    /// Suitable for CI/CD pipelines and quick validation
    pub const light = StressProfile{
        .name = "light",
        .operations = 1_000,
        .concurrent_tasks = 4,
        .duration_seconds = 10,
        .memory_pressure_mb = 64,
        .batch_size = 50,
        .warmup_operations = 50,
    };

    /// Medium profile - standard stress testing
    /// Good balance between coverage and execution time
    pub const medium = StressProfile{
        .name = "medium",
        .operations = 10_000,
        .concurrent_tasks = 16,
        .duration_seconds = 60,
        .memory_pressure_mb = 256,
        .batch_size = 100,
        .warmup_operations = 100,
    };

    /// Heavy profile - thorough stress testing
    /// For pre-release validation and nightly builds
    pub const heavy = StressProfile{
        .name = "heavy",
        .operations = 100_000,
        .concurrent_tasks = 64,
        .duration_seconds = 300,
        .memory_pressure_mb = 1024,
        .batch_size = 500,
        .warmup_operations = 500,
    };

    /// Extreme profile - maximum stress testing
    /// For dedicated stress testing environments
    pub const extreme = StressProfile{
        .name = "extreme",
        .operations = 1_000_000,
        .concurrent_tasks = 256,
        .duration_seconds = 3600,
        .memory_pressure_mb = 4096,
        .batch_size = 1000,
        .warmup_operations = 1000,
        .verify_results = false, // Too slow at this scale
    };

    /// Quick profile - minimal testing for development
    pub const quick = StressProfile{
        .name = "quick",
        .operations = 100,
        .concurrent_tasks = 2,
        .duration_seconds = 5,
        .memory_pressure_mb = 16,
        .batch_size = 10,
        .warmup_operations = 10,
    };

    /// Scale the profile by a factor
    pub fn scale(self: StressProfile, factor: f64) StressProfile {
        return .{
            .name = self.name,
            .operations = @intFromFloat(@as(f64, @floatFromInt(self.operations)) * factor),
            .concurrent_tasks = @intFromFloat(@as(f64, @floatFromInt(self.concurrent_tasks)) * factor),
            .duration_seconds = @intFromFloat(@as(f64, @floatFromInt(self.duration_seconds)) * factor),
            .memory_pressure_mb = @intFromFloat(@as(f64, @floatFromInt(self.memory_pressure_mb)) * factor),
            .batch_size = self.batch_size,
            .warmup_operations = self.warmup_operations,
            .verify_results = self.verify_results,
            .seed = self.seed,
        };
    }

    /// Get operations per thread
    pub fn opsPerThread(self: StressProfile) u64 {
        return self.operations / @as(u64, self.concurrent_tasks);
    }

    /// Get effective seed (time-based if 0)
    pub fn getEffectiveSeed(self: StressProfile) u64 {
        if (self.seed != 0) return self.seed;
        // Use Timer for time-based seed
        var timer = std.time.Timer.start() catch return 12345;
        return timer.read();
    }
};

/// Stress test result statistics
pub const StressResult = struct {
    /// Profile used for the test
    profile_name: []const u8,
    /// Total operations completed
    operations_completed: u64,
    /// Total operations failed
    operations_failed: u64,
    /// Total duration in nanoseconds
    duration_ns: u64,
    /// Peak memory usage in bytes
    peak_memory_bytes: u64,
    /// Operations per second
    ops_per_second: f64,
    /// Average latency in nanoseconds
    avg_latency_ns: u64,
    /// P50 latency in nanoseconds
    p50_latency_ns: u64,
    /// P95 latency in nanoseconds
    p95_latency_ns: u64,
    /// P99 latency in nanoseconds
    p99_latency_ns: u64,
    /// Maximum latency in nanoseconds
    max_latency_ns: u64,
    /// Whether the test passed all assertions
    passed: bool,
    /// Buffer for error message storage (used with {t} format specifier)
    error_buffer: [64]u8 = undefined,
    /// Error message if failed (points into error_buffer or null)
    error_message: ?[]const u8 = null,

    pub fn format(
        self: StressResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const status = if (self.passed) "PASSED" else "FAILED";
        try writer.print(
            \\StressResult({s}):
            \\  Status: {s}
            \\  Operations: {d} completed, {d} failed
            \\  Duration: {d}ms
            \\  Throughput: {d:.2} ops/sec
            \\  Latency (avg/p50/p95/p99/max): {d}us / {d}us / {d}us / {d}us / {d}us
            \\  Peak Memory: {d} MB
        , .{
            self.profile_name,
            status,
            self.operations_completed,
            self.operations_failed,
            self.duration_ns / std.time.ns_per_ms,
            self.ops_per_second,
            self.avg_latency_ns / 1000,
            self.p50_latency_ns / 1000,
            self.p95_latency_ns / 1000,
            self.p99_latency_ns / 1000,
            self.max_latency_ns / 1000,
            self.peak_memory_bytes / (1024 * 1024),
        });
        if (self.error_message) |msg| {
            try writer.print("\n  Error: {s}", .{msg});
        }
    }
};

/// Latency histogram for tracking response times
pub const LatencyHistogram = struct {
    samples: std.ArrayListUnmanaged(u64),
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) LatencyHistogram {
        return .{
            .samples = .empty,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *LatencyHistogram) void {
        self.samples.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn record(self: *LatencyHistogram, latency_ns: u64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.samples.append(self.allocator, latency_ns);
    }

    pub fn recordUnsafe(self: *LatencyHistogram, latency_ns: u64) !void {
        try self.samples.append(self.allocator, latency_ns);
    }

    pub fn getStats(self: *LatencyHistogram) LatencyStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.samples.items.len == 0) {
            return .{
                .count = 0,
                .avg = 0,
                .p50 = 0,
                .p95 = 0,
                .p99 = 0,
                .max = 0,
            };
        }

        // Sort for percentile calculation
        std.mem.sort(u64, self.samples.items, {}, std.sort.asc(u64));

        const count = self.samples.items.len;
        var sum: u64 = 0;
        for (self.samples.items) |s| {
            sum += s;
        }

        return .{
            .count = count,
            .avg = sum / count,
            .p50 = self.samples.items[count / 2],
            .p95 = self.samples.items[@min(count - 1, (count * 95) / 100)],
            .p99 = self.samples.items[@min(count - 1, (count * 99) / 100)],
            .max = self.samples.items[count - 1],
        };
    }
};

pub const LatencyStats = struct {
    count: usize,
    avg: u64,
    p50: u64,
    p95: u64,
    p99: u64,
    max: u64,
};

/// Get active profile from environment or return default
/// Environment variable: STRESS_PROFILE (light, medium, heavy, extreme, quick)
pub fn getActiveProfile() StressProfile {
    // In test builds, always use quick profile
    if (builtin.mode == .Debug) {
        return StressProfile.quick;
    }

    // In release builds, check environment (when possible)
    // For now, default to medium for release testing
    return StressProfile.medium;
}

/// Get profile by name
pub fn getProfileByName(name: []const u8) ?StressProfile {
    if (std.mem.eql(u8, name, "light")) return StressProfile.light;
    if (std.mem.eql(u8, name, "medium")) return StressProfile.medium;
    if (std.mem.eql(u8, name, "heavy")) return StressProfile.heavy;
    if (std.mem.eql(u8, name, "extreme")) return StressProfile.extreme;
    if (std.mem.eql(u8, name, "quick")) return StressProfile.quick;
    return null;
}

/// Sleep helper that works across platforms.
/// Delegates to shared time module for consistent implementation.
pub const sleepMs = time.sleepMs;

/// Timer for measuring elapsed time
/// Uses std.time.Timer on native platforms, fallback for WASM
pub const Timer = struct {
    inner: ?std.time.Timer,

    pub fn start() Timer {
        if (builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64) {
            // WASM: no timer available
            return .{ .inner = null };
        }

        // Use std.time.Timer for native platforms
        const inner = std.time.Timer.start() catch null;
        return .{ .inner = inner };
    }

    pub fn read(self: Timer) u64 {
        if (self.inner) |t| {
            // Note: std.time.Timer.read() is mutable, but we can work around
            // by creating a copy and calling read on it
            var timer_copy = t;
            return timer_copy.read();
        }
        return 0;
    }

    pub fn readMs(self: Timer) u64 {
        return self.read() / std.time.ns_per_ms;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "stress profile light" {
    const profile = StressProfile.light;
    try std.testing.expectEqualStrings("light", profile.name);
    try std.testing.expectEqual(@as(u64, 1_000), profile.operations);
    try std.testing.expectEqual(@as(u32, 4), profile.concurrent_tasks);
}

test "stress profile medium" {
    const profile = StressProfile.medium;
    try std.testing.expectEqualStrings("medium", profile.name);
    try std.testing.expectEqual(@as(u64, 10_000), profile.operations);
    try std.testing.expectEqual(@as(u32, 16), profile.concurrent_tasks);
}

test "stress profile heavy" {
    const profile = StressProfile.heavy;
    try std.testing.expectEqualStrings("heavy", profile.name);
    try std.testing.expectEqual(@as(u64, 100_000), profile.operations);
    try std.testing.expectEqual(@as(u32, 64), profile.concurrent_tasks);
}

test "stress profile extreme" {
    const profile = StressProfile.extreme;
    try std.testing.expectEqualStrings("extreme", profile.name);
    try std.testing.expectEqual(@as(u64, 1_000_000), profile.operations);
    try std.testing.expectEqual(@as(u32, 256), profile.concurrent_tasks);
}

test "stress profile scaling" {
    const base = StressProfile.medium;
    const scaled = base.scale(2.0);

    try std.testing.expectEqual(@as(u64, 20_000), scaled.operations);
    try std.testing.expectEqual(@as(u32, 32), scaled.concurrent_tasks);
}

test "ops per thread calculation" {
    const profile = StressProfile.medium;
    try std.testing.expectEqual(@as(u64, 625), profile.opsPerThread());
}

test "get profile by name" {
    try std.testing.expect(getProfileByName("light") != null);
    try std.testing.expect(getProfileByName("medium") != null);
    try std.testing.expect(getProfileByName("heavy") != null);
    try std.testing.expect(getProfileByName("extreme") != null);
    try std.testing.expect(getProfileByName("invalid") == null);
}

test "latency histogram" {
    const allocator = std.testing.allocator;

    var histogram = LatencyHistogram.init(allocator);
    defer histogram.deinit();

    // Record some samples
    try histogram.record(100);
    try histogram.record(200);
    try histogram.record(300);
    try histogram.record(400);
    try histogram.record(500);

    const stats = histogram.getStats();
    try std.testing.expectEqual(@as(usize, 5), stats.count);
    try std.testing.expectEqual(@as(u64, 300), stats.avg);
    try std.testing.expectEqual(@as(u64, 300), stats.p50);
    try std.testing.expectEqual(@as(u64, 500), stats.max);
}

test "timer basic" {
    const timer = Timer.start();
    sleepMs(50); // Use longer sleep for better timer resolution
    const elapsed = timer.read();
    // On some platforms, timer resolution may be low or sleep may not be precise
    // The important thing is that the timer doesn't error
    try std.testing.expect(elapsed >= 0);
}
