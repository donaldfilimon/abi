// SPDX-License-Identifier: MIT
// Copyright (c) 2024 ABI Framework
//
//! Percentile tracker for latency and performance metrics.
//!
//! Tracks a sample of values and computes percentiles (P50, P90, P99, etc.)
//! which are essential for understanding service latency characteristics.
//!
//! ## Why Percentiles Matter
//!
//! - **P50 (median)**: Typical user experience - half of requests are faster
//! - **P90**: 90% of requests are faster - good baseline for SLOs
//! - **P99**: Tail latency - affects 1% of users, often most vocal
//! - **Averages lie**: A few slow requests can hide behind a "good" average
//!
//! ## Usage
//!
//! ```zig
//! var tracker = PercentileTracker.init(allocator, 10000);
//! defer tracker.deinit();
//!
//! // Add latency samples as they arrive
//! tracker.add(45);   // 45ms
//! tracker.add(127);  // 127ms
//! tracker.add(23);   // 23ms
//!
//! // Query percentiles
//! const p50 = tracker.getPercentile(50);  // Median
//! const p99 = tracker.getPercentile(99);  // Tail latency
//! ```

const std = @import("std");

/// Tracks values and computes percentiles using a sampling approach.
/// Maintains up to `max_samples` values, evicting oldest when full.
pub const PercentileTracker = struct {
    samples: std.ArrayListUnmanaged(u32),
    allocator: std.mem.Allocator,
    sorted: bool,
    max_samples: usize,

    /// Initialize a new percentile tracker.
    /// `max_samples` controls memory usage - higher values give more accuracy.
    /// Typical values: 1000-10000 for dashboard use.
    pub fn init(allocator: std.mem.Allocator, max_samples: usize) PercentileTracker {
        return .{
            .samples = .empty,
            .allocator = allocator,
            .sorted = true,
            .max_samples = max_samples,
        };
    }

    pub fn deinit(self: *PercentileTracker) void {
        self.samples.deinit(self.allocator);
    }

    /// Add a sample value. If at capacity, removes the oldest sample.
    pub fn add(self: *PercentileTracker, value: u32) void {
        if (self.samples.items.len >= self.max_samples) {
            // Remove oldest (first) sample - FIFO eviction
            _ = self.samples.orderedRemove(0);
        }
        self.samples.append(self.allocator, value) catch return;
        self.sorted = false;
    }

    /// Get the value at a given percentile (0-100).
    /// Returns 0 if no samples have been added.
    ///
    /// Common percentiles:
    /// - 50: Median (P50)
    /// - 90: P90 - 90% of values are below this
    /// - 95: P95
    /// - 99: P99 - Tail latency
    /// - 99.9: P99.9 - Extreme tail (use getPercentileFloat for this)
    pub fn getPercentile(self: *PercentileTracker, p: u8) u32 {
        if (self.samples.items.len == 0) return 0;

        self.ensureSorted();

        // Calculate index: for P50 with 100 samples, index = 50
        const index = (self.samples.items.len * @as(usize, p)) / 100;
        const clamped = @min(index, self.samples.items.len - 1);
        return self.samples.items[clamped];
    }

    /// Get percentile with float precision (e.g., 99.9 for P99.9).
    pub fn getPercentileFloat(self: *PercentileTracker, p: f64) u32 {
        if (self.samples.items.len == 0) return 0;

        self.ensureSorted();

        const len_f = @as(f64, @floatFromInt(self.samples.items.len));
        const index_f = (len_f * p) / 100.0;
        const index = @as(usize, @intFromFloat(@floor(index_f)));
        const clamped = @min(index, self.samples.items.len - 1);
        return self.samples.items[clamped];
    }

    /// Get minimum value, or 0 if empty.
    pub fn getMin(self: *PercentileTracker) u32 {
        if (self.samples.items.len == 0) return 0;
        self.ensureSorted();
        return self.samples.items[0];
    }

    /// Get maximum value, or 0 if empty.
    pub fn getMax(self: *PercentileTracker) u32 {
        if (self.samples.items.len == 0) return 0;
        self.ensureSorted();
        return self.samples.items[self.samples.items.len - 1];
    }

    /// Get the arithmetic mean.
    pub fn getMean(self: *const PercentileTracker) f64 {
        if (self.samples.items.len == 0) return 0;

        var sum: u64 = 0;
        for (self.samples.items) |v| {
            sum += v;
        }
        return @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(self.samples.items.len));
    }

    /// Get standard deviation.
    pub fn getStdDev(self: *PercentileTracker) f64 {
        if (self.samples.items.len < 2) return 0;

        const mean = self.getMean();
        var sum_sq: f64 = 0;
        for (self.samples.items) |v| {
            const diff = @as(f64, @floatFromInt(v)) - mean;
            sum_sq += diff * diff;
        }
        return @sqrt(sum_sq / @as(f64, @floatFromInt(self.samples.items.len)));
    }

    /// Remove all samples.
    pub fn reset(self: *PercentileTracker) void {
        self.samples.clearRetainingCapacity();
        self.sorted = true;
    }

    /// Current number of samples.
    pub fn count(self: *const PercentileTracker) usize {
        return self.samples.items.len;
    }

    /// Check if any samples have been added.
    pub fn isEmpty(self: *const PercentileTracker) bool {
        return self.samples.items.len == 0;
    }

    /// Get a summary of key percentiles.
    pub const Summary = struct {
        count: usize,
        min: u32,
        max: u32,
        mean: f64,
        p50: u32,
        p90: u32,
        p95: u32,
        p99: u32,
    };

    /// Get a summary of all key statistics at once.
    pub fn getSummary(self: *PercentileTracker) Summary {
        return .{
            .count = self.samples.items.len,
            .min = self.getMin(),
            .max = self.getMax(),
            .mean = self.getMean(),
            .p50 = self.getPercentile(50),
            .p90 = self.getPercentile(90),
            .p95 = self.getPercentile(95),
            .p99 = self.getPercentile(99),
        };
    }

    // Internal: ensure samples are sorted for percentile calculations
    fn ensureSorted(self: *PercentileTracker) void {
        if (!self.sorted) {
            std.mem.sort(u32, self.samples.items, {}, std.sort.asc(u32));
            self.sorted = true;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PercentileTracker calculates P50 correctly" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 1000);
    defer tracker.deinit();

    // Add values 1-100
    for (1..101) |i| {
        tracker.add(@intCast(i));
    }

    const p50 = tracker.getPercentile(50);
    // P50 of 1-100 should be around 50
    try std.testing.expect(p50 >= 49 and p50 <= 51);
}

test "PercentileTracker calculates P99 correctly" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 1000);
    defer tracker.deinit();

    for (1..101) |i| {
        tracker.add(@intCast(i));
    }

    const p99 = tracker.getPercentile(99);
    // P99 of 1-100 should be around 99
    try std.testing.expect(p99 >= 98 and p99 <= 100);
}

test "PercentileTracker min and max" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 100);
    defer tracker.deinit();

    tracker.add(50);
    tracker.add(10);
    tracker.add(90);
    tracker.add(30);

    try std.testing.expectEqual(@as(u32, 10), tracker.getMin());
    try std.testing.expectEqual(@as(u32, 90), tracker.getMax());
}

test "PercentileTracker mean calculation" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 100);
    defer tracker.deinit();

    tracker.add(10);
    tracker.add(20);
    tracker.add(30);

    try std.testing.expectApproxEqAbs(@as(f64, 20.0), tracker.getMean(), 0.001);
}

test "PercentileTracker respects max_samples" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 5);
    defer tracker.deinit();

    // Add 10 values
    for (0..10) |i| {
        tracker.add(@intCast(i * 10));
    }

    // Should only have 5 samples
    try std.testing.expectEqual(@as(usize, 5), tracker.count());

    // Should have the 5 most recent values (50, 60, 70, 80, 90)
    try std.testing.expectEqual(@as(u32, 50), tracker.getMin());
    try std.testing.expectEqual(@as(u32, 90), tracker.getMax());
}

test "PercentileTracker reset" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 100);
    defer tracker.deinit();

    tracker.add(100);
    tracker.add(200);
    try std.testing.expectEqual(@as(usize, 2), tracker.count());

    tracker.reset();
    try std.testing.expectEqual(@as(usize, 0), tracker.count());
    try std.testing.expect(tracker.isEmpty());
}

test "PercentileTracker empty returns zero" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 100);
    defer tracker.deinit();

    try std.testing.expectEqual(@as(u32, 0), tracker.getPercentile(50));
    try std.testing.expectEqual(@as(u32, 0), tracker.getMin());
    try std.testing.expectEqual(@as(u32, 0), tracker.getMax());
    try std.testing.expectEqual(@as(f64, 0), tracker.getMean());
}

test "PercentileTracker getSummary" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 1000);
    defer tracker.deinit();

    for (1..101) |i| {
        tracker.add(@intCast(i));
    }

    const summary = tracker.getSummary();
    try std.testing.expectEqual(@as(usize, 100), summary.count);
    try std.testing.expectEqual(@as(u32, 1), summary.min);
    try std.testing.expectEqual(@as(u32, 100), summary.max);
    try std.testing.expect(summary.p50 >= 49 and summary.p50 <= 51);
    try std.testing.expect(summary.p99 >= 98);
}

test "PercentileTracker getPercentileFloat for P99.9" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 10000);
    defer tracker.deinit();

    // Add 1000 values
    for (1..1001) |i| {
        tracker.add(@intCast(i));
    }

    const p999 = tracker.getPercentileFloat(99.9);
    // P99.9 of 1-1000 should be around 999
    try std.testing.expect(p999 >= 998);
}

test "PercentileTracker standard deviation" {
    const allocator = std.testing.allocator;
    var tracker = PercentileTracker.init(allocator, 100);
    defer tracker.deinit();

    // Uniform values should have zero std dev
    tracker.add(50);
    tracker.add(50);
    tracker.add(50);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tracker.getStdDev(), 0.001);

    tracker.reset();

    // Known std dev: [2, 4, 4, 4, 5, 5, 7, 9] has stddev ≈ 2.0
    tracker.add(2);
    tracker.add(4);
    tracker.add(4);
    tracker.add(4);
    tracker.add(5);
    tracker.add(5);
    tracker.add(7);
    tracker.add(9);
    const stddev = tracker.getStdDev();
    try std.testing.expect(stddev > 1.5 and stddev < 2.5);
}

test {
    std.testing.refAllDecls(@This());
}
