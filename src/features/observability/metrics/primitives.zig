//! Core Metrics Primitives
//!
//! Thread-safe metric types for counters, gauges, and histograms.
//! Used as building blocks for domain-specific metrics.
//!
//! Implementation lives in `services/shared/utils/metric_types.zig`
//! since these primitives are shared across multiple feature modules.

const std = @import("std");
const metric_types = @import("../../../services/shared/utils/metric_types.zig");

// Re-export all types from the shared implementation
pub const Counter = metric_types.Counter;
pub const Gauge = metric_types.Gauge;
pub const FloatGauge = metric_types.FloatGauge;
pub const default_latency_buckets = metric_types.default_latency_buckets;
pub const Histogram = metric_types.Histogram;
pub const LatencyHistogram = metric_types.LatencyHistogram;

// ============================================================================
// Tests
// ============================================================================

test "Counter basic operations" {
    var counter = Counter{};
    try std.testing.expectEqual(@as(u64, 0), counter.get());

    counter.inc();
    try std.testing.expectEqual(@as(u64, 1), counter.get());

    counter.add(5);
    try std.testing.expectEqual(@as(u64, 6), counter.get());

    counter.reset();
    try std.testing.expectEqual(@as(u64, 0), counter.get());
}

test "Gauge basic operations" {
    var gauge = Gauge{};
    try std.testing.expectEqual(@as(i64, 0), gauge.get());

    gauge.set(42);
    try std.testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.inc();
    try std.testing.expectEqual(@as(i64, 43), gauge.get());

    gauge.dec();
    try std.testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.add(-10);
    try std.testing.expectEqual(@as(i64, 32), gauge.get());
}

test "FloatGauge basic operations" {
    var gauge = FloatGauge{};
    try std.testing.expectApproxEqAbs(@as(f64, 0), gauge.get(), 0.001);

    gauge.set(3.14);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), gauge.get(), 0.001);

    gauge.add(1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 4.14), gauge.get(), 0.001);
}

test "Histogram observe and mean" {
    var hist = LatencyHistogram.initDefault();

    hist.observe(10);
    hist.observe(20);
    hist.observe(30);

    try std.testing.expectEqual(@as(u64, 3), hist.getCount());
    try std.testing.expectApproxEqAbs(@as(f64, 20), hist.mean(), 0.001);
}

test "Histogram percentile" {
    var hist = LatencyHistogram.initDefault();

    // Add values that fall into different buckets
    hist.observe(5);
    hist.observe(10);
    hist.observe(50);
    hist.observe(100);
    hist.observe(500);

    // P50 should be around 50ms bucket
    const p50 = hist.percentile(0.5);
    try std.testing.expect(p50 >= 10 and p50 <= 100);
}

test {
    std.testing.refAllDecls(@This());
}
