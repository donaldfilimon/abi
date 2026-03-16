//! Comprehensive Tests for the Observability Module
//!
//! Core primitive tests live here. Domain-specific tests are in:
//! - `observability_metrics_test.zig` — Core metrics, collectors, Prometheus, sliding window, circuit breaker
//! - `observability_tracing_test.zig` — Spans, tracers, IDs, samplers, OpenTelemetry
//! - `observability_alerting_test.zig` — Alert manager, rules, conditions
//! - `observability_edge_test.zig` — Edge cases, module state

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

// Import observability types from the module
const observability = abi.observability;
const Counter = observability.Counter;
const Gauge = observability.Gauge;
const FloatGauge = observability.FloatGauge;
const Histogram = observability.Histogram;

// ============================================================================
// Counter Tests
// ============================================================================

test "observability: counter increment by 1" {
    var counter = Counter{ .name = "test_counter" };
    try testing.expectEqual(@as(u64, 0), counter.get());

    counter.inc(1);
    try testing.expectEqual(@as(u64, 1), counter.get());

    counter.inc(1);
    try testing.expectEqual(@as(u64, 2), counter.get());
}

test "observability: counter increment by N" {
    var counter = Counter{ .name = "test_counter" };

    counter.inc(5);
    try testing.expectEqual(@as(u64, 5), counter.get());

    counter.inc(10);
    try testing.expectEqual(@as(u64, 15), counter.get());

    counter.inc(100);
    try testing.expectEqual(@as(u64, 115), counter.get());
}

test "observability: counter reset to zero" {
    var counter = Counter{ .name = "test_counter" };

    counter.inc(42);
    try testing.expectEqual(@as(u64, 42), counter.get());

    counter.reset();
    try testing.expectEqual(@as(u64, 0), counter.get());

    // Verify counter works after reset
    counter.inc(1);
    try testing.expectEqual(@as(u64, 1), counter.get());
}

test "observability: counter large value handling" {
    var counter = Counter{ .name = "test_counter" };

    // Test with large values
    counter.inc(std.math.maxInt(u32));
    try testing.expectEqual(@as(u64, std.math.maxInt(u32)), counter.get());

    counter.inc(std.math.maxInt(u32));
    try testing.expectEqual(@as(u64, @as(u64, std.math.maxInt(u32)) * 2), counter.get());
}

test "observability: counter thread-safe concurrent increments" {
    var counter = Counter{ .name = "test_counter" };

    const num_threads = 8;
    const increments_per_thread = 1000;

    var threads: [num_threads]std.Thread = undefined;

    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(c: *Counter) void {
                for (0..increments_per_thread) |_| {
                    c.inc(1);
                }
            }
        }.worker, .{&counter});
    }

    for (&threads) |*t| {
        t.join();
    }

    try testing.expectEqual(@as(u64, num_threads * increments_per_thread), counter.get());
}

// ============================================================================
// Gauge Tests
// ============================================================================

test "observability: gauge set value" {
    var gauge = Gauge{ .name = "test_gauge" };
    try testing.expectEqual(@as(i64, 0), gauge.get());

    gauge.set(42);
    try testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.set(0);
    try testing.expectEqual(@as(i64, 0), gauge.get());

    gauge.set(1000);
    try testing.expectEqual(@as(i64, 1000), gauge.get());
}

test "observability: gauge increment and decrement" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.inc();
    try testing.expectEqual(@as(i64, 1), gauge.get());

    gauge.inc();
    gauge.inc();
    try testing.expectEqual(@as(i64, 3), gauge.get());

    gauge.dec();
    try testing.expectEqual(@as(i64, 2), gauge.get());

    gauge.dec();
    gauge.dec();
    try testing.expectEqual(@as(i64, 0), gauge.get());
}

test "observability: gauge negative values" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.set(-100);
    try testing.expectEqual(@as(i64, -100), gauge.get());

    gauge.add(-50);
    try testing.expectEqual(@as(i64, -150), gauge.get());

    gauge.add(200);
    try testing.expectEqual(@as(i64, 50), gauge.get());
}

test "observability: gauge add operation" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.add(10);
    try testing.expectEqual(@as(i64, 10), gauge.get());

    gauge.add(-5);
    try testing.expectEqual(@as(i64, 5), gauge.get());

    gauge.add(100);
    try testing.expectEqual(@as(i64, 105), gauge.get());
}

test "observability: gauge concurrent updates" {
    var gauge = Gauge{ .name = "test_gauge" };
    gauge.set(1000);

    const num_threads = 8;
    const ops_per_thread = 100;

    var threads: [num_threads]std.Thread = undefined;

    // Half threads increment, half decrement
    for (&threads, 0..) |*t, i| {
        const should_increment = i < num_threads / 2;
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(g: *Gauge, inc: bool) void {
                for (0..ops_per_thread) |_| {
                    if (inc) {
                        g.inc();
                    } else {
                        g.dec();
                    }
                }
            }
        }.worker, .{ &gauge, should_increment });
    }

    for (&threads) |*t| {
        t.join();
    }

    // Equal increments and decrements should leave value unchanged
    try testing.expectEqual(@as(i64, 1000), gauge.get());
}

// ============================================================================
// FloatGauge Tests
// ============================================================================

test "observability: float gauge set value" {
    var gauge = FloatGauge{ .name = "test_float_gauge" };
    try testing.expectApproxEqAbs(@as(f64, 0.0), gauge.get(), 0.0001);

    gauge.set(3.14159);
    try testing.expectApproxEqAbs(@as(f64, 3.14159), gauge.get(), 0.0001);

    gauge.set(-2.71828);
    try testing.expectApproxEqAbs(@as(f64, -2.71828), gauge.get(), 0.0001);
}

test "observability: float gauge add operation" {
    var gauge = FloatGauge{ .name = "test_float_gauge" };

    gauge.add(1.5);
    try testing.expectApproxEqAbs(@as(f64, 1.5), gauge.get(), 0.0001);

    gauge.add(2.5);
    try testing.expectApproxEqAbs(@as(f64, 4.0), gauge.get(), 0.0001);

    gauge.add(-1.0);
    try testing.expectApproxEqAbs(@as(f64, 3.0), gauge.get(), 0.0001);
}

test "observability: float gauge precision" {
    var gauge = FloatGauge{ .name = "test_float_gauge" };

    // Test small values
    gauge.set(0.000001);
    try testing.expectApproxEqAbs(@as(f64, 0.000001), gauge.get(), 0.0000001);

    // Test large values
    gauge.set(1_000_000.123456);
    try testing.expectApproxEqAbs(@as(f64, 1_000_000.123456), gauge.get(), 0.0001);

    // Test accumulated precision
    gauge.set(0);
    for (0..1000) |_| {
        gauge.add(0.001);
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), gauge.get(), 0.0001);
}

test "observability: float gauge concurrent updates" {
    var gauge = FloatGauge{ .name = "test_float_gauge" };

    const num_threads = 4;
    const ops_per_thread = 100;

    var threads: [num_threads]std.Thread = undefined;

    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(g: *FloatGauge) void {
                for (0..ops_per_thread) |_| {
                    g.add(1.0);
                }
            }
        }.worker, .{&gauge});
    }

    for (&threads) |*t| {
        t.join();
    }

    try testing.expectApproxEqAbs(@as(f64, @floatFromInt(num_threads * ops_per_thread)), gauge.get(), 0.0001);
}

// ============================================================================
// Histogram Tests (main module version)
// ============================================================================

test "observability: histogram record single value" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 50, 100, 500, 1000 };
    var hist = try Histogram.init(allocator, "test_histogram", @constCast(&bounds));
    defer hist.deinit(allocator);

    hist.record(25);
    try testing.expectEqual(@as(u64, 1), hist.buckets[1]); // 25 <= 50
}

test "observability: histogram record multiple values" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 50, 100, 500, 1000 };
    var hist = try Histogram.init(allocator, "test_histogram", @constCast(&bounds));
    defer hist.deinit(allocator);

    hist.record(5); // <= 10
    hist.record(25); // <= 50
    hist.record(75); // <= 100
    hist.record(250); // <= 500
    hist.record(750); // <= 1000
    hist.record(2000); // > 1000 (overflow bucket)

    try testing.expectEqual(@as(u64, 1), hist.buckets[0]); // <= 10
    try testing.expectEqual(@as(u64, 1), hist.buckets[1]); // <= 50
    try testing.expectEqual(@as(u64, 1), hist.buckets[2]); // <= 100
    try testing.expectEqual(@as(u64, 1), hist.buckets[3]); // <= 500
    try testing.expectEqual(@as(u64, 1), hist.buckets[4]); // <= 1000
    try testing.expectEqual(@as(u64, 1), hist.buckets[5]); // > 1000 (overflow)
}

test "observability: histogram bucket boundaries" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 20, 30 };
    var hist = try Histogram.init(allocator, "test_histogram", @constCast(&bounds));
    defer hist.deinit(allocator);

    // Test exact boundary values
    hist.record(10); // Should go in first bucket (<=10)
    hist.record(20); // Should go in second bucket (<=20)
    hist.record(30); // Should go in third bucket (<=30)

    try testing.expectEqual(@as(u64, 1), hist.buckets[0]);
    try testing.expectEqual(@as(u64, 1), hist.buckets[1]);
    try testing.expectEqual(@as(u64, 1), hist.buckets[2]);
}

test "observability: histogram overflow bucket" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 20, 30 };
    var hist = try Histogram.init(allocator, "test_histogram", @constCast(&bounds));
    defer hist.deinit(allocator);

    // Values above max bound go to overflow bucket
    hist.record(100);
    hist.record(200);
    hist.record(1000);

    try testing.expectEqual(@as(u64, 0), hist.buckets[0]);
    try testing.expectEqual(@as(u64, 0), hist.buckets[1]);
    try testing.expectEqual(@as(u64, 0), hist.buckets[2]);
    try testing.expectEqual(@as(u64, 3), hist.buckets[3]); // overflow bucket
}

// ============================================================================
// Test discovery for extracted test files
// ============================================================================

test {
    _ = @import("observability_metrics_test.zig");
    _ = @import("observability_tracing_test.zig");
    _ = @import("observability_alerting_test.zig");
    _ = @import("observability_edge_test.zig");
}
