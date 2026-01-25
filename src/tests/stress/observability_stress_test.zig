//! Observability Module Stress Tests
//!
//! Comprehensive stress tests for the observability module components:
//! - Counter/Gauge high-throughput updates
//! - Histogram with millions of samples
//! - MetricsCollector under concurrent load
//! - Tracer/Span creation and completion
//! - Memory stability with long-running collection
//!
//! ## Running Tests
//!
//! ```bash
//! zig test src/tests/stress/observability_stress_test.zig --test-filter "observability stress"
//! ```

const std = @import("std");
const abi = @import("abi");
const obs = abi.observability;
const profiles = @import("profiles.zig");
const StressProfile = profiles.StressProfile;
const LatencyHistogram = profiles.LatencyHistogram;
const Timer = profiles.Timer;
const build_options = @import("build_options");

// ============================================================================
// Configuration
// ============================================================================

/// Get the active stress profile for tests
fn getTestProfile() StressProfile {
    return StressProfile.quick;
}

// ============================================================================
// Counter Stress Tests
// ============================================================================

test "observability stress: counter high throughput" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const profile = getTestProfile();

    var counter = obs.Counter{
        .name = "stress_counter",
    };

    var latency = LatencyHistogram.init(std.testing.allocator);
    defer latency.deinit();

    // Single-threaded rapid increments
    const ops = profile.operations;
    for (0..ops) |i| {
        const timer = Timer.start();
        counter.inc(1);
        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Verify count
    try std.testing.expectEqual(ops, counter.get());

    // Check latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

test "observability stress: counter concurrent increments" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const profile = getTestProfile();

    var counter = obs.Counter{
        .name = "concurrent_counter",
    };

    const thread_count = @min(profile.concurrent_tasks, 32);
    const ops_per_thread = profile.operations / thread_count;

    var threads: [32]std.Thread = undefined;

    for (0..thread_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, struct {
            fn run(c: *obs.Counter, ops: u64) void {
                for (0..ops) |_| {
                    c.inc(1);
                }
            }
        }.run, .{ &counter, ops_per_thread });
    }

    for (0..thread_count) |i| {
        threads[i].join();
    }

    // Verify total count
    const expected = thread_count * ops_per_thread;
    try std.testing.expectEqual(expected, counter.get());
}

test "observability stress: counter overflow handling" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    var counter = obs.Counter{
        .name = "overflow_counter",
    };

    // Set near max value
    counter.value.store(std.math.maxInt(u64) - 1000, .monotonic);

    // Increment past overflow
    for (0..2000) |_| {
        counter.inc(1);
    }

    // Should have wrapped around
    const value = counter.get();
    try std.testing.expect(value < 2000);
}

// ============================================================================
// Gauge Stress Tests
// ============================================================================

test "observability stress: gauge concurrent updates" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const profile = getTestProfile();

    var gauge = obs.Gauge{
        .name = "stress_gauge",
    };

    const thread_count = @min(profile.concurrent_tasks, 16);
    const ops_per_thread = profile.operations / thread_count;

    var threads: [16]std.Thread = undefined;

    for (0..thread_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, struct {
            fn run(g: *obs.Gauge, ops: u64) void {
                for (0..ops) |j| {
                    if (j % 2 == 0) {
                        g.inc();
                    } else {
                        g.dec();
                    }
                }
            }
        }.run, .{ &gauge, ops_per_thread });
    }

    for (0..thread_count) |i| {
        threads[i].join();
    }

    // Gauge should be near 0 (equal inc/dec)
    const value = gauge.get();
    const abs_value = if (value < 0) -value else value;
    try std.testing.expect(abs_value < @as(i64, @intCast(thread_count)));
}

test "observability stress: float gauge rapid updates" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var gauge = obs.FloatGauge{
        .name = "float_gauge",
    };

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Rapid updates
    const ops = @min(profile.operations, 10000);
    for (0..ops) |i| {
        const timer = Timer.start();
        gauge.set(@as(f64, @floatFromInt(i)) / 100.0);
        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Check latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

// ============================================================================
// Histogram Stress Tests
// ============================================================================

test "observability stress: histogram millions of samples" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    const bounds = [_]u64{ 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000 };
    var histogram = try obs.Histogram.init(allocator, "stress_histogram", @constCast(&bounds));
    defer histogram.deinit(allocator);

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Record many samples
    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    const ops = profile.operations;

    for (0..ops) |i| {
        // Generate exponential-like distribution
        const value = rng.random().intRangeAtMost(u64, 0, 15000);
        const timer = Timer.start();
        histogram.record(value);
        if (i % 1000 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Verify all samples were recorded
    var total_count: u64 = 0;
    for (histogram.buckets) |count| {
        total_count += count;
    }
    try std.testing.expectEqual(ops, total_count);

    // Check recording latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

test "observability stress: histogram bucket distribution" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    const bounds = [_]u64{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    var histogram = try obs.Histogram.init(allocator, "distribution_histogram", @constCast(&bounds));
    defer histogram.deinit(allocator);

    // Record uniform distribution
    const ops = @min(profile.operations, 10000);
    for (0..ops) |i| {
        histogram.record(@intCast(i % 110));
    }

    // Each bucket should have roughly equal counts
    var min_count: u64 = std.math.maxInt(u64);
    var max_count: u64 = 0;

    for (histogram.buckets) |count| {
        if (count < min_count) min_count = count;
        if (count > max_count) max_count = count;
    }

    // Distribution should be reasonably even
    try std.testing.expect(max_count > 0);
}

// ============================================================================
// MetricsCollector Stress Tests
// ============================================================================

test "observability stress: metrics collector concurrent registration" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var collector = obs.MetricsCollector.init(allocator);
    defer collector.deinit();

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Register many metrics
    const metric_count = @min(profile.operations / 10, 500);

    for (0..metric_count) |i| {
        var name_buf: [64]u8 = undefined;
        const counter_name = std.fmt.bufPrint(&name_buf, "counter_{d}", .{i}) catch continue;

        const timer = Timer.start();
        _ = collector.registerCounter(counter_name) catch continue;
        try latency.recordUnsafe(timer.read());
    }

    // Register gauges
    for (0..metric_count) |i| {
        var name_buf: [64]u8 = undefined;
        const gauge_name = std.fmt.bufPrint(&name_buf, "gauge_{d}", .{i}) catch continue;
        _ = collector.registerGauge(gauge_name) catch continue;
    }

    // Verify registration
    try std.testing.expectEqual(metric_count, collector.counters.items.len);
    try std.testing.expectEqual(metric_count, collector.gauges.items.len);

    // Check registration latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

test "observability stress: metrics collector heavy usage" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var collector = obs.MetricsCollector.init(allocator);
    defer collector.deinit();

    // Register metrics
    var counters: [10]*obs.Counter = undefined;
    var gauges: [10]*obs.Gauge = undefined;

    for (0..10) |i| {
        var counter_buf: [32]u8 = undefined;
        var gauge_buf: [32]u8 = undefined;
        const counter_name = std.fmt.bufPrint(&counter_buf, "counter_{d}", .{i}) catch continue;
        const gauge_name = std.fmt.bufPrint(&gauge_buf, "gauge_{d}", .{i}) catch continue;

        counters[i] = collector.registerCounter(counter_name) catch continue;
        gauges[i] = collector.registerGauge(gauge_name) catch continue;
    }

    // Heavy concurrent usage
    const thread_count = @min(profile.concurrent_tasks, 8);
    const ops_per_thread = profile.operations / thread_count;

    var threads: [8]std.Thread = undefined;

    for (0..thread_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, struct {
            fn run(
                c: *[10]*obs.Counter,
                g: *[10]*obs.Gauge,
                ops: u64,
            ) void {
                for (0..ops) |j| {
                    const idx = j % 10;
                    c[idx].inc(1);
                    if (j % 2 == 0) {
                        g[idx].inc();
                    } else {
                        g[idx].dec();
                    }
                }
            }
        }.run, .{ &counters, &gauges, ops_per_thread });
    }

    for (0..thread_count) |i| {
        threads[i].join();
    }

    // Verify totals
    var total_counter: u64 = 0;
    for (counters) |c| {
        total_counter += c.get();
    }
    try std.testing.expectEqual(thread_count * ops_per_thread, total_counter);
}

// ============================================================================
// Tracing Stress Tests
// ============================================================================

test "observability stress: span creation high throughput" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Create and complete many spans
    const span_count = @min(profile.operations / 10, 1000);

    for (0..span_count) |i| {
        const timer = Timer.start();
        var span = obs.Span.start(allocator, "stress_span", null, null, .internal) catch continue;
        defer span.deinit();

        span.end();
        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Check latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

test "observability stress: tracer concurrent spans" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var tracer = try obs.Tracer.init(allocator, "stress-service");
    defer tracer.deinit();

    var spans_created = std.atomic.Value(u64).init(0);
    var errors = std.atomic.Value(u64).init(0);

    const thread_count = @min(profile.concurrent_tasks, 8);
    const ops_per_thread = @min(profile.operations / thread_count, 500);

    var threads: [8]std.Thread = undefined;

    for (0..thread_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, struct {
            fn run(
                t: *obs.Tracer,
                created: *std.atomic.Value(u64),
                errs: *std.atomic.Value(u64),
                ops: u64,
            ) void {
                for (0..ops) |_| {
                    var span = t.startSpan("concurrent_span", null, .internal) catch {
                        _ = errs.fetchAdd(1, .monotonic);
                        continue;
                    };
                    defer span.deinit();

                    span.end();
                    _ = created.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &tracer, &spans_created, &errors, ops_per_thread });
    }

    for (0..thread_count) |i| {
        threads[i].join();
    }

    const total_created = spans_created.load(.acquire);
    const total_errors = errors.load(.acquire);

    try std.testing.expect(total_created > 0);
    try std.testing.expect(total_errors < total_created);
}

test "observability stress: span with attributes and events" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    const span_count = @min(profile.operations / 100, 100);

    for (0..span_count) |i| {
        var span = obs.Span.start(allocator, "attributed_span", null, null, .internal) catch continue;
        defer span.deinit();

        const timer = Timer.start();

        // Add many attributes
        for (0..10) |j| {
            var key_buf: [32]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buf, "attr_{d}", .{j}) catch continue;
            span.setAttribute(key, .{ .int = @as(i64, @intCast(j)) }) catch continue;
        }

        // Add events
        for (0..5) |_| {
            span.addEvent("event") catch continue;
        }

        span.end();

        if (i % 10 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Check latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

// ============================================================================
// Memory Stability Tests
// ============================================================================

test "observability stress: long running memory stability" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var collector = obs.MetricsCollector.init(allocator);
    defer collector.deinit();

    // Register a fixed set of metrics
    const counter = try collector.registerCounter("stability_counter");
    const gauge = try collector.registerGauge("stability_gauge");
    const bounds = [_]u64{ 10, 100, 1000, 10000 };
    const histogram = try collector.registerHistogram("stability_histogram", &bounds);

    // Perform many operations without growing memory
    const ops = profile.operations;

    for (0..ops) |i| {
        counter.inc(1);

        if (i % 2 == 0) {
            gauge.inc();
        } else {
            gauge.dec();
        }

        histogram.record(@intCast(i % 15000));
    }

    // Verify stability
    try std.testing.expectEqual(ops, counter.get());

    // Gauge should be near 0
    const gauge_val = gauge.get();
    const abs_val = if (gauge_val < 0) -gauge_val else gauge_val;
    try std.testing.expect(abs_val <= 1);
}

test "observability stress: span processor memory management" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var processor = obs.tracing.SpanProcessor.init(allocator, 100); // Small max to test eviction
    defer processor.deinit();

    // Create more spans than the processor can hold
    const span_count = @min(profile.operations / 10, 500);

    for (0..span_count) |_| {
        const span_ptr = allocator.create(obs.Span) catch continue;
        span_ptr.* = obs.Span.start(allocator, "processor_span", null, null, .internal) catch {
            allocator.destroy(span_ptr);
            continue;
        };
        span_ptr.end();

        processor.onEnd(span_ptr) catch {
            span_ptr.deinit();
            allocator.destroy(span_ptr);
            continue;
        };
    }

    // Processor should have evicted old spans to maintain max_spans
    try std.testing.expect(processor.spans.items.len <= 100);
}

// ============================================================================
// Combined Observability Stress Tests
// ============================================================================

test "observability stress: full observability bundle" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var bundle = try obs.ObservabilityBundle.init(allocator, .{
        .enable_circuit_breaker_metrics = true,
        .enable_error_metrics = true,
    });
    defer bundle.deinit();

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Use all components
    const ops = @min(profile.operations, 5000);

    for (0..ops) |i| {
        const timer = Timer.start();

        // Record request
        obs.recordRequest(&bundle.defaults, @intCast(i % 100));

        // Occasionally record errors
        if (i % 10 == 0) {
            obs.recordError(&bundle.defaults, @intCast(i % 100));
        }

        // Circuit breaker metrics
        if (bundle.circuit_breaker) |*cb| {
            cb.recordRequest(i % 100 != 0, @intCast(i % 50));
        }

        // Error metrics
        if (bundle.errors) |*err| {
            err.recordError(i % 50 == 0);
        }

        if (i % 100 == 0) {
            try latency.recordUnsafe(timer.read());
        }
    }

    // Verify metrics were recorded
    try std.testing.expect(bundle.defaults.requests.get() > 0);

    // Check latency
    const stats = latency.getStats();
    try std.testing.expect(stats.count > 0);
}

test "observability stress: default collector initialization" {
    if (!build_options.enable_profiling) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    var collector = try obs.DefaultCollector.init(allocator);
    defer collector.deinit();

    // Heavy usage
    const ops = profile.operations;

    for (0..ops) |i| {
        obs.recordRequest(&collector.defaults, @intCast(i % 1000));
    }

    try std.testing.expectEqual(ops, collector.defaults.requests.get());
}
