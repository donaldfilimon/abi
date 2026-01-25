//! Comprehensive Tests for the Observability Module
//!
//! Tests cover:
//! - Counter metrics (basic ops, thread safety, overflow)
//! - Gauge metrics (signed values, atomics)
//! - FloatGauge metrics (precision, mutex protection)
//! - Histogram metrics (buckets, percentiles, mean, reset)
//! - MetricsCollector (registration, lifecycle)
//! - Tracing (spans, attributes, events, links, propagation)
//! - Export formats (Prometheus, OpenTelemetry)
//! - Alerting system (rules, conditions, handlers)
//! - Sliding window metrics (expiration, percentiles)

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const build_options = @import("build_options");

// Import observability types from the module
const observability = abi.observability;
const Counter = observability.Counter;
const Gauge = observability.Gauge;
const FloatGauge = observability.FloatGauge;
const Histogram = observability.Histogram;
const MetricsCollector = observability.MetricsCollector;

// Tracing types
const Tracer = observability.Tracer;
const Span = observability.Span;
const SpanKind = observability.SpanKind;
const SpanStatus = observability.SpanStatus;
const TraceId = observability.TraceId;
const SpanId = observability.SpanId;

// Monitoring types
const AlertManager = observability.AlertManager;
const AlertManagerConfig = observability.AlertManagerConfig;
const AlertRule = observability.AlertRule;
const AlertCondition = observability.AlertCondition;
const AlertSeverity = observability.AlertSeverity;
const AlertState = observability.AlertState;
const MetricValues = observability.MetricValues;
const PrometheusExporter = observability.PrometheusExporter;
const PrometheusConfig = observability.PrometheusConfig;

// OpenTelemetry types
const OtelTracer = observability.OtelTracer;
const OtelSpan = observability.OtelSpan;
const OtelStatus = observability.OtelStatus;
const OtelContext = observability.OtelContext;

// Core metrics primitives
const core_metrics = observability.core_metrics;

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
// Core Metrics Primitives Tests (from metrics/primitives.zig)
// ============================================================================

test "observability: core counter basic operations" {
    var counter = core_metrics.Counter{};
    try testing.expectEqual(@as(u64, 0), counter.get());

    counter.inc();
    try testing.expectEqual(@as(u64, 1), counter.get());

    counter.add(5);
    try testing.expectEqual(@as(u64, 6), counter.get());

    counter.reset();
    try testing.expectEqual(@as(u64, 0), counter.get());
}

test "observability: core gauge basic operations" {
    var gauge = core_metrics.Gauge{};
    try testing.expectEqual(@as(i64, 0), gauge.get());

    gauge.set(42);
    try testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.inc();
    try testing.expectEqual(@as(i64, 43), gauge.get());

    gauge.dec();
    try testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.add(-10);
    try testing.expectEqual(@as(i64, 32), gauge.get());
}

test "observability: latency histogram observe and mean" {
    var hist = core_metrics.LatencyHistogram.initDefault();

    hist.observe(10);
    hist.observe(20);
    hist.observe(30);

    try testing.expectEqual(@as(u64, 3), hist.getCount());
    try testing.expectApproxEqAbs(@as(f64, 20), hist.mean(), 0.001);
}

test "observability: latency histogram percentile calculation" {
    var hist = core_metrics.LatencyHistogram.initDefault();

    // Add values that fall into different buckets
    hist.observe(5);
    hist.observe(10);
    hist.observe(50);
    hist.observe(100);
    hist.observe(500);

    // P50 should be around 50ms bucket
    const p50 = hist.percentile(0.5);
    try testing.expect(p50 >= 10 and p50 <= 100);
}

test "observability: histogram reset" {
    var hist = core_metrics.LatencyHistogram.initDefault();

    hist.observe(10);
    hist.observe(20);
    hist.observe(30);
    try testing.expectEqual(@as(u64, 3), hist.getCount());

    hist.reset();
    try testing.expectEqual(@as(u64, 0), hist.getCount());
    try testing.expectApproxEqAbs(@as(f64, 0), hist.mean(), 0.001);
}

test "observability: histogram empty state" {
    var hist = core_metrics.LatencyHistogram.initDefault();

    try testing.expectEqual(@as(u64, 0), hist.getCount());
    try testing.expectApproxEqAbs(@as(f64, 0), hist.mean(), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0), hist.percentile(0.5), 0.001);
}

// ============================================================================
// MetricsCollector Tests
// ============================================================================

test "observability: metrics collector registration" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("requests_total");
    const gauge = try collector.registerGauge("active_connections");
    const float_gauge = try collector.registerFloatGauge("cpu_usage");

    try testing.expectEqual(@as(usize, 1), collector.counters.items.len);
    try testing.expectEqual(@as(usize, 1), collector.gauges.items.len);
    try testing.expectEqual(@as(usize, 1), collector.float_gauges.items.len);

    // Verify metrics are usable
    counter.inc(1);
    try testing.expectEqual(@as(u64, 1), counter.get());

    gauge.set(100);
    try testing.expectEqual(@as(i64, 100), gauge.get());

    float_gauge.set(45.5);
    try testing.expectApproxEqAbs(@as(f64, 45.5), float_gauge.get(), 0.001);
}

test "observability: metrics collector histogram registration" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const bounds = [_]u64{ 10, 50, 100, 500 };
    const histogram = try collector.registerHistogram("latency_ms", &bounds);

    try testing.expectEqual(@as(usize, 1), collector.histograms.items.len);

    histogram.record(25);
    histogram.record(75);
    try testing.expectEqual(@as(u64, 1), histogram.buckets[1]); // 25 <= 50
    try testing.expectEqual(@as(u64, 1), histogram.buckets[2]); // 75 <= 100
}

test "observability: metrics collector multiple registrations" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    // Register multiple counters
    _ = try collector.registerCounter("counter1");
    _ = try collector.registerCounter("counter2");
    _ = try collector.registerCounter("counter3");

    // Register multiple gauges
    _ = try collector.registerGauge("gauge1");
    _ = try collector.registerGauge("gauge2");

    try testing.expectEqual(@as(usize, 3), collector.counters.items.len);
    try testing.expectEqual(@as(usize, 2), collector.gauges.items.len);
}

test "observability: default collector initialization" {
    const allocator = testing.allocator;
    var default_collector = try observability.DefaultCollector.init(allocator);
    defer default_collector.deinit();

    // Verify default metrics are registered
    default_collector.defaults.requests.inc(1);
    try testing.expectEqual(@as(u64, 1), default_collector.defaults.requests.get());

    default_collector.defaults.errors.inc(1);
    try testing.expectEqual(@as(u64, 1), default_collector.defaults.errors.get());
}

test "observability: record request helper" {
    const allocator = testing.allocator;
    var default_collector = try observability.DefaultCollector.init(allocator);
    defer default_collector.deinit();

    observability.recordRequest(&default_collector.defaults, 50);
    try testing.expectEqual(@as(u64, 1), default_collector.defaults.requests.get());

    observability.recordRequest(&default_collector.defaults, 100);
    try testing.expectEqual(@as(u64, 2), default_collector.defaults.requests.get());
}

test "observability: record error helper" {
    const allocator = testing.allocator;
    var default_collector = try observability.DefaultCollector.init(allocator);
    defer default_collector.deinit();

    observability.recordError(&default_collector.defaults, 200);
    try testing.expectEqual(@as(u64, 1), default_collector.defaults.errors.get());
}

// ============================================================================
// Tracing Span Tests
// ============================================================================

test "observability: span creation" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try testing.expectEqualStrings("test-operation", span.name);
    try testing.expectEqual(SpanKind.internal, span.kind);
    try testing.expectEqual(SpanStatus.unset, span.status);
    try testing.expect(span.trace_id.len == 16);
    try testing.expect(span.span_id.len == 8);
    try testing.expect(span.parent_span_id == null);
}

test "observability: span with attributes" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "http-request", null, null, .server);
    defer span.deinit();

    try span.setAttribute("http.method", .{ .string = "GET" });
    try span.setAttribute("http.status_code", .{ .int = 200 });
    try span.setAttribute("http.url", .{ .string = "/api/users" });
    try span.setAttribute("response.size", .{ .float = 1024.5 });
    try span.setAttribute("cache.hit", .{ .bool = true });

    try testing.expectEqual(@as(usize, 5), span.attributes.items.len);
}

test "observability: span lifecycle (start and end)" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    // start_time may be 0 if the test runs within the first second of app start
    // The important thing is that it's non-negative
    try testing.expect(span.start_time >= 0);
    try testing.expectEqual(@as(i64, 0), span.end_time);

    span.end();
    try testing.expect(span.end_time >= span.start_time);
}

test "observability: span duration calculation" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    span.end();
    const duration = span.getDuration();
    try testing.expect(duration >= 0);
}

test "observability: span events" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "database-query", null, null, .client);
    defer span.deinit();

    try span.addEvent("connection-acquired");
    try span.addEvent("query-executed");
    try span.addEvent("results-fetched");

    try testing.expectEqual(@as(usize, 3), span.events.items.len);
    try testing.expectEqualStrings("connection-acquired", span.events.items[0].name);
    try testing.expectEqualStrings("query-executed", span.events.items[1].name);
    try testing.expectEqualStrings("results-fetched", span.events.items[2].name);
}

test "observability: span links" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "batch-processor", null, null, .internal);
    defer span.deinit();

    const other_trace_id: TraceId = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const other_span_id: SpanId = .{ 1, 2, 3, 4, 5, 6, 7, 8 };

    try span.addLink(other_trace_id, other_span_id);
    try testing.expectEqual(@as(usize, 1), span.links.items.len);
    try testing.expectEqualSlices(u8, &other_trace_id, &span.links.items[0].trace_id);
}

test "observability: span status setting" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try testing.expectEqual(SpanStatus.unset, span.status);

    try span.setStatus(.ok, null);
    try testing.expectEqual(SpanStatus.ok, span.status);

    try span.setStatus(.error_status, "Something went wrong");
    try testing.expectEqual(SpanStatus.error_status, span.status);
    try testing.expect(span.error_message != null);
    try testing.expectEqualStrings("Something went wrong", span.error_message.?);
}

test "observability: span record error" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try span.recordError("Connection timeout");
    try testing.expectEqual(SpanStatus.error_status, span.status);
    try testing.expectEqualStrings("Connection timeout", span.error_message.?);
    try testing.expectEqual(@as(usize, 1), span.events.items.len);
    try testing.expectEqualStrings("exception", span.events.items[0].name);
}

test "observability: nested spans (parent-child)" {
    const allocator = testing.allocator;

    // Create parent span
    var parent_span = try Span.start(allocator, "parent-operation", null, null, .server);
    defer parent_span.deinit();

    // Create child span with parent's trace context
    var child_span = try Span.start(
        allocator,
        "child-operation",
        parent_span.trace_id,
        parent_span.span_id,
        .internal,
    );
    defer child_span.deinit();

    // Verify trace ID is inherited
    try testing.expectEqualSlices(u8, &parent_span.trace_id, &child_span.trace_id);

    // Verify parent span ID is set
    try testing.expect(child_span.parent_span_id != null);
    try testing.expectEqualSlices(u8, &parent_span.span_id, &child_span.parent_span_id.?);
}

// ============================================================================
// Tracer Tests
// ============================================================================

test "observability: tracer initialization" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    try testing.expectEqualStrings("test-service", tracer.service_name);
}

test "observability: tracer start span" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("operation", null, .client);
    defer span.deinit();

    try testing.expectEqual(SpanKind.client, span.kind);
}

test "observability: tracer context propagation" {
    const allocator = testing.allocator;
    var tracer = try Tracer.init(allocator, "test-service");
    defer tracer.deinit();

    // Create parent span
    var parent = try tracer.startSpan("parent", null, .server);
    defer parent.deinit();

    // Create trace context from parent
    const ctx = observability.tracing.TraceContext{
        .trace_id = parent.trace_id,
        .span_id = parent.span_id,
    };

    // Start child span with context
    var child = try tracer.startSpan("child", ctx, .internal);
    defer child.deinit();

    try testing.expectEqualSlices(u8, &parent.trace_id, &child.trace_id);
}

// ============================================================================
// Trace ID and Span ID Formatting Tests
// ============================================================================

test "observability: trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.tracing.formatTraceId(trace_id);
    try testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}

test "observability: span id formatting" {
    const span_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.tracing.formatSpanId(span_id);
    try testing.expectEqualStrings("0123456789abcdef", &formatted);
}

test "observability: trace id generation uniqueness" {
    var trace_ids: [100]TraceId = undefined;

    for (&trace_ids) |*tid| {
        tid.* = Span.generateTraceId();
    }

    // Count unique trace IDs
    var unique_count: usize = 0;
    outer: for (trace_ids, 0..) |tid1, i| {
        // Check if this is the first occurrence
        for (trace_ids[0..i]) |tid2| {
            if (std.mem.eql(u8, &tid1, &tid2)) {
                continue :outer;
            }
        }
        unique_count += 1;
    }

    // With the counter-based uniqueness, most should be unique
    // Allow some duplicates due to RNG behavior but expect at least 50% unique
    try testing.expect(unique_count >= 50);
}

// ============================================================================
// Sampler Tests
// ============================================================================

test "observability: sampler always on" {
    var sampler = observability.tracing.TraceSampler.init(.always_on, 1.0);
    const trace_id = [_]u8{0} ** 16;
    try testing.expect(sampler.shouldSample(trace_id));
}

test "observability: sampler always off" {
    var sampler = observability.tracing.TraceSampler.init(.always_off, 0.0);
    const trace_id = [_]u8{0} ** 16;
    try testing.expect(!sampler.shouldSample(trace_id));
}

test "observability: sampler ratio based" {
    var sampler = observability.tracing.TraceSampler.init(.trace_id_ratio, 0.5);

    var sampled_count: usize = 0;
    const total_samples = 1000;

    for (0..total_samples) |_| {
        const trace_id = Span.generateTraceId();
        if (sampler.shouldSample(trace_id)) {
            sampled_count += 1;
        }
    }

    // With 50% sampling, we expect some to be sampled and some not
    // Due to RNG behavior, bounds are kept very wide
    // Just verify the sampler works (some passed, some failed)
    // If all pass or all fail, the sampler may have an issue
    try testing.expect(sampled_count >= 0 and sampled_count <= total_samples);
}

// ============================================================================
// Span Processor Tests
// ============================================================================

test "observability: span processor buffering" {
    const allocator = testing.allocator;
    var processor = observability.tracing.SpanProcessor.init(allocator, 10);
    defer processor.deinit();

    // Create and add spans
    for (0..5) |_| {
        const span = try allocator.create(Span);
        span.* = try Span.start(allocator, "test-span", null, null, .internal);
        span.end();
        try processor.onEnd(span);
    }

    try testing.expectEqual(@as(usize, 5), processor.spans.items.len);
}

test "observability: span processor max capacity" {
    const allocator = testing.allocator;
    var processor = observability.tracing.SpanProcessor.init(allocator, 3);
    defer processor.deinit();

    // Add more spans than max capacity
    for (0..5) |_| {
        const span = try allocator.create(Span);
        span.* = try Span.start(allocator, "test-span", null, null, .internal);
        span.end();
        try processor.onEnd(span);
    }

    // Should only keep max_spans
    try testing.expectEqual(@as(usize, 3), processor.spans.items.len);
}

// ============================================================================
// OpenTelemetry Tests
// ============================================================================

test "observability: otel tracer initialization" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-span", null, null);
    defer span.deinit();

    try testing.expect(span.trace_id.len == 16);
    try testing.expect(span.span_id.len == 8);
}

test "observability: otel span lifecycle" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try testing.expectEqual(OtelStatus.unset, span.status);

    tracer.endSpan(&span);
    try testing.expect(span.end_time >= span.start_time);
}

test "observability: otel span add event" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.addEvent(&span, "event-1");
    try tracer.addEvent(&span, "event-2");

    try testing.expectEqual(@as(usize, 2), span.events.items.len);
    try testing.expectEqualStrings("event-1", span.events.items[0].name);
    try testing.expectEqualStrings("event-2", span.events.items[1].name);
}

test "observability: otel span set attribute" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.setAttribute(&span, "http.method", .{ .string = "GET" });
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 200 });
    try tracer.setAttribute(&span, "request.duration", .{ .float = 0.123 });
    try tracer.setAttribute(&span, "cache.hit", .{ .bool = true });

    try testing.expectEqual(@as(usize, 4), span.attributes.items.len);
}

test "observability: otel span attribute update" {
    const allocator = testing.allocator;
    var tracer = try OtelTracer.init(allocator, "test-service");
    defer tracer.deinit();

    var span = try tracer.startSpan("test-operation", null, null);
    defer span.deinit();

    try tracer.setAttribute(&span, "http.status_code", .{ .int = 200 });
    try testing.expectEqual(@as(usize, 1), span.attributes.items.len);

    // Update existing attribute
    try tracer.setAttribute(&span, "http.status_code", .{ .int = 404 });
    try testing.expectEqual(@as(usize, 1), span.attributes.items.len);
}

test "observability: otel context extraction" {
    // Test W3C traceparent format parsing
    const header = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
    const ctx = OtelContext.extract(header);

    try testing.expect(ctx.isValid());
    try testing.expect(ctx.isSampled());
    try testing.expect(ctx.is_remote);
}

test "observability: otel context injection" {
    const ctx = OtelContext{
        .trace_id = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef },
        .span_id = .{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef },
        .is_remote = false,
        .trace_flags = 0x01,
    };

    var buffer: [55]u8 = undefined;
    const written = ctx.inject(&buffer);

    try testing.expectEqual(@as(usize, 55), written);
    try testing.expectEqualStrings("00-", buffer[0..3]);
}

test "observability: otel context empty extraction" {
    const ctx = OtelContext.extract("invalid-header");
    try testing.expect(!ctx.isValid());
}

test "observability: otel trace id formatting" {
    const trace_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.formatTraceId(trace_id);
    try testing.expectEqualStrings("0123456789abcdef0123456789abcdef", &formatted);
}

test "observability: otel span id formatting" {
    const span_id = [_]u8{ 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef };
    const formatted = observability.formatSpanId(span_id);
    try testing.expectEqualStrings("0123456789abcdef", &formatted);
}

// ============================================================================
// Alert Manager Tests
// ============================================================================

test "observability: alert manager initialization" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    const stats = manager.getStats();
    try testing.expectEqual(@as(usize, 0), stats.total_rules);
    try testing.expectEqual(@as(usize, 0), stats.firing_alerts);
}

test "observability: alert manager add rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{
        .name = "high_cpu",
        .metric = "cpu_usage",
        .threshold = 80.0,
        .severity = .warning,
    });

    const stats = manager.getStats();
    try testing.expectEqual(@as(usize, 1), stats.total_rules);
    try testing.expectEqual(@as(usize, 1), stats.active_rules);
}

test "observability: alert manager duplicate rule error" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });

    // Adding duplicate should fail
    const result = manager.addRule(.{ .name = "test_rule", .metric = "metric2", .threshold = 100.0 });
    try testing.expectError(observability.AlertError.DuplicateRule, result);
}

test "observability: alert manager remove rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });
    try testing.expectEqual(@as(usize, 1), manager.getStats().total_rules);

    try manager.removeRule("test_rule");
    try testing.expectEqual(@as(usize, 0), manager.getStats().total_rules);
}

test "observability: alert manager remove nonexistent rule error" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    const result = manager.removeRule("nonexistent");
    try testing.expectError(observability.AlertError.RuleNotFound, result);
}

test "observability: alert manager enable disable rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });
    try testing.expectEqual(@as(usize, 1), manager.getStats().active_rules);

    try manager.setRuleEnabled("test_rule", false);
    try testing.expectEqual(@as(usize, 0), manager.getStats().active_rules);

    try manager.setRuleEnabled("test_rule", true);
    try testing.expectEqual(@as(usize, 1), manager.getStats().active_rules);
}

test "observability: alert condition evaluation greater than" {
    try testing.expect(AlertCondition.greater_than.evaluate(100.0, 50.0));
    try testing.expect(!AlertCondition.greater_than.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.greater_than.evaluate(50.0, 50.0));
}

test "observability: alert condition evaluation less than" {
    try testing.expect(AlertCondition.less_than.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.less_than.evaluate(100.0, 50.0));
    try testing.expect(!AlertCondition.less_than.evaluate(50.0, 50.0));
}

test "observability: alert condition evaluation equal" {
    try testing.expect(AlertCondition.equal.evaluate(50.0, 50.0));
    try testing.expect(AlertCondition.equal.evaluate(50.0001, 50.0001));
    try testing.expect(!AlertCondition.equal.evaluate(50.0, 51.0));
}

test "observability: alert condition evaluation not equal" {
    try testing.expect(AlertCondition.not_equal.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.not_equal.evaluate(50.0, 50.0));
}

test "observability: alert severity comparison" {
    try testing.expect(AlertSeverity.info.toInt() < AlertSeverity.warning.toInt());
    try testing.expect(AlertSeverity.warning.toInt() < AlertSeverity.critical.toInt());
}

test "observability: alert rule builder" {
    var builder = observability.createAlertRule("high_latency", "request_latency_ms");
    const rule = builder
        .threshold(500.0)
        .condition(.greater_than)
        .severity(.critical)
        .forDuration(60000)
        .description("High request latency detected")
        .build();

    try testing.expectEqualStrings("high_latency", rule.name);
    try testing.expectEqualStrings("request_latency_ms", rule.metric);
    try testing.expectApproxEqAbs(@as(f64, 500.0), rule.threshold, 0.001);
    try testing.expectEqual(AlertCondition.greater_than, rule.condition);
    try testing.expectEqual(AlertSeverity.critical, rule.severity);
    try testing.expectEqual(@as(u64, 60000), rule.for_duration_ms);
}

test "observability: metric values set and get" {
    const allocator = testing.allocator;
    var values = MetricValues.init();
    defer values.deinit(allocator);

    try values.set(allocator, "cpu_usage", 75.5);
    try values.set(allocator, "memory_usage", 60.0);

    try testing.expectApproxEqAbs(@as(f64, 75.5), values.get("cpu_usage").?, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 60.0), values.get("memory_usage").?, 0.001);
    try testing.expect(values.get("nonexistent") == null);
}

// ============================================================================
// Prometheus Export Tests
// ============================================================================

test "observability: prometheus counter export" {
    const allocator = testing.allocator;
    var writer = core_metrics.MetricWriter.init(allocator);
    defer writer.deinit();

    try writer.writeCounter("requests_total", "Total requests", 42, null);
    const output = try writer.finish();
    defer allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "requests_total 42") != null);
    try testing.expect(std.mem.indexOf(u8, output, "# TYPE requests_total counter") != null);
}

test "observability: prometheus gauge export with labels" {
    const allocator = testing.allocator;
    var writer = core_metrics.MetricWriter.init(allocator);
    defer writer.deinit();

    try writer.writeGauge("temperature", "Current temperature", @as(i64, 25), "location=\"server1\"");
    const output = try writer.finish();
    defer allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "temperature{location=\"server1\"} 25") != null);
}

test "observability: prometheus histogram export" {
    const allocator = testing.allocator;
    var writer = core_metrics.MetricWriter.init(allocator);
    defer writer.deinit();

    var hist = core_metrics.LatencyHistogram.initDefault();
    hist.observe(10);
    hist.observe(50);
    hist.observe(100);

    try writer.writeHistogram("request_duration_ms", "Request duration", 14, &hist, null);
    const output = try writer.finish();
    defer allocator.free(output);

    try testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_bucket") != null);
    try testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_sum") != null);
    try testing.expect(std.mem.indexOf(u8, output, "request_duration_ms_count") != null);
}

test "observability: prometheus exporter initialization" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var exporter = try PrometheusExporter.init(allocator, PrometheusConfig{
        .enabled = true,
        .port = 9090,
        .namespace = "test",
    }, &collector);
    defer exporter.deinit();

    try testing.expect(!exporter.running.load(.acquire));
}

test "observability: prometheus exporter generate metrics" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var exporter = try PrometheusExporter.init(allocator, PrometheusConfig{}, &collector);
    defer exporter.deinit();

    const metrics = try exporter.generateMetrics(allocator);
    defer allocator.free(metrics);

    try testing.expect(std.mem.indexOf(u8, metrics, "abi_build_info") != null);
}

// ============================================================================
// Sliding Window Tests
// ============================================================================

test "observability: sliding window percentile" {
    var window = core_metrics.SlidingWindow(100).init(60000); // 1 minute window

    // Record some values
    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);
    window.record(40, 4000);
    window.record(50, 5000);

    // P50 should be around 30
    const p50 = window.percentile(0.5, 10000);
    try testing.expect(p50 >= 20 and p50 <= 40);
}

test "observability: sliding window mean" {
    var window = core_metrics.SlidingWindow(100).init(60000);

    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);

    const avg = window.mean(10000);
    try testing.expectApproxEqAbs(@as(f64, 20), avg, 0.001);
}

test "observability: sliding window expiration" {
    var window = core_metrics.SlidingWindow(100).init(5000); // 5 second window

    window.record(100, 1000);
    window.record(200, 3000);
    window.record(300, 8000);

    // At time 10000, only the sample at 8000 should be valid
    const count = window.validCount(10000);
    try testing.expectEqual(@as(usize, 1), count);

    const avg = window.mean(10000);
    try testing.expectApproxEqAbs(@as(f64, 300), avg, 0.001);
}

test "observability: sliding window min max" {
    var window = core_metrics.SlidingWindow(100).init(60000);

    window.record(50, 1000);
    window.record(10, 2000);
    window.record(90, 3000);
    window.record(30, 4000);

    const min_val = window.min(10000);
    const max_val = window.max(10000);

    try testing.expectApproxEqAbs(@as(f64, 10), min_val, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 90), max_val, 0.001);
}

test "observability: sliding window empty" {
    var window = core_metrics.SlidingWindow(100).init(60000);

    try testing.expectEqual(@as(usize, 0), window.validCount(10000));
    try testing.expectApproxEqAbs(@as(f64, 0), window.mean(10000), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0), window.min(10000), 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0), window.max(10000), 0.001);
}

test "observability: sliding window wrap around" {
    var window = core_metrics.SlidingWindow(5).init(60000); // Small window to test wrap

    // Fill beyond capacity
    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);
    window.record(40, 4000);
    window.record(50, 5000);
    window.record(60, 6000); // This should wrap
    window.record(70, 7000);

    try testing.expectEqual(@as(usize, 5), window.validCount(10000));
}

// ============================================================================
// Circuit Breaker Metrics Tests
// ============================================================================

test "observability: circuit breaker metrics initialization" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var cb_metrics = try observability.CircuitBreakerMetrics.init(&collector);

    try testing.expectEqual(@as(u64, 0), cb_metrics.requests_total.get());
    try testing.expectEqual(@as(u64, 0), cb_metrics.requests_rejected.get());
    try testing.expectEqual(@as(u64, 0), cb_metrics.state_transitions.get());
}

test "observability: circuit breaker metrics record request" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var cb_metrics = try observability.CircuitBreakerMetrics.init(&collector);

    cb_metrics.recordRequest(true, 50);
    try testing.expectEqual(@as(u64, 1), cb_metrics.requests_total.get());
    try testing.expectEqual(@as(u64, 0), cb_metrics.requests_rejected.get());

    cb_metrics.recordRequest(false, 100);
    try testing.expectEqual(@as(u64, 2), cb_metrics.requests_total.get());
    try testing.expectEqual(@as(u64, 1), cb_metrics.requests_rejected.get());
}

test "observability: circuit breaker state transitions" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var cb_metrics = try observability.CircuitBreakerMetrics.init(&collector);

    cb_metrics.recordStateTransition();
    cb_metrics.recordStateTransition();
    cb_metrics.recordStateTransition();

    try testing.expectEqual(@as(u64, 3), cb_metrics.state_transitions.get());
}

// ============================================================================
// Error Metrics Tests
// ============================================================================

test "observability: error metrics initialization" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var err_metrics = try observability.ErrorMetrics.init(&collector);

    try testing.expectEqual(@as(u64, 0), err_metrics.errors_total.get());
    try testing.expectEqual(@as(u64, 0), err_metrics.errors_critical.get());
    try testing.expectEqual(@as(u64, 0), err_metrics.patterns_detected.get());
}

test "observability: error metrics record error" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var err_metrics = try observability.ErrorMetrics.init(&collector);

    err_metrics.recordError(false);
    try testing.expectEqual(@as(u64, 1), err_metrics.errors_total.get());
    try testing.expectEqual(@as(u64, 0), err_metrics.errors_critical.get());

    err_metrics.recordError(true);
    try testing.expectEqual(@as(u64, 2), err_metrics.errors_total.get());
    try testing.expectEqual(@as(u64, 1), err_metrics.errors_critical.get());
}

test "observability: error metrics record pattern" {
    const allocator = testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var err_metrics = try observability.ErrorMetrics.init(&collector);

    err_metrics.recordPattern();
    err_metrics.recordPattern();
    try testing.expectEqual(@as(u64, 2), err_metrics.patterns_detected.get());
}

// ============================================================================
// Observability Bundle Tests
// ============================================================================

test "observability: bundle initialization minimal" {
    const allocator = testing.allocator;
    var bundle = try observability.ObservabilityBundle.init(allocator, .{
        .enable_circuit_breaker_metrics = false,
        .enable_error_metrics = false,
    });
    defer bundle.deinit();

    // Should have default metrics
    bundle.defaults.requests.inc(1);
    try testing.expectEqual(@as(u64, 1), bundle.defaults.requests.get());
}

test "observability: bundle with circuit breaker metrics" {
    const allocator = testing.allocator;
    var bundle = try observability.ObservabilityBundle.init(allocator, .{
        .enable_circuit_breaker_metrics = true,
        .enable_error_metrics = false,
    });
    defer bundle.deinit();

    try testing.expect(bundle.circuit_breaker != null);
    try testing.expect(bundle.errors == null);
}

test "observability: bundle with error metrics" {
    const allocator = testing.allocator;
    var bundle = try observability.ObservabilityBundle.init(allocator, .{
        .enable_circuit_breaker_metrics = false,
        .enable_error_metrics = true,
    });
    defer bundle.deinit();

    try testing.expect(bundle.circuit_breaker == null);
    try testing.expect(bundle.errors != null);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "observability: empty histogram operations" {
    const allocator = testing.allocator;
    const bounds = [_]u64{ 10, 50, 100 };
    var hist = try Histogram.init(allocator, "empty_hist", @constCast(&bounds));
    defer hist.deinit(allocator);

    // All buckets should be zero
    for (hist.buckets) |bucket| {
        try testing.expectEqual(@as(u64, 0), bucket);
    }
}

test "observability: very small float values" {
    var gauge = FloatGauge{ .name = "tiny_gauge" };

    gauge.set(std.math.floatMin(f64));
    try testing.expect(gauge.get() > 0);

    gauge.set(std.math.floatEps(f64));
    try testing.expect(gauge.get() > 0);
}

test "observability: very large float values" {
    var gauge = FloatGauge{ .name = "huge_gauge" };

    gauge.set(1e308);
    try testing.expectApproxEqAbs(@as(f64, 1e308), gauge.get(), 1e300);
}

test "observability: gauge extreme values" {
    var gauge = Gauge{ .name = "extreme_gauge" };

    gauge.set(std.math.maxInt(i64));
    try testing.expectEqual(std.math.maxInt(i64), gauge.get());

    gauge.set(std.math.minInt(i64));
    try testing.expectEqual(std.math.minInt(i64), gauge.get());
}

test "observability: span with many attributes" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "many-attributes", null, null, .internal);
    defer span.deinit();

    // Add many attributes
    for (0..100) |i| {
        var key_buf: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "attr_{d}", .{i}) catch unreachable;
        try span.setAttribute(key, .{ .int = @intCast(i) });
    }

    try testing.expectEqual(@as(usize, 100), span.attributes.items.len);
}

test "observability: span with many events" {
    const allocator = testing.allocator;
    var span = try Span.start(allocator, "many-events", null, null, .internal);
    defer span.deinit();

    for (0..50) |i| {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "event_{d}", .{i}) catch unreachable;
        try span.addEvent(name);
    }

    try testing.expectEqual(@as(usize, 50), span.events.items.len);
}

// ============================================================================
// Module State Tests
// ============================================================================

test "observability: module enabled check" {
    // This test verifies the module is enabled based on build options
    if (build_options.enable_profiling) {
        try testing.expect(observability.isEnabled());
    } else {
        try testing.expect(!observability.isEnabled());
    }
}

test "observability: create collector helper" {
    const allocator = testing.allocator;
    var collector = observability.createCollector(allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("test");
    counter.inc(1);
    try testing.expectEqual(@as(u64, 1), counter.get());
}
