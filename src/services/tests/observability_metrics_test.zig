//! Observability Metrics Tests — Core metrics primitives, collectors,
//! Prometheus export, sliding window, circuit breaker, error metrics, and bundles.

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

const observability = abi.features.observability;
const Counter = observability.Counter;
const Gauge = observability.Gauge;
const FloatGauge = observability.FloatGauge;
const Histogram = observability.Histogram;
const MetricsCollector = observability.MetricsCollector;
const PrometheusExporter = observability.PrometheusExporter;
const PrometheusConfig = observability.PrometheusConfig;
const core_metrics = observability.core_metrics;

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
