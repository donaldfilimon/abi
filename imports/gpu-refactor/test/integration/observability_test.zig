//! Integration Tests: Observability
//!
//! Verifies observability module exports: profiling types, metrics types,
//! tracing types, monitoring backends, and the observability bundle.

const std = @import("std");
const abi = @import("abi");

const observability = abi.observability;

// ============================================================================
// Module Exports
// ============================================================================

test "observability: module exports metric types" {
    _ = observability.Counter;
    _ = observability.Gauge;
    _ = observability.FloatGauge;
    _ = observability.Histogram;
    _ = observability.MetricsCollector;
    _ = observability.DefaultMetrics;
    _ = observability.DefaultCollector;
    _ = observability.CircuitBreakerMetrics;
    _ = observability.ErrorMetrics;
}

test "observability: module exports tracing types" {
    _ = observability.tracing;
    _ = observability.Tracer;
    _ = observability.Span;
    _ = observability.TraceId;
    _ = observability.SpanId;
    _ = observability.SpanKind;
    _ = observability.SpanStatus;
    _ = observability.AttributeValue;
    _ = observability.SpanAttribute;
    _ = observability.SpanEvent;
    _ = observability.SpanLink;
    _ = observability.TraceContext;
    _ = observability.Propagator;
    _ = observability.PropagationFormat;
    _ = observability.TraceSampler;
    _ = observability.SpanProcessor;
    _ = observability.SpanExporter;
    _ = observability.SamplerType;
}

test "observability: module exports monitoring types" {
    _ = observability.monitoring;
    _ = observability.AlertManager;
    _ = observability.AlertManagerConfig;
    _ = observability.AlertRule;
    _ = observability.AlertRuleBuilder;
    _ = observability.Alert;
    _ = observability.AlertState;
    _ = observability.AlertSeverity;
    _ = observability.AlertCondition;
    _ = observability.AlertError;
    _ = observability.AlertStats;
    _ = observability.AlertCallback;
    _ = observability.AlertHandler;
    _ = observability.MetricValues;
}

test "observability: module exports prometheus types" {
    _ = observability.PrometheusExporter;
    _ = observability.PrometheusConfig;
    _ = observability.PrometheusFormatter;
}

test "observability: module exports statsd types" {
    _ = observability.StatsDClient;
    _ = observability.StatsDConfig;
    _ = observability.StatsDError;
}

test "observability: module exports otel types" {
    _ = observability.otel;
    _ = observability.OtelExporter;
    _ = observability.OtelConfig;
    _ = observability.OtelTracer;
    _ = observability.OtelSpan;
    _ = observability.OtelSpanKind;
    _ = observability.OtelContext;
    _ = observability.OtelMetric;
    _ = observability.OtelMetricType;
    _ = observability.OtelAttribute;
    _ = observability.OtelAttributeValue;
    _ = observability.OtelEvent;
    _ = observability.OtelStatus;
}

test "observability: module exports bundle types" {
    _ = observability.ObservabilityBundle;
    _ = observability.BundleConfig;
}

test "observability: module exports shared types" {
    _ = observability.types;
    _ = observability.MetricsConfig;
    _ = observability.MetricsSummary;
    _ = observability.Error;
    _ = observability.MonitoringError;
}

// ============================================================================
// Lifecycle
// ============================================================================

test "observability: isEnabled reflects build option" {
    const enabled = observability.isEnabled();
    _ = enabled;
}

test "observability: isInitialized initially false" {
    _ = observability.isInitialized();
}

// ============================================================================
// Counter
// ============================================================================

test "observability: Counter init and increment" {
    var counter = observability.Counter{ .name = "integration_test_counter" };

    try std.testing.expectEqual(@as(u64, 0), counter.get());

    counter.inc(1);
    try std.testing.expectEqual(@as(u64, 1), counter.get());

    counter.inc(99);
    try std.testing.expectEqual(@as(u64, 100), counter.get());

    counter.reset();
    try std.testing.expectEqual(@as(u64, 0), counter.get());
}

// ============================================================================
// Gauge
// ============================================================================

test "observability: Gauge set and add" {
    var gauge = observability.Gauge{ .name = "integration_test_gauge" };

    gauge.set(42);
    try std.testing.expectEqual(@as(i64, 42), gauge.get());

    gauge.add(-10);
    try std.testing.expectEqual(@as(i64, 32), gauge.get());

    gauge.inc();
    gauge.dec();
    try std.testing.expectEqual(@as(i64, 32), gauge.get());
}

// ============================================================================
// FloatGauge
// ============================================================================

test "observability: FloatGauge set and add" {
    var gauge = observability.FloatGauge{ .name = "integration_float_gauge" };

    gauge.set(3.14);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), gauge.get(), 0.001);

    gauge.add(1.86);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), gauge.get(), 0.001);
}

// ============================================================================
// Histogram
// ============================================================================

test "observability: Histogram init and record" {
    var bounds = [_]u64{ 10, 50, 100, 500 };
    var hist = try observability.Histogram.init(std.testing.allocator, "integration_hist", &bounds);
    defer hist.deinit(std.testing.allocator);

    hist.record(5);
    hist.record(25);
    hist.record(75);
    hist.record(200);
    hist.record(1000);

    try std.testing.expectEqual(@as(u64, 1), hist.buckets[0]); // <=10
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[1]); // <=50
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[2]); // <=100
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[3]); // <=500
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[4]); // overflow
}

// ============================================================================
// MetricsCollector
// ============================================================================

test "observability: MetricsCollector registers mixed metrics" {
    var collector = observability.MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("int_req_total");
    const gauge = try collector.registerGauge("int_active");
    const float_gauge = try collector.registerFloatGauge("int_cpu");

    counter.inc(10);
    gauge.set(5);
    float_gauge.set(0.75);

    try std.testing.expectEqual(@as(u64, 10), counter.get());
    try std.testing.expectEqual(@as(i64, 5), gauge.get());
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), float_gauge.get(), 0.001);
}

// ============================================================================
// DefaultCollector
// ============================================================================

test "observability: DefaultCollector records requests and errors" {
    var dc = try observability.DefaultCollector.init(std.testing.allocator);
    defer dc.deinit();

    observability.recordRequest(&dc.defaults, 50);
    observability.recordRequest(&dc.defaults, 100);
    observability.recordError(&dc.defaults, 500);

    try std.testing.expectEqual(@as(u64, 2), dc.defaults.requests.get());
    try std.testing.expectEqual(@as(u64, 1), dc.defaults.errors.get());
}

// ============================================================================
// CircuitBreakerMetrics
// ============================================================================

test "observability: CircuitBreakerMetrics tracking" {
    var collector = observability.MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    var cb = try observability.CircuitBreakerMetrics.init(&collector);

    cb.recordRequest(true, 50);
    cb.recordRequest(true, 100);
    cb.recordRequest(false, 0);
    cb.recordStateTransition();

    try std.testing.expectEqual(@as(u64, 3), cb.requests_total.get());
    try std.testing.expectEqual(@as(u64, 1), cb.requests_rejected.get());
    try std.testing.expectEqual(@as(u64, 1), cb.state_transitions.get());
}

// ============================================================================
// ErrorMetrics
// ============================================================================

test "observability: ErrorMetrics tracking" {
    var collector = observability.MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    var err_metrics = try observability.ErrorMetrics.init(&collector);

    err_metrics.recordError(false);
    err_metrics.recordError(true);
    err_metrics.recordPattern();

    try std.testing.expectEqual(@as(u64, 2), err_metrics.errors_total.get());
    try std.testing.expectEqual(@as(u64, 1), err_metrics.errors_critical.get());
    try std.testing.expectEqual(@as(u64, 1), err_metrics.patterns_detected.get());
}

// ============================================================================
// BundleConfig
// ============================================================================

test "observability: BundleConfig defaults" {
    const config = observability.BundleConfig{};

    try std.testing.expect(config.enable_circuit_breaker_metrics);
    try std.testing.expect(config.enable_error_metrics);
    try std.testing.expect(config.prometheus == null);
    try std.testing.expect(config.otel == null);
}

// ============================================================================
// Span Kind Enum
// ============================================================================

test "observability: SpanKind variants" {
    const internal = observability.SpanKind.internal;
    const server = observability.SpanKind.server;
    const client = observability.SpanKind.client;

    try std.testing.expect(internal != server);
    try std.testing.expect(server != client);
}

test {
    std.testing.refAllDecls(@This());
}
