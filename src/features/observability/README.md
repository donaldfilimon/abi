# Observability Module
> **Last reviewed:** 2026-01-31

Unified observability layer providing metrics collection, distributed tracing, and performance profiling.

## Overview

The observability module consolidates three critical aspects of system monitoring:
- **Metrics**: Track system behavior (counters, gauges, histograms)
- **Tracing**: Distributed request tracing with OpenTelemetry support
- **Monitoring**: Alerting rules, Prometheus/StatsD export, circuit breaker tracking

## Key Components

### Metrics Primitives

**Counter**: Thread-safe monotonic counter for counting events
```zig
var counter = Counter{ .name = "requests_total" };
counter.inc(1);
const value = counter.get();
```

**Gauge**: Integer gauge that can increase or decrease
```zig
var gauge = Gauge{ .name = "active_connections" };
gauge.inc();
gauge.dec();
gauge.set(100);
```

**FloatGauge**: Floating-point gauge with mutex protection
```zig
var gauge = FloatGauge{ .name = "cpu_usage_percent" };
gauge.set(75.5);
gauge.add(0.5);
```

**Histogram**: Bucketed metric for recording distributions
```zig
var histogram = try Histogram.init(allocator, "latency_ms", &.{ 10, 50, 100, 500 });
histogram.record(45);
defer histogram.deinit(allocator);
```

### MetricsCollector

Central registry for metrics with lifecycle management.

```zig
var collector = MetricsCollector.init(allocator);
defer collector.deinit();

const requests = try collector.registerCounter("http_requests_total");
const latency = try collector.registerHistogram("request_latency_ms", &.{ 1, 5, 10, 50, 100 });

requests.inc(1);
latency.record(25);
```

### Tracing

Distributed tracing with `Tracer`, `Span`, and OpenTelemetry integration.

```zig
const observability = @import("abi").observability;

var tracer = try observability.OtelTracer.init(allocator, "my-service");
defer tracer.deinit();

var span = try tracer.startSpan("database_query", null, null);
defer span.deinit();

span.setAttribute("db.statement", "SELECT * FROM users");
```

### Monitoring & Alerting

**PrometheusExporter**: Export metrics in Prometheus format

```zig
const exporter = try observability.PrometheusExporter.init(
    allocator,
    .{ .port = 9090 },
    &collector,
);
try exporter.start();
defer exporter.stop();
```

**AlertManager**: Define and trigger alerts based on metric conditions

```zig
var alert_manager = try observability.AlertManager.init(allocator, .{});
defer alert_manager.deinit();

const rule = try observability.AlertRuleBuilder
    .init("high_error_rate")
    .withCondition(.{ .threshold = 100, .duration_secs = 60 })
    .build(allocator);

try alert_manager.addRule(rule);
```

**StatsDClient**: Export metrics to StatsD

```zig
const statsd = try observability.StatsDClient.init(allocator, .{
    .host = "127.0.0.1",
    .port = 8125,
});
try statsd.gauge("memory_usage", 1024);
```

### ObservabilityBundle

All-in-one observability setup combining metrics, tracing, and exporters.

```zig
const bundle = try observability.ObservabilityBundle.init(allocator, .{
    .enable_circuit_breaker_metrics = true,
    .enable_error_metrics = true,
    .prometheus = .{ .port = 9090 },
    .otel = .{ .service_name = "my-service" },
});
defer bundle.deinit();

try bundle.start();
defer bundle.stop();

bundle.defaults.requests.inc(1);
bundle.defaults.latency_ms.record(50);
```

## Standard Metrics

The module provides pre-configured default metrics:

| Metric | Type | Purpose |
|--------|------|---------|
| `requests_total` | Counter | Total requests received |
| `errors_total` | Counter | Total errors encountered |
| `latency_ms` | Histogram | Request latency distribution (1, 5, 10, 25, 50, 100, 250, 500, 1000ms buckets) |
| `circuit_breaker_requests_total` | Counter | Total circuit breaker requests |
| `circuit_breaker_requests_rejected` | Counter | Rejected requests (circuit open) |
| `circuit_breaker_state_transitions` | Counter | State changes (closed → open → half-open) |
| `errors_critical` | Counter | Critical errors only |
| `error_patterns_detected` | Counter | Detected error patterns |

## Usage Examples

### Basic Metrics Collection

```zig
const observability = @import("abi").observability;

var collector = observability.MetricsCollector.init(allocator);
defer collector.deinit();

const requests = try collector.registerCounter("api_requests");
const errors = try collector.registerCounter("api_errors");

// Record metrics
requests.inc(1);
if (requestFailed()) {
    errors.inc(1);
}
```

### Request Tracking with Histogram

```zig
const time = @import("../../services/shared/time.zig");
var timer = time.Timer.start() catch return error.TimerFailed;
defer {
    const elapsed_ns = timer.read();
    const elapsed_ms = elapsed_ns / 1_000_000;
    latency_histogram.record(elapsed_ms);
}

// ... perform work
```

### Integration with OpenTelemetry

```zig
const bundle = try observability.ObservabilityBundle.init(allocator, .{
    .otel = .{ .service_name = "payment-service" },
});
defer bundle.deinit();

if (try bundle.startSpan("process_payment")) |span| {
    defer span.deinit();
    span.setAttribute("user_id", "12345");
    span.setAttribute("amount", "99.99");
    // OpenTelemetry span automatically traced
}
```

### Prometheus Export

```zig
const exporter = try observability.PrometheusExporter.init(
    allocator,
    .{ .port = 9090 },
    &collector,
);
try exporter.start();
defer exporter.stop();

// Metrics exposed at http://localhost:9090/metrics
```

## Configuration

The observability module is feature-gated by the `-Denable-profiling` build flag. When disabled, operations return `error.ObservabilityDisabled`.

See `src/core/config/observability.zig` for configuration options including:
- Metrics collection enable/disable
- Export backend selection (Prometheus, OpenTelemetry, StatsD)
- Alert thresholds and notification channels
- Tracing sample rates

## Related Documentation

- **Monitoring**: See `docs/content/observability.html` for setup and guidance
- **Benchmarking**: See `benchmarks/README.md` for performance profiling
- **GPU Metrics**: GPU backends expose accelerator-specific metrics via this module
- **Network Metrics**: Distributed compute exposes request/RPC metrics

## Error Handling

The module defines specific error types:

```zig
pub const Error = error{
    ObservabilityDisabled,    // Feature disabled at build time
    MetricsError,             // Metric registration failed
    TracingError,             // Span creation failed
    ExportFailed,             // Export backend connection error
};
```

Check error types when initializing exporters or in production code:

```zig
const exporter = observability.PrometheusExporter.init(
    allocator,
    config,
    &collector,
) catch |err| {
    std.debug.print("Failed to initialize Prometheus: {}\n", .{err});
    return err;
};
```

## Thread Safety

All metric primitives are thread-safe:
- **Counter/Gauge**: Use `std.atomic.Value` for lock-free operations
- **FloatGauge**: Uses `sync.Mutex` for safe floating-point access
- **MetricsCollector**: Safe concurrent registration and recording

## Performance

- Metric recording is O(1) for all primitive types
- Histogram bucket lookup is O(n) where n = number of bounds
- No allocations during metric recording (only during registration)
- Thread-safe atomic operations with minimal contention
