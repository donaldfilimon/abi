---
title: "Observability"
description: "Metrics, tracing, and profiling"
section: "Operations"
order: 3
---

# Observability

The observability module provides unified metrics collection, distributed
tracing, alerting, and integration with Prometheus, OpenTelemetry, and StatsD.
It is the system-level counterpart to [Analytics](analytics.html), focusing on
infrastructure health rather than product usage.

- **Build flag:** `-Denable-profiling=true` (default: enabled)
- **Namespace:** `abi.observability`
- **Source:** `src/features/observability/`

**Important:** The build flag is `-Denable-profiling`, not `-Denable-observability`.

## Overview

The module is organized into four areas:

- **Metrics** -- Counters, gauges, float gauges, and histograms with a `MetricsCollector` registry. Includes default metrics for HTTP requests, errors, and latency, plus specialized `CircuitBreakerMetrics` and `ErrorMetrics`.
- **Tracing** -- Distributed tracing with `Tracer`, `Span`, `TraceId`, `SpanId`, context propagation (`W3C TraceContext`), and configurable sampling.
- **Monitoring** -- Consolidated alerting rules, Prometheus exporter, and StatsD client.
- **OpenTelemetry** -- Full OTLP exporter with `OtelTracer`, `OtelSpan`, `OtelMetric`, and resource attributes.

The `ObservabilityBundle` struct ties all of these together into a single
initialization point.

## Quick Start

```zig
const abi = @import("abi");
const obs = abi.observability;

// Create a metrics collector
var collector = obs.MetricsCollector.init(allocator);
defer collector.deinit();

// Register metrics
const requests = try collector.registerCounter("http_requests_total");
const active = try collector.registerGauge("active_connections");
const latency = try collector.registerHistogram(
    "request_latency_ms",
    &[_]u64{ 10, 50, 100, 500, 1000 },
);

// Record data
requests.inc(1);
active.set(42);
latency.record(75);
```

## API Reference

### Metrics Types

| Type | Description |
|------|-------------|
| `MetricsCollector` | Registry for counters, gauges, and histograms |
| `Counter` | Monotonically increasing unsigned counter (atomic) |
| `Gauge` | Signed integer gauge with `inc()`, `dec()`, `add()`, `set()` (atomic) |
| `FloatGauge` | Floating-point gauge with `add()` and `set()` (atomic) |
| `Histogram` | Bucketed distribution with configurable bounds |
| `DefaultMetrics` | Standard HTTP metrics (`requests`, `errors`, `latency_ms`) |
| `DefaultCollector` | Convenience wrapper: collector + default metrics in one |
| `CircuitBreakerMetrics` | Metrics for circuit breaker state tracking |
| `ErrorMetrics` | Error counters with critical/total/pattern detection |

### Tracing Types

| Type | Description |
|------|-------------|
| `Tracer` | Creates and manages spans |
| `Span` | Single unit of work in a distributed trace |
| `TraceId` | 128-bit trace identifier |
| `SpanId` | 64-bit span identifier |
| `SpanKind` | `internal`, `server`, `client`, `producer`, `consumer` |
| `SpanStatus` | `unset`, `ok`, `error` |
| `SpanAttribute` | Key-value attribute on a span |
| `SpanEvent` | Timestamped event within a span |
| `SpanLink` | Link to another span for causal relationships |
| `TraceContext` | W3C TraceContext propagation |
| `Propagator` | Context propagation across service boundaries |
| `TraceSampler` | Configurable trace sampling strategies |
| `SpanProcessor` | Processes completed spans |
| `SpanExporter` | Exports spans to a backend |

### Alerting Types

| Type | Description |
|------|-------------|
| `AlertManager` | Manages alert rules and fires alerts |
| `AlertRule` | Condition that triggers an alert |
| `AlertRuleBuilder` | Fluent builder for alert rules |
| `Alert` | Fired alert with severity and state |
| `AlertState` | `inactive`, `pending`, `firing`, `resolved` |
| `AlertSeverity` | `info`, `warning`, `critical` |

### OpenTelemetry Types

| Type | Description |
|------|-------------|
| `OtelExporter` | OTLP exporter for traces and metrics |
| `OtelConfig` | Configuration for the OTLP endpoint |
| `OtelTracer` | OpenTelemetry-compatible tracer |
| `OtelSpan` | OpenTelemetry span |
| `OtelMetric` | OpenTelemetry metric |
| `OtelContext` | OpenTelemetry context |

### Key Functions

| Function | Description |
|----------|-------------|
| `createCollector(allocator)` | Create a new `MetricsCollector` |
| `registerDefaultMetrics(collector)` | Register standard HTTP metrics |
| `recordRequest(defaults, latency_ms)` | Record an HTTP request |
| `recordError(defaults, status_code)` | Record an HTTP error |
| `generateMetricsOutput(...)` | Export metrics in Prometheus text format |

## ObservabilityBundle

The `ObservabilityBundle` provides one-shot initialization of all observability
components:

```zig
var bundle = try obs.ObservabilityBundle.init(allocator, .{
    .enable_circuit_breaker_metrics = true,
    .enable_error_metrics = true,
    .prometheus = .{
        .port = 9090,
        .path = "/metrics",
    },
    .otel = .{
        .endpoint = "http://localhost:4317",
        .service_name = "my-service",
    },
});
defer bundle.deinit();

try bundle.start();
defer bundle.stop();

// Use the bundle's tracer
if (try bundle.startSpan("handle_request")) |span| {
    defer span.end();
    // ... do work ...
}
```

## Configuration

```zig
// Framework-level observability config
var framework = try abi.Framework.builder(allocator)
    .withObservability(.{
        .metrics_enabled = true,
        .tracing_enabled = true,
    })
    .build();
defer framework.deinit();
```

## CLI Commands

Observability data can be inspected via the CLI:

```bash
zig build run -- system-info       # Shows feature status including profiling
```

## Examples

See `examples/observability.zig` for a working example demonstrating counters,
gauges, and histograms.

```bash
zig build run-observability
```

## Disabling at Build Time

```bash
zig build -Denable-profiling=false
```

When disabled, all metric types become no-op stubs. Counters always return 0,
gauges ignore `set()`/`inc()`/`dec()`, histograms discard samples, and the
`ObservabilityBundle` returns stubs. Code using the observability API compiles
cleanly regardless of the flag -- zero binary overhead when disabled.

## Related

- [Analytics](analytics.html) -- User-facing event tracking (complementary)
- [Benchmarks](benchmarks.html) -- Performance measurement with profiler integration
- [Deployment](deployment.html) -- Production monitoring with Grafana/Prometheus/Jaeger
