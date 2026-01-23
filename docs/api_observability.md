# Observability API Reference
> **Codebase Status:** Synced with repository as of 2026-01-22.

**Source:** `src/observability/mod.zig`

The observability module provides unified metrics collection, distributed tracing, and alerting capabilities with support for multiple export formats (Prometheus, OpenTelemetry, StatsD).

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize metrics collector
    var metrics = abi.observability.MetricsCollector.init(allocator);
    defer metrics.deinit();

    // Register and use counters
    const request_counter = try metrics.registerCounter("http_requests_total");
    request_counter.inc(1);

    // Register and use histograms
    const latency_bounds = [_]u64{ 10, 50, 100, 250, 500, 1000 };
    const latency_hist = try metrics.registerHistogram("http_request_latency_ms", &latency_bounds);
    latency_hist.record(42);

    std.debug.print("Requests: {d}\n", .{request_counter.get()});
}
```

---

## Metrics Primitives

### `Counter`

Monotonically increasing counter (thread-safe).

```zig
pub const Counter = struct {
    name: []const u8,
    value: std.atomic.Value(u64),

    pub fn inc(self: *Counter, delta: u64) void;
    pub fn get(self: *const Counter) u64;
};
```

**Usage:**
```zig
const counter = try metrics.registerCounter("events_total");
counter.inc(1);
counter.inc(5);
std.debug.print("Total: {d}\n", .{counter.get()}); // 6
```

### `Histogram`

Distribution of values across configurable buckets.

```zig
pub const Histogram = struct {
    name: []const u8,
    buckets: []u64,
    bounds: []u64,

    pub fn init(allocator: Allocator, name: []const u8, bounds: []u64) !Histogram;
    pub fn deinit(self: *Histogram, allocator: Allocator) void;
    pub fn record(self: *Histogram, value: u64) void;
};
```

**Usage:**
```zig
const bounds = [_]u64{ 10, 50, 100, 500, 1000 };
const hist = try metrics.registerHistogram("latency_ms", &bounds);
hist.record(42);  // Falls in 10-50 bucket
hist.record(150); // Falls in 100-500 bucket
```

### `MetricsCollector`

Central registry for all metrics.

```zig
pub const MetricsCollector = struct {
    pub fn init(allocator: Allocator) MetricsCollector;
    pub fn deinit(self: *MetricsCollector) void;
    pub fn registerCounter(self: *MetricsCollector, name: []const u8) !*Counter;
    pub fn registerHistogram(self: *MetricsCollector, name: []const u8, bounds: []const u64) !*Histogram;
};
```

---

## Tracing

### `Tracer`

Distributed tracing context manager.

```zig
pub const Tracer = struct {
    pub fn init(allocator: Allocator, service_name: []const u8) Tracer;
    pub fn deinit(self: *Tracer) void;
    pub fn startSpan(self: *Tracer, name: []const u8, parent: ?SpanId) Span;
    pub fn currentTrace(self: *const Tracer) ?TraceId;
};
```

### `Span`

Represents a unit of work in a trace.

```zig
pub const Span = struct {
    trace_id: TraceId,
    span_id: SpanId,
    parent_id: ?SpanId,
    name: []const u8,
    start_time: i64,
    end_time: ?i64,
    status: SpanStatus,
    kind: SpanKind,

    pub fn end(self: *Span) void;
    pub fn setStatus(self: *Span, status: SpanStatus) void;
    pub fn setAttribute(self: *Span, key: []const u8, value: anytype) void;
    pub fn addEvent(self: *Span, name: []const u8) void;
};
```

### Span Types

```zig
pub const SpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const SpanStatus = enum {
    unset,
    ok,
    error,
};

pub const TraceId = [16]u8;
pub const SpanId = [8]u8;
```

**Usage:**
```zig
var tracer = Tracer.init(allocator, "my-service");
defer tracer.deinit();

var span = tracer.startSpan("process_request", null);
defer span.end();

span.setAttribute("user_id", "12345");
span.setStatus(.ok);
```

---

## Alerting

### `AlertManager`

Manages alert rules and notifications.

```zig
pub const AlertManager = struct {
    pub fn init(allocator: Allocator, config: AlertManagerConfig) AlertManager;
    pub fn deinit(self: *AlertManager) void;
    pub fn addRule(self: *AlertManager, rule: AlertRule) !void;
    pub fn removeRule(self: *AlertManager, name: []const u8) bool;
    pub fn evaluate(self: *AlertManager, metrics: *MetricsCollector) void;
    pub fn getActiveAlerts(self: *AlertManager) []const Alert;
};
```

### `AlertManagerConfig`

```zig
pub const AlertManagerConfig = struct {
    /// Evaluation interval in seconds (default: 60)
    evaluation_interval_sec: u32 = 60,
    /// Default notification channels
    notification_channels: []const NotificationChannel = &.{},
    /// Enable alert grouping (default: true)
    group_alerts: bool = true,
    /// Group wait time in seconds (default: 30)
    group_wait_sec: u32 = 30,
};
```

### `AlertRule`

```zig
pub const AlertRule = struct {
    name: []const u8,
    condition: AlertCondition,
    severity: AlertSeverity,
    duration_sec: u32,       // How long condition must be true
    labels: []const Label,   // Additional labels
    annotations: []const Annotation,
};
```

### `AlertRuleBuilder`

Fluent API for building alert rules.

```zig
pub const AlertRuleBuilder = struct {
    pub fn init(name: []const u8) AlertRuleBuilder;
    pub fn withCondition(self: *AlertRuleBuilder, condition: AlertCondition) *AlertRuleBuilder;
    pub fn withSeverity(self: *AlertRuleBuilder, severity: AlertSeverity) *AlertRuleBuilder;
    pub fn withDuration(self: *AlertRuleBuilder, duration_sec: u32) *AlertRuleBuilder;
    pub fn build(self: *AlertRuleBuilder) AlertRule;
};
```

### Alert Types

```zig
pub const AlertSeverity = enum {
    info,
    warning,
    critical,
    page,
};

pub const AlertState = enum {
    inactive,
    pending,
    firing,
    resolved,
};

pub const AlertCondition = union(enum) {
    counter_above: struct { metric: []const u8, threshold: u64 },
    counter_below: struct { metric: []const u8, threshold: u64 },
    rate_above: struct { metric: []const u8, threshold: f64, window_sec: u32 },
    histogram_p99_above: struct { metric: []const u8, threshold: u64 },
};
```

**Usage:**
```zig
var alert_mgr = AlertManager.init(allocator, .{});
defer alert_mgr.deinit();

const rule = AlertRuleBuilder.init("high_error_rate")
    .withCondition(.{ .rate_above = .{
        .metric = "errors_total",
        .threshold = 0.05,
        .window_sec = 300,
    }})
    .withSeverity(.critical)
    .withDuration(60)
    .build();

try alert_mgr.addRule(rule);
```

---

## Exporters

### Prometheus Exporter

```zig
pub const PrometheusExporter = struct {
    pub fn init(allocator: Allocator, port: u16) !PrometheusExporter;
    pub fn deinit(self: *PrometheusExporter) void;
    pub fn start(self: *PrometheusExporter) !void;
    pub fn stop(self: *PrometheusExporter) void;
    pub fn registerCollector(self: *PrometheusExporter, collector: *MetricsCollector) void;
};
```

### StatsD Exporter

```zig
pub const StatsdExporter = struct {
    pub fn init(allocator: Allocator, host: []const u8, port: u16) !StatsdExporter;
    pub fn deinit(self: *StatsdExporter) void;
    pub fn flush(self: *StatsdExporter, collector: *MetricsCollector) !void;
};
```

### OpenTelemetry Exporter

```zig
pub const OtelExporter = struct {
    pub fn init(allocator: Allocator, endpoint: []const u8) !OtelExporter;
    pub fn deinit(self: *OtelExporter) void;
    pub fn exportMetrics(self: *OtelExporter, collector: *MetricsCollector) !void;
    pub fn exportTraces(self: *OtelExporter, tracer: *Tracer) !void;
};
```

---

## Framework Integration

### `ObservabilityBundle`

Pre-configured observability setup for Framework integration.

```zig
pub const ObservabilityBundle = struct {
    pub fn init(allocator: Allocator, config: ObservabilityConfig) !ObservabilityBundle;
    pub fn deinit(self: *ObservabilityBundle) void;
    pub fn metrics(self: *ObservabilityBundle) *MetricsCollector;
    pub fn tracer(self: *ObservabilityBundle) *Tracer;
    pub fn alertManager(self: *ObservabilityBundle) *AlertManager;
};
```

### Via Framework Context

```zig
const config = abi.Config.init().withProfiling(true);
var fw = try abi.Framework.init(allocator, config);
defer fw.deinit();

if (fw.observability()) |obs| {
    const counter = try obs.metrics().registerCounter("my_metric");
    counter.inc(1);
}
```

---

## Usage Patterns

### Basic Metrics Collection

```zig
var metrics = MetricsCollector.init(allocator);
defer metrics.deinit();

const requests = try metrics.registerCounter("http_requests_total");
const errors = try metrics.registerCounter("http_errors_total");
const latency = try metrics.registerHistogram("http_latency_ms", &.{ 10, 50, 100, 500 });

// In request handler
requests.inc(1);
latency.record(elapsed_ms);
if (response_code >= 500) errors.inc(1);
```

### Request Tracing

```zig
var tracer = Tracer.init(allocator, "api-service");
defer tracer.deinit();

fn handleRequest(tracer: *Tracer, request: Request) !Response {
    var span = tracer.startSpan("handle_request", null);
    defer span.end();

    span.setAttribute("method", request.method);
    span.setAttribute("path", request.path);

    // Database call with child span
    var db_span = tracer.startSpan("db_query", span.span_id);
    defer db_span.end();
    const data = try db.query(request.params);

    span.setStatus(.ok);
    return Response.ok(data);
}
```

### Alert Configuration

```zig
var alert_mgr = AlertManager.init(allocator, .{
    .evaluation_interval_sec = 30,
});
defer alert_mgr.deinit();

// High latency alert
try alert_mgr.addRule(AlertRuleBuilder.init("high_latency")
    .withCondition(.{ .histogram_p99_above = .{
        .metric = "http_latency_ms",
        .threshold = 500,
    }})
    .withSeverity(.warning)
    .withDuration(300)
    .build());

// Error rate alert
try alert_mgr.addRule(AlertRuleBuilder.init("high_errors")
    .withCondition(.{ .rate_above = .{
        .metric = "http_errors_total",
        .threshold = 0.01,
        .window_sec = 60,
    }})
    .withSeverity(.critical)
    .withDuration(60)
    .build());
```

---

## Related Documentation

- [Metrics Guide](metrics.md)
- [Tracing Guide](tracing.md)
- [Alerting Configuration](alerting.md)
- [API Reference](API_REFERENCE.md)
