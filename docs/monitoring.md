# Monitoring & Observability

> **Status**: Requires `-Denable-profiling=true` at build time.

The **Monitoring** module provides comprehensive observability tools for tracking application health, performance, and behavior.

---

## Overview

ABI's monitoring stack includes:

- **Logging** - Structured log output with levels and context
- **Metrics** - Counters, gauges, and histograms for quantitative data
- **Alerting** - Configurable rules with threshold-based notifications
- **Tracing** - Request flow tracking across operations
- **Profiling** - Performance measurement and bottleneck detection

---

## Logging

Use `abi.log` for structured logging with automatic context.

### Basic Usage

```zig
const abi = @import("abi");

// Log levels
abi.log.debug("Debug message: {d}", .{value});
abi.log.info("Server started on port {d}", .{port});
abi.log.warn("Connection pool low: {d} remaining", .{count});
abi.log.err("Failed to connect: {t}", .{error_value});
```

### Scoped Logging

```zig
const log = std.log.scoped(.my_module);

pub fn processRequest(req: Request) !void {
    log.info("Processing request {d}", .{req.id});
    defer log.debug("Request {d} complete", .{req.id});
    // ...
}
```

### Log Levels

| Level | Use Case |
|-------|----------|
| `debug` | Detailed debugging information |
| `info` | Normal operational messages |
| `warn` | Potential issues that don't stop execution |
| `err` | Errors that affect functionality |

---

## Metrics

Enable metrics collection to track quantitative data about your application.

### Metrics Collector

```zig
const abi = @import("abi");

var metrics = try abi.monitoring.MetricsCollector.init(
    allocator,
    abi.monitoring.DEFAULT_METRICS_CONFIG,
    4,  // worker count
);
defer metrics.deinit();

// Record task execution times
metrics.recordTaskExecution(task_id, duration_ns);

// Get summary statistics
const summary = metrics.getSummary();
std.debug.print("Total tasks: {d}\n", .{summary.total_tasks});
std.debug.print("Avg execution: {d} us\n", .{summary.avg_execution_ns / 1000});
std.debug.print("P99 execution: {d} us\n", .{summary.p99_execution_ns / 1000});
```

### Metric Types

#### Counters

Track cumulative values that only increase.

```zig
// Record distinct events
metrics.incrementCounter("requests_total");
metrics.incrementCounterBy("bytes_processed", byte_count);
```

#### Gauges

Track instantaneous values that can go up or down.

```zig
// Set current value
metrics.setGauge("active_connections", connection_count);
metrics.setGauge("memory_usage_bytes", @as(f64, @floatFromInt(used_memory)));

// Read current value
const current = metrics.getGauge("active_connections");
```

#### Histograms

Track value distributions for latency, sizes, etc.

```zig
// Record observations
metrics.recordHistogram("request_latency_ms", latency_ms);
metrics.recordHistogram("response_size_bytes", size);

// Get percentiles
const p50 = metrics.getHistogramPercentile("request_latency_ms", 0.50);
const p95 = metrics.getHistogramPercentile("request_latency_ms", 0.95);
const p99 = metrics.getHistogramPercentile("request_latency_ms", 0.99);
```

### Compute Engine Metrics

The compute engine provides built-in metrics:

```zig
var engine = try abi.compute.createDefaultEngine(allocator);
defer engine.deinit();

// Run workloads...

// Get engine metrics summary
const summary = engine.getMetricsSummary();
std.debug.print("Tasks completed: {d}\n", .{summary.total_tasks});
std.debug.print("Avg task time: {d} us\n", .{summary.avg_execution_ns / 1000});
std.debug.print("Min task time: {d} us\n", .{summary.min_execution_ns / 1000});
std.debug.print("Max task time: {d} us\n", .{summary.max_execution_ns / 1000});
std.debug.print("Work steals: {d}\n", .{summary.work_steals});
```

---

## Alerting

Configure rules to trigger notifications when metrics cross thresholds.

### Alert Manager

```zig
const alerting = abi.monitoring.alerting;

var manager = try alerting.AlertManager.init(allocator, .{
    .evaluation_interval_ms = 15_000,    // Check every 15 seconds
    .default_for_duration_ms = 60_000,   // Alert after 1 minute
});
defer manager.deinit();
```

### Adding Rules

```zig
// High error rate alert
try manager.addRule(.{
    .name = "high_error_rate",
    .metric = "errors_total",
    .condition = .greater_than,
    .threshold = 100,
    .severity = .critical,
    .for_duration_ms = 30_000,  // Must exceed threshold for 30s
});

// Low memory alert
try manager.addRule(.{
    .name = "low_memory",
    .metric = "memory_available_bytes",
    .condition = .less_than,
    .threshold = 100 * 1024 * 1024,  // 100MB
    .severity = .warning,
});

// CPU saturation alert
try manager.addRule(.{
    .name = "cpu_saturation",
    .metric = "cpu_usage_percent",
    .condition = .greater_than,
    .threshold = 90,
    .severity = .warning,
    .for_duration_ms = 120_000,  // 2 minutes
});
```

### Alert Conditions

| Condition | Description |
|-----------|-------------|
| `greater_than` | Metric > threshold |
| `less_than` | Metric < threshold |
| `equal_to` | Metric == threshold |
| `not_equal_to` | Metric != threshold |
| `greater_or_equal` | Metric >= threshold |
| `less_or_equal` | Metric <= threshold |

### Alert Severities

| Severity | Use Case |
|----------|----------|
| `info` | Informational, no action required |
| `warning` | Potential issue, investigate soon |
| `critical` | Immediate action required |

### Notification Handlers

```zig
// Register a handler function
try manager.addHandler(.{
    .callback = struct {
        fn handle(alert: alerting.Alert) void {
            std.debug.print("[ALERT] {t}: {s} ({t})\n", .{
                alert.severity,  // Zig 0.16: Use {t} directly instead of @tagName
                alert.rule_name,
                alert.state,
            });
            // Send to external system (Slack, PagerDuty, etc.)
        }
    }.handle,
    .min_severity = .warning,  // Only trigger for warning+
});

// Evaluate rules periodically (Zig 0.16: use time utilities)
const time_utils = @import("shared/utils/time.zig");
while (running) {
    try manager.evaluate(metrics);
    time_utils.sleepSeconds(15);
}
```

### Alert States

Alerts transition through these states:

```
inactive -> pending -> firing -> resolved
    ^                              |
    |______________________________|
```

| State | Description |
|-------|-------------|
| `inactive` | Condition not met |
| `pending` | Condition met, waiting for `for_duration` |
| `firing` | Condition met for required duration |
| `resolved` | Previously firing, now inactive |

---

## Tracing

Track request flow across operations for debugging and performance analysis.

### Basic Tracing

```zig
const tracing = abi.monitoring.tracing;

var tracer = try tracing.Tracer.init(allocator, .{
    .sample_rate = 1.0,  // 100% sampling
    .max_spans = 10000,
});
defer tracer.deinit();

// Start a trace
var span = tracer.startSpan("handle_request");
defer span.end();

// Add context
span.setTag("user_id", user.id);
span.setTag("endpoint", "/api/search");

// Child spans
{
    var db_span = tracer.startSpanWithParent("database_query", span);
    defer db_span.end();

    const result = try db.query(sql);
    db_span.setTag("row_count", result.rows.len);
}
```

### Span Attributes

```zig
span.setTag("http.method", "POST");
span.setTag("http.status_code", 200);
span.setTag("db.statement", "SELECT * FROM users");
span.setBaggage("request_id", request_id);
```

### Trace Export

```zig
// Export traces for analysis
const traces = tracer.getCompletedTraces();
for (traces) |trace| {
    std.debug.print("Trace {s}: {d} spans, {d}ms\n", .{
        trace.id,
        trace.spans.len,
        trace.duration_ms,
    });
}
```

---

## Profiling

Measure performance to identify bottlenecks.

### Timer-Based Profiling

```zig
const time_utils = abi.shared.time;

// Measure operation duration
var timer = try std.time.Timer.start();
performExpensiveOperation();
const elapsed_ns = timer.read();

std.debug.print("Operation took {d} us\n", .{elapsed_ns / 1000});
```

### Stopwatch Utility

```zig
const time_utils = abi.shared.time;

var watch = try time_utils.Stopwatch.start();

// ... do work ...

const elapsed_ms = watch.elapsedMs();
std.debug.print("Elapsed: {d} ms\n", .{elapsed_ms});

// Lap times
watch.lap();
// ... more work ...
const lap_ms = watch.lapElapsedMs();
```

### Engine Profiling

```zig
var engine = try abi.compute.createDefaultEngine(allocator);
defer engine.deinit();

// Enable detailed profiling
engine.enableProfiling(.{
    .track_steals = true,
    .track_wait_times = true,
    .histogram_buckets = 100,
});

// Run workloads...

// Export profiling data
const profile = engine.getProfilingData();
std.debug.print("Worker utilization: {d:.2}%\n", .{profile.avg_utilization * 100});
std.debug.print("Steal success rate: {d:.2}%\n", .{profile.steal_success_rate * 100});
```

### GPU Profiling

```zig
var gpu = try abi.Gpu.init(allocator, .{
    .enable_profiling = true,
});
defer gpu.deinit();

// Run GPU operations...

// Get GPU metrics
if (gpu.getMetricsSummary()) |summary| {
    std.debug.print("Total kernels: {d}\n", .{summary.total_kernel_invocations});
    std.debug.print("Avg kernel time: {d:.3} us\n", .{
        @as(f64, @floatFromInt(summary.avg_kernel_time_ns)) / 1000.0,
    });
    std.debug.print("Total transfers: {d}\n", .{summary.total_transfers});
    std.debug.print("Bytes transferred: {d}\n", .{summary.total_bytes_transferred});
}
```

---

## Configuration

### Metrics Configuration

```zig
const config = abi.monitoring.MetricsConfig{
    .enable_histograms = true,
    .histogram_buckets = 50,
    .max_metrics = 1000,
    .flush_interval_ms = 10_000,
    .enable_percentiles = true,
};

var metrics = try abi.monitoring.MetricsCollector.init(allocator, config, worker_count);
```

### Alerting Configuration

```zig
const config = abi.monitoring.alerting.AlertManagerConfig{
    .evaluation_interval_ms = 15_000,
    .default_for_duration_ms = 60_000,
    .max_rules = 100,
    .max_handlers = 10,
};
```

---

## CLI Commands

```bash
# Show system and framework status
zig build run -- system-info

# Run with profiling enabled
zig build -Denable-profiling=true run -- <command>

# Run benchmarks with metrics
zig build benchmarks
```

---

## Best Practices

1. **Use appropriate log levels** - Don't log debug in production
2. **Sample traces in production** - Use 1-10% sampling for high-traffic systems
3. **Set meaningful alert thresholds** - Base on historical data
4. **Use `for_duration`** - Avoid alerting on transient spikes
5. **Export metrics** - Integrate with external monitoring systems
6. **Profile before optimizing** - Measure, don't guess

---

## See Also

- [Compute Engine](compute.md) - Engine metrics and profiling
- [GPU Acceleration](gpu.md) - GPU metrics and profiling
- [Performance Baseline](PERFORMANCE_BASELINE.md) - Benchmark targets
- [Framework](framework.md) - Framework configuration
- [Troubleshooting](troubleshooting.md) - Debugging issues
