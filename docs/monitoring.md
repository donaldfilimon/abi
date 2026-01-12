# Observability

ABI includes built-in tools for monitoring application health and performance.

## Logging

Use `abi.log` (which wraps `std.log`) for structured logging.

```zig
abi.log.info("Server started on port {d}", .{port});
```

## Metrics

Enable metrics with `-Denable-profiling=true`.

- **Counters**: Track distinct events (e.g., `requests_total`).
- **Gauges**: Track instantaneous values (e.g., `memory_usage`).
- **Histograms**: Track distributions (e.g., `request_latency_ms`).

```zig
var metrics = abi.monitoring.MetricsCollector.init(allocator);
metrics.recordTaskExecution(task_id, duration_ns);
```

## Alerting

Alerting rules evaluate metrics and trigger notifications for pending/firing states.
Use the `AlertManager` to configure rules with conditions, thresholds, and severity.

```zig
var manager = try abi.monitoring.AlertManager.init(allocator, .{});
defer manager.deinit();

try manager.addRule(.{
    .name = "error_spike",
    .metric = "errors_total",
    .condition = .rate_of_change,
    .threshold = 25,
    .severity = .critical,
    .for_duration_ms = 0,
});

var values = abi.monitoring.MetricValues.init();
defer values.deinit(allocator);

try values.set(allocator, "errors_total", 100);
try manager.evaluate(&values);
```

`rate_of_change` compares the latest sample against the previous sample; the first
sample establishes a baseline and does not trigger. When a metric disappears,
non-`.absent` rules are treated as not met so pending/firing alerts can resolve.

## Profiling

The framework measures the execution time of work-stealing tasks and can export summaries.

```zig
const summary = engine.getMetricsSummary();
std.debug.print("Avg Task Time: {d} us\n", .{summary.avg_execution_ns / 1000});
```

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
