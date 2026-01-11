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

## Profiling

The framework measures the execution time of work-stealing tasks and can export summaries.

```zig
const summary = engine.getMetricsSummary();
std.debug.print("Avg Task Time: {d} us\n", .{summary.avg_execution_ns / 1000});
```
