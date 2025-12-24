# Observability

ABI exposes monitoring utilities under `abi.monitoring`:

- `logging.zig`: structured logging helpers
- `metrics.zig`: metrics primitives and Prometheus helpers
- `tracing.zig`: trace spans and context propagation
- `performance.zig` / `performance_profiler.zig`: profiling helpers
- `memory_tracker.zig`: allocation tracking
- `health.zig`: health checks

## Usage
Import the feature module:
```zig
const monitoring = abi.monitoring;
```

Enable observability by wiring the helpers into your application lifecycle.
