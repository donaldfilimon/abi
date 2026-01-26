# observability API Reference

> Metrics, tracing, and monitoring

**Source:** [`src/observability/mod.zig`](../../src/observability/mod.zig)

---

Observability Module

Unified observability with metrics, tracing, and profiling.

## Features
- Metrics collection and export (Prometheus, OpenTelemetry, StatsD)
- Distributed tracing
- Performance profiling
- Circuit breakers and error aggregation
- Alerting rules and notifications

---

## API

### `pub const Gauge`

<sup>**type**</sup>

Gauge metric - a value that can increase or decrease.
Uses atomic i64 for thread-safe operations.

### `pub const FloatGauge`

<sup>**type**</sup>

Float gauge metric - for f64 values requiring mutex protection.

---

*Generated automatically by `zig build gendocs`*
