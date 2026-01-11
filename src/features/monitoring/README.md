//! # Monitoring
//!
//! Observability hooks for metrics, tracing, and health checks.
//!
//! ## Features
//!
//! - **Metrics Export**: Counter, gauge, histogram primitives
//! - **Tracing**: Distributed trace context propagation
//! - **Health Checks**: Endpoint health monitoring
//! - **OpenTelemetry**: OTLP export support (planned)
//!
//! ## Usage
//!
//! ```zig
//! const monitoring = @import("abi").monitoring;
//!
//! // Record a counter
//! try monitoring.counter("requests_total").inc();
//!
//! // Record a gauge
//! try monitoring.gauge("active_connections").set(42);
//!
//! // Record a histogram
//! try monitoring.histogram("request_duration_ms").observe(123.4);
//! ```
//!
//! ## Sub-modules
//!
//! - `mod.zig` - Public API aggregation
//! - `metrics.zig` - Metric collection and export
//! - `health.zig` - Health check endpoints
//!
//! ## Adding Backends
//!
//! Create a sub-module implementing the metric/trace interfaces,
//! then re-export from `mod.zig`.
//!
//! ## See Also
//!
//! - [Observability](../../shared/observability/README.md)
//! - [Monitoring Documentation](../../../docs/monitoring.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

