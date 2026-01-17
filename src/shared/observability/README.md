//! # Observability Utilities
//!
//! Metrics, tracing, and health reporting for application monitoring.
//!
//! ## Features
//!
//! - **Metrics Collection**: Counters, gauges, histograms
//! - **Distributed Tracing**: Span context propagation
//! - **Health Reporting**: Readiness and liveness checks
//! - **Error Context**: Structured error tracking with categories
//!
//! ## Metrics Types
//!
//! | Type | Description | Use Case |
//! |------|-------------|----------|
//! | Counter | Monotonic increasing | Request counts, errors |
//! | Gauge | Point-in-time value | Memory usage, connections |
//! | Histogram | Distribution | Latency percentiles |
//!
//! ## Usage
//!
//! ### Metrics
//!
//! ```zig
//! const abi = @import("abi");
//!
//! var metrics = abi.monitoring.MetricsCollector.init(allocator);
//! defer metrics.deinit();
//!
//! metrics.recordTaskExecution(task_id, duration_ns);
//! metrics.incrementCounter("requests_total");
//! metrics.setGauge("memory_usage", current_usage);
//!
//! const summary = metrics.getSummary();
//! ```
//!
//! ### Error Context
//!
//! ```zig
//! const ErrorContext = @import("shared").utils.ErrorContext;
//!
//! var ctx = ErrorContext.init("database_query");
//! ctx.category = .database;
//! ctx.addDetail("query", query_string);
//! ctx.log();
//! ```
//!
//! ## Feature Flag
//!
//! Requires `-Denable-profiling=true` (default: enabled).
//!
//! ## See Also
//!
//! - [Monitoring Documentation](../../../docs/monitoring.md)
//! - [Logging Module](../logging/README.md)

