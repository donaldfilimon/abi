//! Observability Metrics Module
//!
//! Provides core metric primitives and domain-specific metric types.
//! These primitives are designed to be shared across the codebase for
//! consistent metrics collection and export.
//!
//! ## Quick Start
//!
//! ```zig
//! const metrics = @import("observability/metrics/mod.zig");
//!
//! // Create a counter
//! var requests = metrics.Counter{};
//! requests.inc();
//!
//! // Create a histogram
//! var latency = metrics.LatencyHistogram.initDefault();
//! latency.observe(42.5);
//!
//! // Export to Prometheus format
//! var writer = metrics.MetricWriter.init(allocator);
//! defer writer.deinit();
//! try writer.writeCounter("requests_total", "Total requests", requests.get(), null);
//! const output = try writer.finish();
//! ```
//!
//! ## Thread Safety
//!
//! All metric types are thread-safe:
//! - Counter and Gauge use atomic operations
//! - FloatGauge and Histogram use mutex protection
//! - SlidingWindow uses mutex protection

pub const primitives = @import("primitives.zig");
pub const prometheus = @import("prometheus.zig");
pub const sliding_window = @import("sliding_window.zig");
pub const collector = @import("collector.zig");
const std = @import("std");

// Re-export core types (lightweight, no name field)
pub const Counter = primitives.Counter;
pub const Gauge = primitives.Gauge;
pub const FloatGauge = primitives.FloatGauge;
pub const Histogram = primitives.Histogram;
pub const LatencyHistogram = primitives.LatencyHistogram;
pub const default_latency_buckets = primitives.default_latency_buckets;

// Re-export collector types (named, for MetricsCollector)
pub const MetricsCollector = collector.MetricsCollector;
pub const DefaultMetrics = collector.DefaultMetrics;
pub const DefaultCollector = collector.DefaultCollector;
pub const CircuitBreakerMetrics = collector.CircuitBreakerMetrics;
pub const ErrorMetrics = collector.ErrorMetrics;
pub const createCollector = collector.createCollector;
pub const registerDefaultMetrics = collector.registerDefaultMetrics;
pub const recordRequest = collector.recordRequest;
pub const recordError = collector.recordError;

// Re-export prometheus types
pub const MetricWriter = prometheus.MetricWriter;

// Re-export sliding window types
pub const SlidingWindow = sliding_window.SlidingWindow;
pub const StandardWindow = sliding_window.StandardWindow;
pub const TimestampedSample = sliding_window.TimestampedSample;

test {
    _ = primitives;
    _ = prometheus;
    _ = sliding_window;
    _ = collector;
}

test {
    std.testing.refAllDecls(@This());
}
