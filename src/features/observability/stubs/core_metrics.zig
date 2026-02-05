//! Core Metrics Module (Stub)
//!
//! Provides stub implementations of shared metric primitives when observability
//! is disabled. These stubs maintain API parity with the real implementation.

pub const primitives = @import("./core_metrics/primitives.zig");
pub const prometheus = @import("./core_metrics/prometheus.zig");
pub const sliding_window = @import("./core_metrics/sliding_window.zig");

// Re-export core types
pub const Counter = primitives.Counter;
pub const Gauge = primitives.Gauge;
pub const FloatGauge = primitives.FloatGauge;
pub const Histogram = primitives.Histogram;
pub const LatencyHistogram = primitives.LatencyHistogram;
pub const default_latency_buckets = primitives.default_latency_buckets;

// Re-export prometheus types
pub const MetricWriter = prometheus.MetricWriter;

// Re-export sliding window types
pub const SlidingWindow = sliding_window.SlidingWindow;
pub const StandardWindow = sliding_window.StandardWindow;
pub const TimestampedSample = sliding_window.TimestampedSample;
