//! Observability Configuration
//!
//! Configuration for metrics, tracing, and profiling.

const std = @import("std");

/// Observability and monitoring configuration.
pub const ObservabilityConfig = struct {
    /// Enable metrics collection.
    metrics_enabled: bool = true,

    /// Enable distributed tracing.
    tracing_enabled: bool = true,

    /// Enable performance profiling.
    profiling_enabled: bool = false,

    /// Metrics export endpoint.
    metrics_endpoint: ?[]const u8 = null,

    /// Trace sampling rate (0.0 - 1.0).
    trace_sample_rate: f32 = 0.1,

    pub fn defaults() ObservabilityConfig {
        return .{};
    }

    /// Full observability (all features enabled).
    pub fn full() ObservabilityConfig {
        return .{
            .metrics_enabled = true,
            .tracing_enabled = true,
            .profiling_enabled = true,
            .trace_sample_rate = 1.0,
        };
    }
};
