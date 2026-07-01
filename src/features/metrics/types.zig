//! Shared types for the metrics feature.
//!
//! Lightweight in-process observability (counters, gauges) for scheduler,
//! AI pipeline, WDBX, and MCP surfaces. Opt-in for minimal builds.

const std = @import("std");

pub const MetricsError = error{
    FeatureDisabled,
    OutOfMemory,
    MetricNotFound,
};

pub const Error = MetricsError;

pub const CounterSnapshot = struct {
    name: []const u8,
    value: u64,
};

pub const GaugeSnapshot = struct {
    name: []const u8,
    value: f64,
};

test {
    std.testing.refAllDecls(@This());
}
