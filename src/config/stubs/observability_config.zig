const std = @import("std");

pub const ObservabilityConfig = struct {
    metrics_enabled: bool = false,
    tracing_enabled: bool = false,
    profiling_enabled: bool = false,
    metrics_endpoint: ?[]const u8 = null,
    trace_sample_rate: f32 = 0.0,

    pub fn defaults() ObservabilityConfig {
        return .{};
    }

    pub fn full() ObservabilityConfig {
        return .{};
    }
};
