const std = @import("std");
const types = @import("types.zig");

// Prometheus (stubs)
pub const PrometheusExporter = struct {};
pub const PrometheusFormatter = struct {};

pub fn generateMetricsOutput(_: std.mem.Allocator, _: *anyopaque) types.Error![]const u8 {
    return error.ObservabilityDisabled;
}

// StatsD (stub)
pub const statsd = struct {};
