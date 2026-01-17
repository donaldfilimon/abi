//! Observability Stub Module

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

pub const MetricsCollector = struct {};
pub const MetricsConfig = struct {};
pub const MetricsSummary = struct {};
pub const Tracer = struct {};
pub const Span = struct {
    pub fn start(_: []const u8) Span {
        return .{};
    }
    pub fn end(_: *Span) void {}
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.ObservabilityConfig) Error!*Context {
        return error.ObservabilityDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn recordMetric(_: *Context, _: []const u8, _: f64) Error!void {
        return error.ObservabilityDisabled;
    }

    pub fn startSpan(_: *Context, _: []const u8) Error!Span {
        return error.ObservabilityDisabled;
    }

    pub fn getSummary(_: *Context) ?MetricsSummary {
        return null;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.ObservabilityDisabled;
}

pub fn deinit() void {}
