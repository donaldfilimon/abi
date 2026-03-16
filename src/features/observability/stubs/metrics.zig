const std = @import("std");
const types = @import("types.zig");

// Core metrics types (stubs)
pub const MetricsCollector = struct {
    pub fn init(_: std.mem.Allocator) MetricsCollector {
        return .{};
    }
    pub fn deinit(_: *MetricsCollector) void {}
    pub fn registerCounter(_: *MetricsCollector, _: []const u8) types.Error!*Counter {
        return error.ObservabilityDisabled;
    }
    pub fn registerGauge(_: *MetricsCollector, _: []const u8) types.Error!*Gauge {
        return error.ObservabilityDisabled;
    }
    pub fn registerFloatGauge(_: *MetricsCollector, _: []const u8) types.Error!*FloatGauge {
        return error.ObservabilityDisabled;
    }
    pub fn registerHistogram(_: *MetricsCollector, _: []const u8, _: []const u64) types.Error!*Histogram {
        return error.ObservabilityDisabled;
    }
};

pub const Counter = struct {
    pub fn inc(_: *Counter, _: u64) void {}
    pub fn get(_: *const Counter) u64 {
        return 0;
    }
    pub fn reset(_: *Counter) void {}
};

/// Gauge metric - stub implementation.
pub const Gauge = struct {
    name: []const u8 = "",
    pub fn set(_: *Gauge, _: i64) void {}
    pub fn inc(_: *Gauge) void {}
    pub fn dec(_: *Gauge) void {}
    pub fn add(_: *Gauge, _: i64) void {}
    pub fn get(_: *const Gauge) i64 {
        return 0;
    }
};

/// Float gauge metric - stub implementation.
/// Note: get() requires mutable reference for mutex locking in the real implementation.
pub const FloatGauge = struct {
    name: []const u8 = "",
    pub fn set(_: *FloatGauge, _: f64) void {}
    pub fn add(_: *FloatGauge, _: f64) void {}
    pub fn get(_: *FloatGauge) f64 {
        return 0.0;
    }
};

pub const Histogram = struct {
    pub fn record(_: *Histogram, _: u64) void {}
};

pub const DefaultMetrics = struct {
    requests: ?*Counter = null,
    errors: ?*Counter = null,
    latency_ms: ?*Histogram = null,
};

pub const DefaultCollector = struct {
    pub fn init(_: std.mem.Allocator) types.Error!DefaultCollector {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *DefaultCollector) void {}
};

// Circuit breaker and error metrics (stubs)
pub const CircuitBreakerMetrics = struct {
    pub fn init(_: *MetricsCollector) types.Error!CircuitBreakerMetrics {
        return error.ObservabilityDisabled;
    }
    pub fn recordRequest(_: *CircuitBreakerMetrics, _: bool, _: u64) void {}
    pub fn recordStateTransition(_: *CircuitBreakerMetrics) void {}
};

pub const ErrorMetrics = struct {
    pub fn init(_: *MetricsCollector) types.Error!ErrorMetrics {
        return error.ObservabilityDisabled;
    }
    pub fn recordError(_: *ErrorMetrics, _: bool) void {}
    pub fn recordPattern(_: *ErrorMetrics) void {}
};

// Convenience functions (stubs)
pub fn createCollector(_: std.mem.Allocator) MetricsCollector {
    return .{};
}

pub fn registerDefaultMetrics(_: *MetricsCollector) types.Error!DefaultMetrics {
    return error.ObservabilityDisabled;
}

pub fn recordRequest(_: *DefaultMetrics, _: u64) void {}
pub fn recordError(_: *DefaultMetrics, _: u64) void {}

test {
    std.testing.refAllDecls(@This());
}
