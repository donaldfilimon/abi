//! Observability Stub Module
//!
//! Stub implementation when profiling/observability is disabled.
//! All functions return ObservabilityDisabled error or no-op.

const std = @import("std");
const config_module = @import("../config.zig");

pub const Error = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

// Core metrics types (stubs)
pub const MetricsCollector = struct {
    pub fn init(_: std.mem.Allocator) MetricsCollector {
        return .{};
    }
    pub fn deinit(_: *MetricsCollector) void {}
    pub fn registerCounter(_: *MetricsCollector, _: []const u8) Error!*Counter {
        return error.ObservabilityDisabled;
    }
    pub fn registerHistogram(_: *MetricsCollector, _: []const u8, _: []const u64) Error!*Histogram {
        return error.ObservabilityDisabled;
    }
};

pub const Counter = struct {
    pub fn inc(_: *Counter, _: u64) void {}
    pub fn get(_: *Counter) u64 {
        return 0;
    }
};

pub const Histogram = struct {
    pub fn record(_: *Histogram, _: u64) void {}
};

pub const MetricsConfig = struct {};
pub const MetricsSummary = struct {};

// Convenience functions (stubs)
pub fn createCollector(_: std.mem.Allocator) MetricsCollector {
    return .{};
}

pub fn registerDefaultMetrics(_: *MetricsCollector) Error!DefaultMetrics {
    return error.ObservabilityDisabled;
}

pub fn recordRequest(_: *DefaultMetrics, _: u64) void {}
pub fn recordError(_: *DefaultMetrics, _: u64) void {}

pub const DefaultMetrics = struct {
    requests: ?*Counter = null,
    errors: ?*Counter = null,
    latency_ms: ?*Histogram = null,
};

pub const DefaultCollector = struct {
    pub fn init(_: std.mem.Allocator) Error!DefaultCollector {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *DefaultCollector) void {}
};

// Circuit breaker and error metrics (stubs)
pub const CircuitBreakerMetrics = struct {
    pub fn init(_: *MetricsCollector) Error!CircuitBreakerMetrics {
        return error.ObservabilityDisabled;
    }
    pub fn recordRequest(_: *CircuitBreakerMetrics, _: bool, _: u64) void {}
    pub fn recordStateTransition(_: *CircuitBreakerMetrics) void {}
};

pub const ErrorMetrics = struct {
    pub fn init(_: *MetricsCollector) Error!ErrorMetrics {
        return error.ObservabilityDisabled;
    }
    pub fn recordError(_: *ErrorMetrics, _: bool) void {}
    pub fn recordPattern(_: *ErrorMetrics) void {}
};

// Unified observability bundle (stub)
pub const ObservabilityBundle = struct {
    pub fn init(_: std.mem.Allocator, _: BundleConfig) Error!ObservabilityBundle {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *ObservabilityBundle) void {}
    pub fn start(_: *ObservabilityBundle) Error!void {
        return error.ObservabilityDisabled;
    }
    pub fn stop(_: *ObservabilityBundle) void {}
    pub fn startSpan(_: *ObservabilityBundle, _: []const u8) Error!?OtelSpan {
        return error.ObservabilityDisabled;
    }
};

pub const BundleConfig = struct {
    enable_circuit_breaker_metrics: bool = true,
    enable_error_metrics: bool = true,
    prometheus: ?PrometheusConfig = null,
    otel: ?OtelConfig = null,
};

// Alerting (stubs)
pub const alerting = struct {};
pub const AlertManager = struct {};
pub const AlertManagerConfig = struct {};
pub const AlertRule = struct {};
pub const AlertRuleBuilder = struct {};
pub const Alert = struct {};
pub const AlertState = enum { pending, firing, resolved };
pub const AlertSeverity = enum { info, warning, critical };
pub const AlertCondition = struct {};
pub const AlertError = error{AlertDisabled};
pub const AlertStats = struct {};
pub const AlertCallback = *const fn () void;
pub const AlertHandler = struct {};
pub const MetricValues = struct {};
pub fn createAlertRule() Error!AlertRule {
    return error.ObservabilityDisabled;
}

// Prometheus (stubs)
pub const prometheus = struct {};
pub const PrometheusExporter = struct {};
pub const PrometheusConfig = struct {};
pub const PrometheusFormatter = struct {};
pub fn generateMetricsOutput(_: std.mem.Allocator, _: *MetricsCollector) Error![]const u8 {
    return error.ObservabilityDisabled;
}

// OpenTelemetry (stubs)
pub const otel = struct {};
pub const OtelExporter = struct {};
pub const OtelConfig = struct {
    service_name: []const u8 = "",
};
pub const OtelTracer = struct {};
pub const OtelSpan = struct {};
pub const OtelSpanKind = enum { internal, server, client };
pub const OtelContext = struct {};
pub const OtelMetric = struct {};
pub const OtelMetricType = enum { counter, gauge, histogram };
pub const OtelAttribute = struct {};
pub const OtelAttributeValue = union { string: []const u8, int: i64, float: f64, bool: bool };
pub const OtelEvent = struct {};
pub const OtelStatus = enum { ok, @"error" };
pub fn formatTraceId(_: [16]u8) [32]u8 {
    return [_]u8{0} ** 32;
}
pub fn formatSpanId(_: [8]u8) [16]u8 {
    return [_]u8{0} ** 16;
}
pub fn createOtelResource(_: std.mem.Allocator, _: []const u8) Error!void {
    return error.ObservabilityDisabled;
}

// StatsD (stub)
pub const statsd = struct {};

// Tracing
pub const Tracer = struct {};
pub const Span = struct {
    pub fn start(_: []const u8) Span {
        return .{};
    }
    pub fn end(_: *Span) void {}
};

// Context for Framework integration
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
