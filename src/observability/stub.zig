//! Observability Stub Module
//!
//! Stub implementation when profiling/observability is disabled.
//! All functions return ObservabilityDisabled error or no-op.

const std = @import("std");
const config_module = @import("../config/mod.zig");

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

// Tracing types (API-compatible stubs)
pub const TraceId = [16]u8;
pub const SpanId = [8]u8;
pub const SpanKind = enum { internal, server, client, producer, consumer };
pub const SpanStatus = enum { unset, ok, @"error" };
pub const AttributeValue = union(enum) { string: []const u8, int: i64, float: f64, bool: bool };
pub const SpanAttribute = struct { key: []const u8, value: AttributeValue };
pub const SpanEvent = struct { name: []const u8, timestamp: i64 = 0, attributes: []const SpanAttribute = &.{} };
pub const SpanLink = struct { trace_id: TraceId, span_id: SpanId, attributes: []const SpanAttribute = &.{} };

pub const Tracer = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) Error!Tracer {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *Tracer) void {}
    pub fn startSpan(_: *Tracer, _: []const u8, _: ?*const TraceContext, _: SpanKind) Error!Span {
        return error.ObservabilityDisabled;
    }
};

pub const Span = struct {
    pub fn start(_: std.mem.Allocator, _: []const u8, _: ?TraceId, _: ?SpanId, _: SpanKind) Error!Span {
        return error.ObservabilityDisabled;
    }
    pub fn end(_: *Span) void {}
    pub fn deinit(_: *Span) void {}
    pub fn setAttribute(_: *Span, _: []const u8, _: AttributeValue) Error!void {
        return error.ObservabilityDisabled;
    }
    pub fn addEvent(_: *Span, _: []const u8) Error!void {
        return error.ObservabilityDisabled;
    }
};

pub const TraceContext = struct {
    trace_id: TraceId = [_]u8{0} ** 16,
    span_id: SpanId = [_]u8{0} ** 8,
    is_remote: bool = false,
    trace_flags: u8 = 0x01,

    pub fn extract(_: []const u8) TraceContext {
        return .{};
    }
    pub fn inject(_: TraceContext, _: []u8) usize {
        return 0;
    }
};

pub const PropagationFormat = enum { w3c, b3, jaeger, aws_xray };
pub const Propagator = struct {
    pub fn init(_: PropagationFormat) Propagator {
        return .{};
    }
};

pub const TraceSampler = struct {
    pub fn init(_: SamplerType, _: f64) TraceSampler {
        return .{};
    }
    pub fn shouldSample(_: *TraceSampler, _: TraceId) bool {
        return false;
    }

    pub const SamplerType = enum { always_on, always_off, trace_id_ratio };
};

pub const SpanProcessor = struct {
    pub fn init(_: std.mem.Allocator) SpanProcessor {
        return .{};
    }
    pub fn deinit(_: *SpanProcessor) void {}
};

pub const SpanExporter = struct {
    pub fn init(_: std.mem.Allocator) SpanExporter {
        return .{};
    }
    pub fn deinit(_: *SpanExporter) void {}
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
