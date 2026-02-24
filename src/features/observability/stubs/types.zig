const std = @import("std");

pub const Error = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

pub const MetricsConfig = struct {};
pub const MetricsSummary = struct {};

pub const BundleConfig = struct {
    enable_circuit_breaker_metrics: bool = true,
    enable_error_metrics: bool = true,
    prometheus: ?PrometheusConfig = null,
    otel: ?OtelConfig = null,
};

pub const PrometheusConfig = struct {};
pub const OtelConfig = struct {
    service_name: []const u8 = "",
};

pub const OtelSpanKind = enum { internal, server, client };
pub const OtelMetricType = enum { counter, gauge, histogram };
pub const OtelStatus = enum { ok, @"error" };

pub const TraceId = [16]u8;
pub const SpanId = [8]u8;
pub const SpanKind = enum { internal, server, client, producer, consumer };
pub const SpanStatus = enum { unset, ok, @"error" };
pub const AttributeValue = union(enum) { string: []const u8, int: i64, float: f64, bool: bool };
pub const SpanAttribute = struct { key: []const u8, value: AttributeValue };
pub const SpanEvent = struct { name: []const u8, timestamp: i64 = 0, attributes: []const SpanAttribute = &.{} };
pub const SpanLink = struct { trace_id: TraceId, span_id: SpanId, attributes: []const SpanAttribute = &.{} };

pub const PropagationFormat = enum { w3c, b3, jaeger, aws_xray };
pub const SamplerType = enum { always_on, always_off, trace_id_ratio };

pub const AlertState = enum { pending, firing, resolved };
pub const AlertSeverity = enum { info, warning, critical };
pub const AlertError = error{AlertDisabled};
pub const AlertCallback = *const fn () void;

test {
    std.testing.refAllDecls(@This());
}
