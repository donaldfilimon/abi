//! Observability Stub Module
//!
//! API-compatible no-op implementations when profiling is disabled.
//! Build with `-Dfeat-profiling=true` for the real implementation.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Shared types (from types.zig)
// ============================================================================

pub const shared_types = @import("types.zig");

// ============================================================================
// Local Stubs Imports
// ============================================================================

const types = @import("stubs/types.zig");
const metrics_stubs = @import("stubs/metrics.zig");
pub const tracing = @import("stubs/tracing.zig");
pub const alerting = @import("stubs/alerting.zig");
const exporters = @import("stubs/exporters.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const Error = shared_types.Error;
pub const MetricsConfig = shared_types.MetricsConfig;
pub const MetricsSummary = shared_types.MetricsSummary;
pub const BundleConfig = types.BundleConfig;
pub const PrometheusConfig = types.PrometheusConfig;
pub const OtelConfig = types.OtelConfig;
pub const MonitoringError = shared_types.MonitoringError;
pub const OtelSpanKind = types.OtelSpanKind;
pub const OtelMetricType = types.OtelMetricType;
pub const OtelStatus = types.OtelStatus;
pub const TraceId = types.TraceId;
pub const SpanId = types.SpanId;
pub const SpanKind = types.SpanKind;
pub const SpanStatus = types.SpanStatus;
pub const AttributeValue = types.AttributeValue;
pub const SpanAttribute = types.SpanAttribute;
pub const SpanEvent = types.SpanEvent;
pub const SpanLink = types.SpanLink;
pub const PropagationFormat = types.PropagationFormat;
pub const SamplerType = types.SamplerType;
pub const AlertState = types.AlertState;
pub const AlertSeverity = types.AlertSeverity;
pub const AlertError = types.AlertError;
pub const AlertCallback = types.AlertCallback;

// Metrics
pub const MetricsCollector = metrics_stubs.MetricsCollector;
pub const Counter = metrics_stubs.Counter;
pub const Gauge = metrics_stubs.Gauge;
pub const FloatGauge = metrics_stubs.FloatGauge;
pub const Histogram = metrics_stubs.Histogram;
pub const DefaultMetrics = metrics_stubs.DefaultMetrics;
pub const DefaultCollector = metrics_stubs.DefaultCollector;
pub const CircuitBreakerMetrics = metrics_stubs.CircuitBreakerMetrics;
pub const ErrorMetrics = metrics_stubs.ErrorMetrics;
pub const createCollector = metrics_stubs.createCollector;
pub const registerDefaultMetrics = metrics_stubs.registerDefaultMetrics;
pub const recordRequest = metrics_stubs.recordRequest;
pub const recordError = metrics_stubs.recordError;

// Tracing
pub const Tracer = tracing.Tracer;
pub const Span = tracing.Span;
pub const TraceContext = tracing.TraceContext;
pub const Propagator = tracing.Propagator;
pub const TraceSampler = tracing.TraceSampler;
pub const SpanProcessor = tracing.SpanProcessor;
pub const SpanExporter = tracing.SpanExporter;
pub const otel = tracing.otel;
pub const OtelExporter = tracing.OtelExporter;
pub const OtelTracer = tracing.OtelTracer;
pub const OtelSpan = tracing.OtelSpan;
pub const OtelContext = tracing.OtelContext;
pub const OtelMetric = tracing.OtelMetric;
pub const OtelAttribute = tracing.OtelAttribute;
pub const OtelAttributeValue = types.AttributeValue;
pub const OtelEvent = tracing.OtelEvent;
pub const formatTraceId = tracing.formatTraceId;
pub const formatSpanId = tracing.formatSpanId;
pub const createOtelResource = tracing.createOtelResource;

// Alerting
pub const AlertManager = alerting.AlertManager;
pub const AlertManagerConfig = alerting.AlertManagerConfig;
pub const AlertRule = alerting.AlertRule;
pub const AlertRuleBuilder = alerting.AlertRuleBuilder;
pub const Alert = alerting.Alert;
pub const AlertCondition = alerting.AlertCondition;
pub const AlertStats = alerting.AlertStats;
pub const AlertHandler = alerting.AlertHandler;
pub const MetricValues = alerting.MetricValues;
pub const createAlertRule = alerting.createAlertRule;

// Exporters
pub const prometheus = struct {
    pub const PrometheusExporter = exporters.PrometheusExporter;
    pub const PrometheusConfig = types.PrometheusConfig;
    pub const PrometheusFormatter = exporters.PrometheusFormatter;
    pub const generateMetricsOutput = exporters.generateMetricsOutput;
};
pub const PrometheusExporter = exporters.PrometheusExporter;
pub const PrometheusFormatter = exporters.PrometheusFormatter;
pub const generateMetricsOutput = exporters.generateMetricsOutput;
pub const statsd = exporters.statsd;

// Monitoring re-exports for parity with mod.zig
pub const monitoring = alerting;
// MonitoringError is re-exported from shared_types above
pub const StatsDClient = struct {};
pub const StatsDConfig = struct {};
pub const StatsDError = error{FeatureDisabled};

// Unified observability bundle (stub)
pub const ObservabilityBundle = struct {
    pub fn init(_: std.mem.Allocator, _: BundleConfig) Error!ObservabilityBundle {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *ObservabilityBundle) void {}
    pub fn start(_: *ObservabilityBundle) Error!void {
        return error.FeatureDisabled;
    }
    pub fn stop(_: *ObservabilityBundle) void {}
    pub fn startSpan(_: *ObservabilityBundle, _: []const u8) Error!?OtelSpan {
        return error.FeatureDisabled;
    }
};

// Context for Framework integration
pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.ObservabilityConfig) Error!*Context {
        return error.ObservabilityDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn recordMetric(_: *Context, _: []const u8, _: f64) Error!void {
        return error.FeatureDisabled;
    }
    pub fn startSpan(_: *Context, _: []const u8) Error!Span {
        return error.FeatureDisabled;
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

// System Information Helper (Stub)
pub const system_info = @import("stubs/system_info.zig");
pub const SystemInfo = system_info.SystemInfo;

// Core Metrics Module (Stub)
pub const core_metrics = @import("stubs/core_metrics.zig");

test {
    std.testing.refAllDecls(@This());
}
