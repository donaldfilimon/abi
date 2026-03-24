//! Observability Stub Module
//!
//! API-compatible no-op implementations when observability is disabled.
//! Build with `-Dfeat-observability=true` for the real implementation.

const std = @import("std");
const stub_helpers = @import("../../core/stub_helpers.zig");
const config_module = @import("../../core/config/mod.zig");

// --- Shared types (from types.zig) ---
pub const types = @import("types.zig");
pub const Error = types.Error;
pub const MetricsConfig = types.MetricsConfig;
pub const MetricsSummary = types.MetricsSummary;
pub const MonitoringError = types.MonitoringError;

// --- Local Stubs Imports ---
const stub_types = @import("stubs/types.zig");
const metrics_stubs = @import("stubs/metrics.zig");
pub const tracing = @import("stubs/tracing.zig");
pub const alerting = @import("stubs/alerting.zig");
const exporters = @import("stubs/exporters.zig");

// --- Stub-only types from stubs/types.zig ---
pub const BundleConfig = stub_types.BundleConfig;
pub const PrometheusConfig = stub_types.PrometheusConfig;
pub const OtelConfig = stub_types.OtelConfig;
pub const OtelSpanKind = stub_types.OtelSpanKind;
pub const OtelMetricType = stub_types.OtelMetricType;
pub const OtelStatus = stub_types.OtelStatus;
pub const TraceId = stub_types.TraceId;
pub const SpanId = stub_types.SpanId;
pub const SpanKind = stub_types.SpanKind;
pub const SpanStatus = stub_types.SpanStatus;
pub const AttributeValue = stub_types.AttributeValue;
pub const SpanAttribute = stub_types.SpanAttribute;
pub const SpanEvent = stub_types.SpanEvent;
pub const SpanLink = stub_types.SpanLink;
pub const PropagationFormat = stub_types.PropagationFormat;
pub const SamplerType = stub_types.SamplerType;
pub const AlertState = stub_types.AlertState;
pub const AlertSeverity = stub_types.AlertSeverity;
pub const AlertError = stub_types.AlertError;
pub const AlertCallback = stub_types.AlertCallback;
pub const ObservabilityBundle = stub_types.ObservabilityBundle;
pub const StatsDClient = stub_types.StatsDClient;
pub const StatsDConfig = stub_types.StatsDConfig;
pub const StatsDError = stub_types.StatsDError;

// --- Metrics ---
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

// --- Tracing ---
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
pub const OtelAttributeValue = stub_types.AttributeValue;
pub const OtelEvent = tracing.OtelEvent;
pub const formatTraceId = tracing.formatTraceId;
pub const formatSpanId = tracing.formatSpanId;
pub const createOtelResource = tracing.createOtelResource;

// --- Alerting ---
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

// --- Exporters ---
pub const prometheus = struct {
    pub const PrometheusExporter = exporters.PrometheusExporter;
    pub const PrometheusConfig = stub_types.PrometheusConfig;
    pub const PrometheusFormatter = exporters.PrometheusFormatter;
    pub const generateMetricsOutput = exporters.generateMetricsOutput;
};
pub const PrometheusExporter = exporters.PrometheusExporter;
pub const PrometheusFormatter = exporters.PrometheusFormatter;
pub const generateMetricsOutput = exporters.generateMetricsOutput;
pub const statsd = exporters.statsd;

// --- Monitoring ---
pub const monitoring = alerting;

// --- Context ---
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

// --- Module Lifecycle ---
const _stub = stub_helpers.StubFeatureNoConfig(error{FeatureDisabled});
pub const init = _stub.init;
pub const deinit = _stub.deinit;
pub const isEnabled = _stub.isEnabled;
pub const isInitialized = _stub.isInitialized;

// --- System Information & Core Metrics ---
pub const system_info = @import("stubs/system_info.zig");
pub const SystemInfo = system_info.SystemInfo;
pub const core_metrics = @import("stubs/core_metrics.zig");

test {
    std.testing.refAllDecls(@This());
}
