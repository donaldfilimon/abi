//! Observability Module
//!
//! Unified observability with metrics, tracing, and profiling.
//!
//! ## Features
//! - Metrics collection and export (Prometheus, OpenTelemetry, StatsD)
//! - Distributed tracing
//! - Performance profiling
//! - Circuit breakers and error aggregation
//! - Alerting rules and notifications
//!
//! ## Usage
//! ```zig
//! const observability = @import("observability");
//!
//! // Create metrics collector
//! var collector = observability.MetricsCollector.init(allocator);
//! defer collector.deinit();
//!
//! // Register and use metrics
//! const counter = try collector.registerCounter("requests_total");
//! counter.inc(1);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/monitoring and shared/observability
const features_monitoring = @import("../features/monitoring/mod.zig");
const shared_observability = @import("../shared/observability/mod.zig");

// Core metrics types
pub const MetricsCollector = features_monitoring.MetricsCollector;
pub const Counter = features_monitoring.Counter;
pub const Histogram = features_monitoring.Histogram;

// Convenience functions
pub const createCollector = features_monitoring.createCollector;
pub const registerDefaultMetrics = features_monitoring.registerDefaultMetrics;
pub const recordRequest = features_monitoring.recordRequest;
pub const recordError = features_monitoring.recordError;

// Default metrics bundle
pub const DefaultMetrics = features_monitoring.DefaultMetrics;
pub const DefaultCollector = features_monitoring.DefaultCollector;

// Circuit breaker and error metrics
pub const CircuitBreakerMetrics = features_monitoring.CircuitBreakerMetrics;
pub const ErrorMetrics = features_monitoring.ErrorMetrics;

// Unified observability bundle
pub const ObservabilityBundle = features_monitoring.ObservabilityBundle;
pub const BundleConfig = features_monitoring.BundleConfig;

// Alerting
pub const alerting = features_monitoring.alerting;
pub const AlertManager = features_monitoring.AlertManager;
pub const AlertManagerConfig = features_monitoring.AlertManagerConfig;
pub const AlertRule = features_monitoring.AlertRule;
pub const AlertRuleBuilder = features_monitoring.AlertRuleBuilder;
pub const Alert = features_monitoring.Alert;
pub const AlertState = features_monitoring.AlertState;
pub const AlertSeverity = features_monitoring.AlertSeverity;
pub const AlertCondition = features_monitoring.AlertCondition;
pub const AlertError = features_monitoring.AlertError;
pub const AlertStats = features_monitoring.AlertStats;
pub const AlertCallback = features_monitoring.AlertCallback;
pub const AlertHandler = features_monitoring.AlertHandler;
pub const MetricValues = features_monitoring.MetricValues;
pub const createAlertRule = features_monitoring.createAlertRule;

// Prometheus
pub const prometheus = features_monitoring.prometheus;
pub const PrometheusExporter = features_monitoring.PrometheusExporter;
pub const PrometheusConfig = features_monitoring.PrometheusConfig;
pub const PrometheusFormatter = features_monitoring.PrometheusFormatter;
pub const generateMetricsOutput = features_monitoring.generateMetricsOutput;

// OpenTelemetry
pub const otel = features_monitoring.otel;
pub const OtelExporter = features_monitoring.OtelExporter;
pub const OtelConfig = features_monitoring.OtelConfig;
pub const OtelTracer = features_monitoring.OtelTracer;
pub const OtelSpan = features_monitoring.OtelSpan;
pub const OtelSpanKind = features_monitoring.OtelSpanKind;
pub const OtelContext = features_monitoring.OtelContext;
pub const OtelMetric = features_monitoring.OtelMetric;
pub const OtelMetricType = features_monitoring.OtelMetricType;
pub const OtelAttribute = features_monitoring.OtelAttribute;
pub const OtelAttributeValue = features_monitoring.OtelAttributeValue;
pub const OtelEvent = features_monitoring.OtelEvent;
pub const OtelStatus = features_monitoring.OtelStatus;
pub const formatTraceId = features_monitoring.formatTraceId;
pub const formatSpanId = features_monitoring.formatSpanId;
pub const createOtelResource = features_monitoring.createOtelResource;

// StatsD
pub const statsd = features_monitoring.statsd;

// Legacy type aliases for backward compatibility
pub const MetricsConfig = struct {};
pub const MetricsSummary = struct {};

// Tracing from shared/observability
pub const Tracer = shared_observability.Tracer;
pub const Span = shared_observability.Span;

pub const Error = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

/// Observability context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.ObservabilityConfig,
    metrics: ?*MetricsCollector = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.ObservabilityConfig) !*Context {
        if (!isEnabled()) return error.ObservabilityDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        // Initialize metrics if enabled
        if (cfg.metrics_enabled) {
            const collector = try allocator.create(MetricsCollector);
            collector.* = MetricsCollector.init(allocator);
            ctx.metrics = collector;
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.metrics) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.allocator.destroy(self);
    }

    /// Record a metric.
    pub fn recordMetric(self: *Context, name: []const u8, value: f64) !void {
        if (self.metrics) |m| {
            try m.record(name, value);
        }
    }

    /// Start a trace span.
    pub fn startSpan(self: *Context, name: []const u8) !Span {
        _ = self;
        return Span.start(name);
    }

    /// Get metrics summary.
    pub fn getSummary(self: *Context) ?MetricsSummary {
        if (self.metrics) |m| {
            return m.getSummary();
        }
        return null;
    }
};

pub fn isEnabled() bool {
    return build_options.enable_profiling;
}

pub fn isInitialized() bool {
    return features_monitoring.isInitialized();
}

pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.ObservabilityDisabled;
    features_monitoring.init(allocator) catch return error.ObservabilityDisabled;
}

pub fn deinit() void {
    features_monitoring.deinit();
}
