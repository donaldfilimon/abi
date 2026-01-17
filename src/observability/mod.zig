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

// Import from local files (consolidated from features/monitoring)
pub const alerting = @import("alerting.zig");
pub const prometheus = @import("prometheus.zig");
pub const otel = @import("otel.zig");
pub const statsd = @import("statsd.zig");

// Core metrics from shared/observability
const shared_observability = @import("../shared/observability/mod.zig");

// Core metrics types
pub const MetricsCollector = shared_observability.MetricsCollector;
pub const Counter = shared_observability.Counter;
pub const Histogram = shared_observability.Histogram;

// Alerting exports
pub const AlertManager = alerting.AlertManager;
pub const AlertManagerConfig = alerting.AlertManagerConfig;
pub const AlertRule = alerting.AlertRule;
pub const AlertRuleBuilder = alerting.AlertRuleBuilder;
pub const Alert = alerting.Alert;
pub const AlertState = alerting.AlertState;
pub const AlertSeverity = alerting.AlertSeverity;
pub const AlertCondition = alerting.AlertCondition;
pub const AlertError = alerting.AlertError;
pub const AlertStats = alerting.AlertStats;
pub const AlertCallback = alerting.AlertCallback;
pub const AlertHandler = alerting.AlertHandler;
pub const MetricValues = alerting.MetricValues;
pub const createAlertRule = alerting.createRule;

// Prometheus exports
pub const PrometheusExporter = prometheus.PrometheusExporter;
pub const PrometheusConfig = prometheus.PrometheusConfig;
pub const PrometheusFormatter = prometheus.PrometheusFormatter;
pub const generateMetricsOutput = prometheus.generateMetricsOutput;

// OpenTelemetry exports
pub const OtelExporter = otel.OtelExporter;
pub const OtelConfig = otel.OtelConfig;
pub const OtelTracer = otel.OtelTracer;
pub const OtelSpan = otel.OtelSpan;
pub const OtelSpanKind = otel.OtelSpanKind;
pub const OtelContext = otel.OtelContext;
pub const OtelMetric = otel.OtelMetric;
pub const OtelMetricType = otel.OtelMetricType;
pub const OtelAttribute = otel.OtelAttribute;
pub const OtelAttributeValue = otel.OtelAttributeValue;
pub const OtelEvent = otel.OtelEvent;
pub const OtelStatus = otel.OtelStatus;
pub const formatTraceId = otel.formatTraceId;
pub const formatSpanId = otel.formatSpanId;
pub const createOtelResource = otel.createOtelResource;

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

pub const MonitoringError = error{
    MonitoringDisabled,
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return MonitoringError.MonitoringDisabled;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_profiling;
}

pub fn isInitialized() bool {
    return initialized;
}

pub const DefaultMetrics = struct {
    requests: *Counter,
    errors: *Counter,
    latency_ms: *Histogram,
};

pub const DefaultCollector = struct {
    collector: MetricsCollector,
    defaults: DefaultMetrics,

    pub fn init(allocator: std.mem.Allocator) !DefaultCollector {
        var collector = MetricsCollector.init(allocator);
        errdefer collector.deinit();
        const defaults = try registerDefaultMetrics(&collector);
        return .{
            .collector = collector,
            .defaults = defaults,
        };
    }

    pub fn deinit(self: *DefaultCollector) void {
        self.collector.deinit();
        self.* = undefined;
    }
};

pub fn createCollector(allocator: std.mem.Allocator) MetricsCollector {
    return MetricsCollector.init(allocator);
}

const DEFAULT_LATENCY_BOUNDS = [_]u64{ 1, 5, 10, 25, 50, 100, 250, 500, 1000 };

pub fn registerDefaultMetrics(collector: *MetricsCollector) !DefaultMetrics {
    const requests = try collector.registerCounter("requests_total");
    const errors = try collector.registerCounter("errors_total");
    const latency = try collector.registerHistogram("latency_ms", &DEFAULT_LATENCY_BOUNDS);
    return .{
        .requests = requests,
        .errors = errors,
        .latency_ms = latency,
    };
}

pub fn recordRequest(metrics: *DefaultMetrics, latency_ms: u64) void {
    metrics.requests.inc(1);
    metrics.latency_ms.record(latency_ms);
}

pub fn recordError(metrics: *DefaultMetrics, latency_ms: u64) void {
    metrics.errors.inc(1);
    metrics.latency_ms.record(latency_ms);
}

// ============================================================================
// Circuit Breaker Metrics Integration
// ============================================================================

/// Circuit breaker metrics for observability.
pub const CircuitBreakerMetrics = struct {
    requests_total: *Counter,
    requests_rejected: *Counter,
    state_transitions: *Counter,
    latency_ms: *Histogram,

    pub fn init(collector: *MetricsCollector) !CircuitBreakerMetrics {
        return .{
            .requests_total = try collector.registerCounter("circuit_breaker_requests_total"),
            .requests_rejected = try collector.registerCounter("circuit_breaker_requests_rejected"),
            .state_transitions = try collector.registerCounter("circuit_breaker_state_transitions"),
            .latency_ms = try collector.registerHistogram("circuit_breaker_latency_ms", &DEFAULT_LATENCY_BOUNDS),
        };
    }

    pub fn recordRequest(self: *CircuitBreakerMetrics, success: bool, latency_ms: u64) void {
        self.requests_total.inc(1);
        self.latency_ms.record(latency_ms);
        if (!success) {
            self.requests_rejected.inc(1);
        }
    }

    pub fn recordStateTransition(self: *CircuitBreakerMetrics) void {
        self.state_transitions.inc(1);
    }
};

/// Error aggregation metrics for observability.
pub const ErrorMetrics = struct {
    errors_total: *Counter,
    errors_critical: *Counter,
    patterns_detected: *Counter,

    pub fn init(collector: *MetricsCollector) !ErrorMetrics {
        return .{
            .errors_total = try collector.registerCounter("errors_total"),
            .errors_critical = try collector.registerCounter("errors_critical"),
            .patterns_detected = try collector.registerCounter("error_patterns_detected"),
        };
    }

    pub fn recordError(self: *ErrorMetrics, is_critical: bool) void {
        self.errors_total.inc(1);
        if (is_critical) {
            self.errors_critical.inc(1);
        }
    }

    pub fn recordPattern(self: *ErrorMetrics) void {
        self.patterns_detected.inc(1);
    }
};

/// Unified observability bundle combining all monitoring features.
pub const ObservabilityBundle = struct {
    allocator: std.mem.Allocator,
    collector: MetricsCollector,
    defaults: DefaultMetrics,
    circuit_breaker: ?CircuitBreakerMetrics,
    errors: ?ErrorMetrics,
    prometheus_exporter: ?*PrometheusExporter,
    otel_exporter: ?*OtelExporter,
    tracer: ?*OtelTracer,

    pub fn init(allocator: std.mem.Allocator, config: BundleConfig) !ObservabilityBundle {
        var collector = MetricsCollector.init(allocator);
        errdefer collector.deinit();

        const defaults = try registerDefaultMetrics(&collector);

        var cb_metrics: ?CircuitBreakerMetrics = null;
        if (config.enable_circuit_breaker_metrics) {
            cb_metrics = try CircuitBreakerMetrics.init(&collector);
        }

        var err_metrics: ?ErrorMetrics = null;
        if (config.enable_error_metrics) {
            err_metrics = try ErrorMetrics.init(&collector);
        }

        var prom: ?*PrometheusExporter = null;
        if (config.prometheus) |prom_config| {
            const exporter = try allocator.create(PrometheusExporter);
            exporter.* = try PrometheusExporter.init(allocator, prom_config, &collector);
            prom = exporter;
        }

        var otel_exp: ?*OtelExporter = null;
        if (config.otel) |otel_config| {
            const exporter = try allocator.create(OtelExporter);
            exporter.* = try OtelExporter.init(allocator, otel_config);
            otel_exp = exporter;
        }

        var tracer: ?*OtelTracer = null;
        if (config.otel) |otel_config| {
            const t = try allocator.create(OtelTracer);
            t.* = try OtelTracer.init(allocator, otel_config.service_name);
            tracer = t;
        }

        return .{
            .allocator = allocator,
            .collector = collector,
            .defaults = defaults,
            .circuit_breaker = cb_metrics,
            .errors = err_metrics,
            .prometheus_exporter = prom,
            .otel_exporter = otel_exp,
            .tracer = tracer,
        };
    }

    pub fn deinit(self: *ObservabilityBundle) void {
        if (self.tracer) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        if (self.otel_exporter) |e| {
            e.deinit();
            self.allocator.destroy(e);
        }
        if (self.prometheus_exporter) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        self.collector.deinit();
        self.* = undefined;
    }

    /// Start all exporters.
    pub fn start(self: *ObservabilityBundle) !void {
        if (self.prometheus_exporter) |p| {
            try p.start();
        }
        if (self.otel_exporter) |e| {
            try e.start();
        }
    }

    /// Stop all exporters.
    pub fn stop(self: *ObservabilityBundle) void {
        if (self.prometheus_exporter) |p| {
            p.stop();
        }
        if (self.otel_exporter) |e| {
            e.stop();
        }
    }

    /// Start a new span with the tracer.
    pub fn startSpan(self: *ObservabilityBundle, name: []const u8) !?OtelSpan {
        if (self.tracer) |t| {
            return try t.startSpan(name, null, null);
        }
        return null;
    }
};

pub const BundleConfig = struct {
    enable_circuit_breaker_metrics: bool = true,
    enable_error_metrics: bool = true,
    prometheus: ?PrometheusConfig = null,
    otel: ?OtelConfig = null,
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

// Tests

test "default metrics register" {
    var collector = MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();
    const defaults = try registerDefaultMetrics(&collector);
    defaults.requests.inc(1);
    defaults.errors.inc(2);
    defaults.latency_ms.record(42);
    try std.testing.expectEqual(@as(u64, 1), defaults.requests.get());
    try std.testing.expectEqual(@as(u64, 2), defaults.errors.get());
}

test "default collector convenience" {
    var bundle = try DefaultCollector.init(std.testing.allocator);
    defer bundle.deinit();

    recordRequest(&bundle.defaults, 10);
    recordError(&bundle.defaults, 20);
    try std.testing.expectEqual(@as(u64, 1), bundle.defaults.requests.get());
    try std.testing.expectEqual(@as(u64, 1), bundle.defaults.errors.get());
}

test "monitoring module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "circuit breaker metrics" {
    var collector = MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    var metrics = try CircuitBreakerMetrics.init(&collector);
    metrics.recordRequest(true, 10);
    metrics.recordRequest(false, 5);
    metrics.recordStateTransition();

    try std.testing.expectEqual(@as(u64, 2), metrics.requests_total.get());
    try std.testing.expectEqual(@as(u64, 1), metrics.requests_rejected.get());
    try std.testing.expectEqual(@as(u64, 1), metrics.state_transitions.get());
}

test "error metrics" {
    var collector = MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    var metrics = try ErrorMetrics.init(&collector);
    metrics.recordError(false);
    metrics.recordError(true);
    metrics.recordPattern();

    try std.testing.expectEqual(@as(u64, 2), metrics.errors_total.get());
    try std.testing.expectEqual(@as(u64, 1), metrics.errors_critical.get());
    try std.testing.expectEqual(@as(u64, 1), metrics.patterns_detected.get());
}

test "observability bundle basic" {
    var bundle = try ObservabilityBundle.init(std.testing.allocator, .{});
    defer bundle.deinit();

    try std.testing.expect(bundle.circuit_breaker != null);
    try std.testing.expect(bundle.errors != null);

    recordRequest(&bundle.defaults, 10);
    try std.testing.expectEqual(@as(u64, 1), bundle.defaults.requests.get());
}
