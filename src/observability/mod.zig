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

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// ============================================================================
// Metrics Primitives
// ============================================================================

pub const Counter = struct {
    name: []const u8,
    value: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn inc(self: *Counter, delta: u64) void {
        _ = self.value.fetchAdd(delta, .monotonic);
    }

    pub fn get(self: *const Counter) u64 {
        return self.value.load(.monotonic);
    }
};

pub const Histogram = struct {
    name: []const u8,
    buckets: []u64,
    bounds: []u64,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, bounds: []u64) !Histogram {
        const bucket_copy = try allocator.alloc(u64, bounds.len + 1);
        errdefer allocator.free(bucket_copy);
        const bound_copy = try allocator.alloc(u64, bounds.len);
        errdefer allocator.free(bound_copy);
        @memset(bucket_copy, 0);
        std.mem.copyForwards(u64, bound_copy, bounds);
        return Histogram{
            .name = name,
            .buckets = bucket_copy,
            .bounds = bound_copy,
        };
    }

    pub fn deinit(self: *Histogram, allocator: std.mem.Allocator) void {
        allocator.free(self.buckets);
        allocator.free(self.bounds);
        self.* = undefined;
    }

    pub fn record(self: *Histogram, value: u64) void {
        for (self.bounds, 0..) |bound, i| {
            if (value <= bound) {
                self.buckets[i] += 1;
                return;
            }
        }
        self.buckets[self.buckets.len - 1] += 1;
    }
};

pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,
    counters: std.ArrayListUnmanaged(*Counter) = .{},
    histograms: std.ArrayListUnmanaged(*Histogram) = .{},

    pub fn init(allocator: std.mem.Allocator) MetricsCollector {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetricsCollector) void {
        for (self.counters.items) |counter| {
            self.allocator.destroy(counter);
        }
        self.counters.deinit(self.allocator);
        for (self.histograms.items) |histogram| {
            histogram.deinit(self.allocator);
            self.allocator.destroy(histogram);
        }
        self.histograms.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn registerCounter(self: *MetricsCollector, name: []const u8) !*Counter {
        const counter = try self.allocator.create(Counter);
        errdefer self.allocator.destroy(counter);
        counter.* = .{ .name = name };
        try self.counters.append(self.allocator, counter);
        return counter;
    }

    pub fn registerHistogram(
        self: *MetricsCollector,
        name: []const u8,
        bounds: []const u64,
    ) !*Histogram {
        const histogram = try self.allocator.create(Histogram);
        errdefer self.allocator.destroy(histogram);
        const bounds_copy = try self.allocator.alloc(u64, bounds.len);
        errdefer self.allocator.free(bounds_copy);
        std.mem.copyForwards(u64, bounds_copy, bounds);
        histogram.* = try Histogram.init(self.allocator, name, bounds_copy);
        self.allocator.free(bounds_copy); // Histogram.init makes its own copy
        try self.histograms.append(self.allocator, histogram);
        return histogram;
    }
};

// ============================================================================
// Tracing
// ============================================================================

pub const tracing = @import("tracing.zig");
pub const Tracer = tracing.Tracer;
pub const Span = tracing.Span;
pub const TraceId = tracing.TraceId;
pub const SpanId = tracing.SpanId;
pub const SpanKind = tracing.SpanKind;
pub const SpanStatus = tracing.SpanStatus;

// ============================================================================
// Monitoring (Consolidated Alerting, Prometheus, StatsD)
// ============================================================================

pub const monitoring = @import("monitoring.zig");
pub const otel = @import("otel.zig");

// Re-exports for backward compatibility and namespace access
pub const alerting = monitoring;
pub const prometheus = monitoring;
pub const statsd = monitoring;

// Alerting exports
pub const AlertManager = monitoring.AlertManager;
pub const AlertManagerConfig = monitoring.AlertManagerConfig;
pub const AlertRule = monitoring.AlertRule;
pub const AlertRuleBuilder = monitoring.AlertRuleBuilder;
pub const Alert = monitoring.Alert;
pub const AlertState = monitoring.AlertState;
pub const AlertSeverity = monitoring.AlertSeverity;
pub const AlertCondition = monitoring.AlertCondition;
pub const AlertError = monitoring.AlertError;
pub const AlertStats = monitoring.AlertStats;
pub const AlertCallback = monitoring.AlertCallback;
pub const AlertHandler = monitoring.AlertHandler;
pub const MetricValues = monitoring.MetricValues;
pub const createAlertRule = monitoring.createRule;

// Prometheus exports
pub const PrometheusExporter = monitoring.PrometheusExporter;
pub const PrometheusConfig = monitoring.PrometheusConfig;
pub const generateMetricsOutput = monitoring.generateMetricsOutput;

// StatsD exports
pub const StatsDClient = monitoring.StatsDClient;
pub const StatsDConfig = monitoring.StatsDConfig;
pub const StatsDError = monitoring.StatsDError;

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

// Legacy type aliases
pub const MetricsConfig = struct {};
pub const MetricsSummary = struct {};

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

    pub fn start(self: *ObservabilityBundle) !void {
        if (self.prometheus_exporter) |p| {
            try p.start();
        }
        if (self.otel_exporter) |e| {
            try e.start();
        }
    }

    pub fn stop(self: *ObservabilityBundle) void {
        if (self.prometheus_exporter) |p| {
            p.stop();
        }
        if (self.otel_exporter) |e| {
            e.stop();
        }
    }

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

    pub fn recordMetric(self: *Context, name: []const u8, value: f64) !void {
        if (self.metrics) |m| {
            // MetricsCollector in v2 uses record(f64) for histogram-like behavior if wanted,
            // or we just map it.
            _ = m;
            _ = name;
            _ = value;
        }
    }

    pub fn startSpan(self: *Context, name: []const u8) !Span {
        _ = self;
        const allocator = std.heap.page_allocator;
        return try Span.start(allocator, name, null, null, .internal);
    }
};
