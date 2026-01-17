//! Stub for Monitoring feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.MonitoringDisabled for all operations.

const std = @import("std");
const stub_root = @This();

pub const MonitoringError = error{
    MonitoringDisabled,
};

// Top-level type definitions
pub const AlertManager = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: AlertManagerConfig) @This() {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn addRule(self: *@This(), rule: AlertRule) AlertError!void {
        _ = self;
        _ = rule;
        return error.MonitoringDisabled;
    }

    pub fn removeRule(self: *@This(), name: []const u8) bool {
        _ = self;
        _ = name;
        return false;
    }

    pub fn addHandler(self: *@This(), handler: AlertHandler) AlertError!void {
        _ = self;
        _ = handler;
        return error.MonitoringDisabled;
    }

    pub fn evaluate(self: *@This(), metrics: MetricValues) AlertError!void {
        _ = self;
        _ = metrics;
        return error.MonitoringDisabled;
    }

    pub fn getStats(self: *@This()) AlertStats {
        _ = self;
        return .{};
    }

    pub fn getActiveAlerts(self: *@This(), allocator: std.mem.Allocator) AlertError![]Alert {
        _ = self;
        _ = allocator;
        return error.MonitoringDisabled;
    }
};

pub const AlertManagerConfig = struct {
    evaluation_interval_ms: u64 = 15_000,
    default_for_duration_ms: u64 = 60_000,
    max_alerts: u32 = 1000,
};

pub const AlertRule = struct {
    name: []const u8,
    metric: []const u8,
    condition: AlertCondition = .greater_than,
    threshold: f64 = 0.0,
    severity: AlertSeverity = .warning,
    for_duration_ms: u64 = 0,
    labels: ?std.StringHashMap([]const u8) = null,
    annotations: ?std.StringHashMap([]const u8) = null,
};

pub const AlertRuleBuilder = struct {
    rule: AlertRule,

    pub fn init(name: []const u8) @This() {
        return .{ .rule = .{ .name = name, .metric = "" } };
    }

    pub fn metric(self: *@This(), m: []const u8) *@This() {
        self.rule.metric = m;
        return self;
    }

    pub fn condition(self: *@This(), c: AlertCondition) *@This() {
        self.rule.condition = c;
        return self;
    }

    pub fn threshold(self: *@This(), t: f64) *@This() {
        self.rule.threshold = t;
        return self;
    }

    pub fn severity(self: *@This(), s: AlertSeverity) *@This() {
        self.rule.severity = s;
        return self;
    }

    pub fn forDuration(self: *@This(), ms: u64) *@This() {
        self.rule.for_duration_ms = ms;
        return self;
    }

    pub fn build(self: *@This()) AlertRule {
        return self.rule;
    }
};

pub const Alert = struct {
    rule_name: []const u8 = "",
    state: AlertState = .inactive,
    severity: AlertSeverity = .info,
    fired_at: ?i64 = null,
    resolved_at: ?i64 = null,
    value: f64 = 0.0,
};

pub const AlertState = enum {
    inactive,
    pending,
    firing,
    resolved,
};

pub const AlertSeverity = enum {
    info,
    warning,
    critical,
};

pub const AlertCondition = enum {
    greater_than,
    less_than,
    equal_to,
    not_equal_to,
    greater_than_or_equal,
    less_than_or_equal,
};

pub const AlertError = error{
    MonitoringDisabled,
    RuleNotFound,
    InvalidRule,
    OutOfMemory,
};

pub const AlertStats = struct {
    rules_count: u32 = 0,
    active_alerts: u32 = 0,
    total_evaluations: u64 = 0,
    total_firings: u64 = 0,
};

pub const AlertCallback = *const fn (alert: Alert) void;

pub const AlertHandler = struct {
    callback: AlertCallback,
    min_severity: AlertSeverity = .info,
};

pub const MetricValues = std.StringHashMap(f64);

pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn registerCounter(self: *@This(), name: []const u8) MonitoringError!*Counter {
        _ = self;
        _ = name;
        return error.MonitoringDisabled;
    }

    pub fn registerHistogram(self: *@This(), name: []const u8, bounds: []const u64) MonitoringError!*Histogram {
        _ = self;
        _ = name;
        _ = bounds;
        return error.MonitoringDisabled;
    }
};

pub const Counter = struct {
    value: u64 = 0,

    pub fn inc(self: *@This(), delta: u64) void {
        _ = self;
        _ = delta;
    }

    pub fn get(self: *const @This()) u64 {
        _ = self;
        return 0;
    }
};

pub const Histogram = struct {
    pub fn record(self: *@This(), value: u64) void {
        _ = self;
        _ = value;
    }
};

pub const PrometheusConfig = struct {
    enabled: bool = true,
    path: []const u8 = "/metrics",
    port: u16 = 9090,
    namespace: []const u8 = "abi",
    include_timestamp: bool = false,
};

pub const PrometheusExporter = struct {
    allocator: std.mem.Allocator,
    config: PrometheusConfig,
    metrics: *MetricsCollector,

    pub fn init(
        allocator: std.mem.Allocator,
        config: PrometheusConfig,
        metrics: *MetricsCollector,
    ) MonitoringError!PrometheusExporter {
        _ = allocator;
        _ = config;
        _ = metrics;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *PrometheusExporter) void {
        _ = self;
    }

    pub fn start(self: *PrometheusExporter) MonitoringError!void {
        _ = self;
        return error.MonitoringDisabled;
    }

    pub fn stop(self: *PrometheusExporter) void {
        _ = self;
    }

    pub fn generateMetrics(self: *PrometheusExporter, allocator: std.mem.Allocator) MonitoringError![]const u8 {
        _ = self;
        _ = allocator;
        return error.MonitoringDisabled;
    }
};

pub const PrometheusFormatter = struct {
    allocator: std.mem.Allocator,
    namespace: []const u8,

    pub fn init(allocator: std.mem.Allocator, namespace: []const u8) PrometheusFormatter {
        return .{ .allocator = allocator, .namespace = namespace };
    }

    pub fn deinit(self: *PrometheusFormatter) void {
        _ = self;
    }

    pub fn formatMetrics(self: *PrometheusFormatter, collector: *MetricsCollector) MonitoringError![]const u8 {
        _ = self;
        _ = collector;
        return error.MonitoringDisabled;
    }
};

pub fn generateMetricsOutput(
    allocator: std.mem.Allocator,
    collector: *MetricsCollector,
    config: PrometheusConfig,
) MonitoringError![]const u8 {
    _ = allocator;
    _ = collector;
    _ = config;
    return error.MonitoringDisabled;
}

pub const OtelConfig = struct {
    service_name: []const u8 = "abi",
    endpoint: []const u8 = "http://localhost:4318",
    api_key: ?[]const u8 = null,
    sample_rate: f64 = 1.0,
};

pub const OtelExporter = struct {
    allocator: std.mem.Allocator,
    config: OtelConfig,

    pub fn init(allocator: std.mem.Allocator, config: OtelConfig) MonitoringError!OtelExporter {
        _ = allocator;
        _ = config;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *OtelExporter) void {
        _ = self;
    }

    pub fn start(self: *OtelExporter) MonitoringError!void {
        _ = self;
        return error.MonitoringDisabled;
    }

    pub fn stop(self: *OtelExporter) void {
        _ = self;
    }
};

pub const OtelSpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const OtelStatus = enum {
    unset,
    ok,
    @"error",
};

pub const OtelAttributeValue = union(enum) {
    string: []const u8,
    int: i64,
    float: f64,
    bool: bool,
};

pub const OtelAttribute = struct {
    key: []const u8,
    value: OtelAttributeValue,
};

pub const OtelEvent = struct {
    name: []const u8,
    timestamp_ns: u64 = 0,
    attributes: []const OtelAttribute = &.{},
};

pub const OtelSpan = struct {
    name: []const u8 = "",
    kind: OtelSpanKind = .internal,
    status: OtelStatus = .unset,
    attributes: []const OtelAttribute = &.{},
    events: []const OtelEvent = &.{},

    pub fn end(self: *OtelSpan) void {
        _ = self;
    }
};

pub const OtelMetricType = enum {
    counter,
    gauge,
    histogram,
};

pub const OtelMetric = struct {
    name: []const u8,
    metric_type: OtelMetricType,
    value: f64,
};

pub const OtelTracer = struct {
    allocator: std.mem.Allocator,
    service_name: []const u8,

    pub fn init(allocator: std.mem.Allocator, service_name: []const u8) MonitoringError!OtelTracer {
        _ = allocator;
        _ = service_name;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *OtelTracer) void {
        _ = self;
    }

    pub fn startSpan(
        self: *OtelTracer,
        name: []const u8,
        parent: ?OtelSpan,
        kind: ?OtelSpanKind,
    ) MonitoringError!OtelSpan {
        _ = self;
        _ = name;
        _ = parent;
        _ = kind;
        return error.MonitoringDisabled;
    }
};

pub const OtelContext = struct {
    trace_id: [16]u8 = .{0} ** 16,
    span_id: [8]u8 = .{0} ** 8,
};

pub fn formatTraceId(_: [16]u8) [32]u8 {
    return .{0} ** 32;
}

pub fn formatSpanId(_: [8]u8) [16]u8 {
    return .{0} ** 16;
}

pub fn createOtelResource(_: OtelConfig) []const OtelAttribute {
    return &.{};
}

pub const DefaultMetrics = struct {
    requests: *Counter,
    errors: *Counter,
    latency_ms: *Histogram,
};

pub const DefaultCollector = struct {
    collector: MetricsCollector,
    defaults: DefaultMetrics,

    pub fn init(allocator: std.mem.Allocator) MonitoringError!@This() {
        _ = allocator;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const CircuitBreakerMetrics = struct {
    requests_total: *Counter,
    requests_rejected: *Counter,
    state_transitions: *Counter,
    latency_ms: *Histogram,

    pub fn init(collector: *MetricsCollector) MonitoringError!CircuitBreakerMetrics {
        _ = collector;
        return error.MonitoringDisabled;
    }

    pub fn recordRequest(self: *CircuitBreakerMetrics, success: bool, latency_ms: u64) void {
        _ = self;
        _ = success;
        _ = latency_ms;
    }

    pub fn recordStateTransition(self: *CircuitBreakerMetrics) void {
        _ = self;
    }
};

pub const ErrorMetrics = struct {
    errors_total: *Counter,
    errors_critical: *Counter,
    patterns_detected: *Counter,

    pub fn init(collector: *MetricsCollector) MonitoringError!ErrorMetrics {
        _ = collector;
        return error.MonitoringDisabled;
    }

    pub fn recordError(self: *ErrorMetrics, is_critical: bool) void {
        _ = self;
        _ = is_critical;
    }

    pub fn recordPattern(self: *ErrorMetrics) void {
        _ = self;
    }
};

pub const BundleConfig = struct {
    enable_circuit_breaker_metrics: bool = true,
    enable_error_metrics: bool = true,
    prometheus: ?PrometheusConfig = null,
    otel: ?OtelConfig = null,
};

pub const ObservabilityBundle = struct {
    allocator: std.mem.Allocator,
    collector: MetricsCollector,
    defaults: DefaultMetrics,
    circuit_breaker: ?CircuitBreakerMetrics,
    errors: ?ErrorMetrics,
    prometheus: ?*PrometheusExporter,
    otel_exporter: ?*OtelExporter,
    tracer: ?*OtelTracer,

    pub fn init(allocator: std.mem.Allocator, config: BundleConfig) MonitoringError!ObservabilityBundle {
        _ = allocator;
        _ = config;
        return error.MonitoringDisabled;
    }

    pub fn deinit(self: *ObservabilityBundle) void {
        _ = self;
    }

    pub fn start(self: *ObservabilityBundle) MonitoringError!void {
        _ = self;
        return error.MonitoringDisabled;
    }

    pub fn stop(self: *ObservabilityBundle) void {
        _ = self;
    }

    pub fn startSpan(self: *ObservabilityBundle, name: []const u8) MonitoringError!?OtelSpan {
        _ = self;
        _ = name;
        return error.MonitoringDisabled;
    }
};

// Top-level function for creating alert rules
pub fn createAlertRule(name: []const u8) AlertRuleBuilder {
    return AlertRuleBuilder.init(name);
}

// Sub-module namespace (alerting)
pub const alerting = struct {
    pub const AlertManager = stub_root.AlertManager;
    pub const AlertManagerConfig = stub_root.AlertManagerConfig;
    pub const AlertRule = stub_root.AlertRule;
    pub const AlertRuleBuilder = stub_root.AlertRuleBuilder;
    pub const Alert = stub_root.Alert;
    pub const AlertState = stub_root.AlertState;
    pub const AlertSeverity = stub_root.AlertSeverity;
    pub const AlertCondition = stub_root.AlertCondition;
    pub const AlertError = stub_root.AlertError;
    pub const AlertStats = stub_root.AlertStats;
    pub const AlertCallback = stub_root.AlertCallback;
    pub const AlertHandler = stub_root.AlertHandler;
    pub const MetricValues = stub_root.MetricValues;

    pub const createRule = stub_root.createAlertRule;
};

pub const prometheus = struct {
    pub const PrometheusExporter = stub_root.PrometheusExporter;
    pub const PrometheusConfig = stub_root.PrometheusConfig;
    pub const PrometheusFormatter = stub_root.PrometheusFormatter;
    pub const generateMetricsOutput = stub_root.generateMetricsOutput;
};

pub const otel = struct {
    pub const OtelExporter = stub_root.OtelExporter;
    pub const OtelConfig = stub_root.OtelConfig;
    pub const OtelTracer = stub_root.OtelTracer;
    pub const OtelSpan = stub_root.OtelSpan;
    pub const OtelSpanKind = stub_root.OtelSpanKind;
    pub const OtelContext = stub_root.OtelContext;
    pub const OtelMetric = stub_root.OtelMetric;
    pub const OtelMetricType = stub_root.OtelMetricType;
    pub const OtelAttribute = stub_root.OtelAttribute;
    pub const OtelAttributeValue = stub_root.OtelAttributeValue;
    pub const OtelEvent = stub_root.OtelEvent;
    pub const OtelStatus = stub_root.OtelStatus;
    pub const formatTraceId = stub_root.formatTraceId;
    pub const formatSpanId = stub_root.formatSpanId;
    pub const createOtelResource = stub_root.createOtelResource;
};

pub const statsd = struct {};

// Module lifecycle
var initialized: bool = false;

pub fn init(_: std.mem.Allocator) MonitoringError!void {
    return error.MonitoringDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

// Convenience functions
pub fn createCollector(allocator: std.mem.Allocator) MetricsCollector {
    return MetricsCollector.init(allocator);
}

pub fn registerDefaultMetrics(collector: *MetricsCollector) MonitoringError!DefaultMetrics {
    _ = collector;
    return error.MonitoringDisabled;
}

pub fn recordRequest(metrics: *DefaultMetrics, latency_ms: u64) void {
    _ = metrics;
    _ = latency_ms;
}

pub fn recordError(metrics: *DefaultMetrics, latency_ms: u64) void {
    _ = metrics;
    _ = latency_ms;
}
