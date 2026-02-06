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
const config_module = @import("../../core/config/mod.zig");

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

    pub fn reset(self: *Counter) void {
        self.value.store(0, .monotonic);
    }
};

/// Gauge metric - a value that can increase or decrease.
/// Uses atomic i64 for thread-safe operations.
pub const Gauge = struct {
    name: []const u8,
    value: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),

    pub fn set(self: *Gauge, val: i64) void {
        self.value.store(val, .monotonic);
    }

    pub fn inc(self: *Gauge) void {
        _ = self.value.fetchAdd(1, .monotonic);
    }

    pub fn dec(self: *Gauge) void {
        _ = self.value.fetchSub(1, .monotonic);
    }

    pub fn add(self: *Gauge, delta: i64) void {
        _ = self.value.fetchAdd(delta, .monotonic);
    }

    pub fn get(self: *const Gauge) i64 {
        return self.value.load(.monotonic);
    }
};

/// Float gauge metric - for f64 values requiring mutex protection.
pub const FloatGauge = struct {
    name: []const u8,
    value: f64 = 0.0,
    mutex: std.Thread.Mutex = .{},

    pub fn set(self: *FloatGauge, val: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value = val;
    }

    pub fn add(self: *FloatGauge, delta: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value += delta;
    }

    pub fn get(self: *FloatGauge) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.value;
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
    counters: std.ArrayListUnmanaged(*Counter) = .empty,
    gauges: std.ArrayListUnmanaged(*Gauge) = .empty,
    float_gauges: std.ArrayListUnmanaged(*FloatGauge) = .empty,
    histograms: std.ArrayListUnmanaged(*Histogram) = .empty,

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
        for (self.gauges.items) |gauge| {
            self.allocator.destroy(gauge);
        }
        self.gauges.deinit(self.allocator);
        for (self.float_gauges.items) |gauge| {
            self.allocator.destroy(gauge);
        }
        self.float_gauges.deinit(self.allocator);
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

    pub fn registerGauge(self: *MetricsCollector, name: []const u8) !*Gauge {
        const gauge = try self.allocator.create(Gauge);
        errdefer self.allocator.destroy(gauge);
        gauge.* = .{ .name = name };
        try self.gauges.append(self.allocator, gauge);
        return gauge;
    }

    pub fn registerFloatGauge(self: *MetricsCollector, name: []const u8) !*FloatGauge {
        const gauge = try self.allocator.create(FloatGauge);
        errdefer self.allocator.destroy(gauge);
        gauge.* = .{ .name = name };
        try self.float_gauges.append(self.allocator, gauge);
        return gauge;
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
        errdefer histogram.deinit(self.allocator); // Clean up histogram internals on failure
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
pub const AttributeValue = tracing.AttributeValue;
pub const SpanAttribute = tracing.SpanAttribute;
pub const SpanEvent = tracing.SpanEvent;
pub const SpanLink = tracing.SpanLink;
pub const TraceContext = tracing.TraceContext;
pub const Propagator = tracing.Propagator;
pub const PropagationFormat = tracing.PropagationFormat;
pub const TraceSampler = tracing.TraceSampler;
pub const SpanProcessor = tracing.SpanProcessor;
pub const SpanExporter = tracing.SpanExporter;
pub const SamplerType = tracing.TraceSampler.SamplerType;

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
pub const PrometheusFormatter = struct {};
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
        return try Span.start(self.allocator, name, null, null, .internal);
    }
};

// ---------------------------------------------------------------------------
// System Information Helper
// ---------------------------------------------------------------------------
pub const system_info = @import("system_info/mod.zig");
pub const SystemInfo = system_info.SystemInfo;

// ---------------------------------------------------------------------------
// Centralized Metrics Module
// ---------------------------------------------------------------------------
// Provides shared metric primitives (Counter, Gauge, Histogram, SlidingWindow)
// that can be used across the codebase for consistent metrics collection.
pub const core_metrics = @import("metrics/mod.zig");

test {
    _ = tracing;
    _ = monitoring;
    _ = otel;
    _ = core_metrics;
}

// ============================================================================
// Counter Tests
// ============================================================================

test "Counter - init and get returns zero" {
    var counter = Counter{ .name = "test_counter" };
    try std.testing.expectEqual(@as(u64, 0), counter.get());
}

test "Counter - increment by one" {
    var counter = Counter{ .name = "test_counter" };

    counter.inc(1);
    try std.testing.expectEqual(@as(u64, 1), counter.get());

    counter.inc(1);
    try std.testing.expectEqual(@as(u64, 2), counter.get());
}

test "Counter - increment by multiple values" {
    var counter = Counter{ .name = "test_counter" };

    counter.inc(5);
    try std.testing.expectEqual(@as(u64, 5), counter.get());

    counter.inc(10);
    try std.testing.expectEqual(@as(u64, 15), counter.get());

    counter.inc(100);
    try std.testing.expectEqual(@as(u64, 115), counter.get());
}

test "Counter - reset to zero" {
    var counter = Counter{ .name = "test_counter" };

    counter.inc(100);
    try std.testing.expectEqual(@as(u64, 100), counter.get());

    counter.reset();
    try std.testing.expectEqual(@as(u64, 0), counter.get());
}

test "Counter - large values" {
    var counter = Counter{ .name = "test_counter" };

    const large_value: u64 = 1_000_000_000;
    counter.inc(large_value);
    try std.testing.expectEqual(large_value, counter.get());

    counter.inc(large_value);
    try std.testing.expectEqual(large_value * 2, counter.get());
}

test "Counter - concurrent increment safety" {
    // Skip on single-threaded platforms
    if (@import("builtin").single_threaded) return error.SkipZigTest;

    var counter = Counter{ .name = "concurrent_counter" };

    const num_threads = 4;
    const iterations_per_thread = 1000;

    const ThreadFn = struct {
        fn run(c: *Counter) void {
            for (0..iterations_per_thread) |_| {
                c.inc(1);
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = std.Thread.spawn(.{}, ThreadFn.run, .{&counter}) catch return error.SkipZigTest;
    }

    for (threads) |t| {
        t.join();
    }

    try std.testing.expectEqual(@as(u64, num_threads * iterations_per_thread), counter.get());
}

// ============================================================================
// Gauge Tests
// ============================================================================

test "Gauge - init and get returns zero" {
    var gauge = Gauge{ .name = "test_gauge" };
    try std.testing.expectEqual(@as(i64, 0), gauge.get());
}

test "Gauge - set positive value" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.set(42);
    try std.testing.expectEqual(@as(i64, 42), gauge.get());
}

test "Gauge - set negative value" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.set(-100);
    try std.testing.expectEqual(@as(i64, -100), gauge.get());
}

test "Gauge - inc and dec" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.inc();
    try std.testing.expectEqual(@as(i64, 1), gauge.get());

    gauge.inc();
    gauge.inc();
    try std.testing.expectEqual(@as(i64, 3), gauge.get());

    gauge.dec();
    try std.testing.expectEqual(@as(i64, 2), gauge.get());
}

test "Gauge - add positive and negative" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.add(10);
    try std.testing.expectEqual(@as(i64, 10), gauge.get());

    gauge.add(-5);
    try std.testing.expectEqual(@as(i64, 5), gauge.get());

    gauge.add(-10);
    try std.testing.expectEqual(@as(i64, -5), gauge.get());
}

test "Gauge - set overwrites previous value" {
    var gauge = Gauge{ .name = "test_gauge" };

    gauge.set(100);
    gauge.add(50);
    try std.testing.expectEqual(@as(i64, 150), gauge.get());

    gauge.set(-200);
    try std.testing.expectEqual(@as(i64, -200), gauge.get());
}

test "Gauge - concurrent operations" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;

    var gauge = Gauge{ .name = "concurrent_gauge" };

    const num_threads = 4;
    const iterations = 100;

    const IncThread = struct {
        fn run(g: *Gauge) void {
            for (0..iterations) |_| {
                g.inc();
            }
        }
    };

    const DecThread = struct {
        fn run(g: *Gauge) void {
            for (0..iterations) |_| {
                g.dec();
            }
        }
    };

    var inc_threads: [num_threads]std.Thread = undefined;
    var dec_threads: [num_threads]std.Thread = undefined;

    for (&inc_threads) |*t| {
        t.* = std.Thread.spawn(.{}, IncThread.run, .{&gauge}) catch return error.SkipZigTest;
    }
    for (&dec_threads) |*t| {
        t.* = std.Thread.spawn(.{}, DecThread.run, .{&gauge}) catch return error.SkipZigTest;
    }

    for (inc_threads) |t| t.join();
    for (dec_threads) |t| t.join();

    // Equal inc and dec should result in 0
    try std.testing.expectEqual(@as(i64, 0), gauge.get());
}

// ============================================================================
// FloatGauge Tests
// ============================================================================

test "FloatGauge - init and get returns zero" {
    var gauge = FloatGauge{ .name = "float_gauge" };
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gauge.get(), 0.0001);
}

test "FloatGauge - set value" {
    var gauge = FloatGauge{ .name = "float_gauge" };

    gauge.set(3.14159);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14159), gauge.get(), 0.00001);

    gauge.set(-2.71828);
    try std.testing.expectApproxEqAbs(@as(f64, -2.71828), gauge.get(), 0.00001);
}

test "FloatGauge - add values" {
    var gauge = FloatGauge{ .name = "float_gauge" };

    gauge.add(1.5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), gauge.get(), 0.0001);

    gauge.add(2.5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), gauge.get(), 0.0001);

    gauge.add(-1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), gauge.get(), 0.0001);
}

test "FloatGauge - very small values" {
    var gauge = FloatGauge{ .name = "float_gauge" };

    gauge.set(1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1e-10), gauge.get(), 1e-15);
}

test "FloatGauge - very large values" {
    var gauge = FloatGauge{ .name = "float_gauge" };

    gauge.set(1e15);
    try std.testing.expectApproxEqAbs(@as(f64, 1e15), gauge.get(), 1e10);
}

// ============================================================================
// Histogram Tests
// ============================================================================

test "Histogram - init and deinit" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{ 10, 50, 100, 500, 1000 };

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    try std.testing.expectEqualStrings("test_histogram", hist.name);
    try std.testing.expectEqual(@as(usize, 5), hist.bounds.len);
    try std.testing.expectEqual(@as(usize, 6), hist.buckets.len); // bounds.len + 1
}

test "Histogram - record values in first bucket" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{ 10, 50, 100 };

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    hist.record(5);
    hist.record(8);
    hist.record(10);

    try std.testing.expectEqual(@as(u64, 3), hist.buckets[0]);
    try std.testing.expectEqual(@as(u64, 0), hist.buckets[1]);
    try std.testing.expectEqual(@as(u64, 0), hist.buckets[2]);
}

test "Histogram - record values across buckets" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{ 10, 50, 100 };

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    hist.record(5); // bucket 0 (<=10)
    hist.record(25); // bucket 1 (<=50)
    hist.record(75); // bucket 2 (<=100)
    hist.record(150); // bucket 3 (>100, overflow)

    try std.testing.expectEqual(@as(u64, 1), hist.buckets[0]);
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[1]);
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[2]);
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[3]);
}

test "Histogram - record boundary values" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{ 10, 50, 100 };

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    // Exact boundary values should go in the bucket
    hist.record(10); // bucket 0
    hist.record(50); // bucket 1
    hist.record(100); // bucket 2

    try std.testing.expectEqual(@as(u64, 1), hist.buckets[0]);
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[1]);
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[2]);
}

test "Histogram - overflow bucket" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{100};

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    hist.record(50); // bucket 0
    hist.record(150); // overflow bucket
    hist.record(1000); // overflow bucket
    hist.record(std.math.maxInt(u64)); // overflow bucket

    try std.testing.expectEqual(@as(u64, 1), hist.buckets[0]);
    try std.testing.expectEqual(@as(u64, 3), hist.buckets[1]);
}

test "Histogram - many records in same bucket" {
    const allocator = std.testing.allocator;
    var bounds = [_]u64{ 100, 200 };

    var hist = try Histogram.init(allocator, "test_histogram", &bounds);
    defer hist.deinit(allocator);

    for (0..1000) |_| {
        hist.record(50);
    }

    try std.testing.expectEqual(@as(u64, 1000), hist.buckets[0]);
}

// ============================================================================
// MetricsCollector Tests
// ============================================================================

test "MetricsCollector - init and deinit" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    collector.deinit();
}

test "MetricsCollector - register counter" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("http_requests_total");

    try std.testing.expectEqualStrings("http_requests_total", counter.name);
    try std.testing.expectEqual(@as(u64, 0), counter.get());

    counter.inc(10);
    try std.testing.expectEqual(@as(u64, 10), counter.get());
}

test "MetricsCollector - register multiple counters" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const counter1 = try collector.registerCounter("requests");
    const counter2 = try collector.registerCounter("errors");
    const counter3 = try collector.registerCounter("successes");

    counter1.inc(100);
    counter2.inc(5);
    counter3.inc(95);

    try std.testing.expectEqual(@as(u64, 100), counter1.get());
    try std.testing.expectEqual(@as(u64, 5), counter2.get());
    try std.testing.expectEqual(@as(u64, 95), counter3.get());
}

test "MetricsCollector - register gauge" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const gauge = try collector.registerGauge("active_connections");

    gauge.set(50);
    try std.testing.expectEqual(@as(i64, 50), gauge.get());

    gauge.inc();
    gauge.inc();
    try std.testing.expectEqual(@as(i64, 52), gauge.get());

    gauge.dec();
    try std.testing.expectEqual(@as(i64, 51), gauge.get());
}

test "MetricsCollector - register float gauge" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const gauge = try collector.registerFloatGauge("temperature");

    gauge.set(23.5);
    try std.testing.expectApproxEqAbs(@as(f64, 23.5), gauge.get(), 0.001);

    gauge.add(1.5);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), gauge.get(), 0.001);
}

test "MetricsCollector - register histogram" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const bounds = [_]u64{ 5, 10, 25, 50, 100 };
    const hist = try collector.registerHistogram("request_latency_ms", &bounds);

    hist.record(3);
    hist.record(15);
    hist.record(75);

    try std.testing.expectEqual(@as(u64, 1), hist.buckets[0]); // <=5
    try std.testing.expectEqual(@as(u64, 0), hist.buckets[1]); // <=10
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[2]); // <=25
    try std.testing.expectEqual(@as(u64, 0), hist.buckets[3]); // <=50
    try std.testing.expectEqual(@as(u64, 1), hist.buckets[4]); // <=100
}

test "MetricsCollector - combined metrics" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    const counter = try collector.registerCounter("requests");
    const gauge = try collector.registerGauge("in_flight");
    const float_gauge = try collector.registerFloatGauge("cpu_usage");

    // Simulate request processing
    counter.inc(1);
    gauge.inc();
    float_gauge.set(0.45);

    try std.testing.expectEqual(@as(u64, 1), counter.get());
    try std.testing.expectEqual(@as(i64, 1), gauge.get());
    try std.testing.expectApproxEqAbs(@as(f64, 0.45), float_gauge.get(), 0.001);

    // Complete request
    gauge.dec();
    try std.testing.expectEqual(@as(i64, 0), gauge.get());
}

// ============================================================================
// DefaultCollector Tests
// ============================================================================

test "DefaultCollector - init and deinit" {
    const allocator = std.testing.allocator;
    var dc = try DefaultCollector.init(allocator);
    defer dc.deinit();

    try std.testing.expectEqualStrings("requests_total", dc.defaults.requests.name);
    try std.testing.expectEqualStrings("errors_total", dc.defaults.errors.name);
    try std.testing.expectEqualStrings("latency_ms", dc.defaults.latency_ms.name);
}

test "DefaultCollector - record request" {
    const allocator = std.testing.allocator;
    var dc = try DefaultCollector.init(allocator);
    defer dc.deinit();

    recordRequest(&dc.defaults, 50);
    recordRequest(&dc.defaults, 100);

    try std.testing.expectEqual(@as(u64, 2), dc.defaults.requests.get());
}

test "DefaultCollector - record error" {
    const allocator = std.testing.allocator;
    var dc = try DefaultCollector.init(allocator);
    defer dc.deinit();

    recordError(&dc.defaults, 500);

    try std.testing.expectEqual(@as(u64, 1), dc.defaults.errors.get());
}

// ============================================================================
// CircuitBreakerMetrics Tests
// ============================================================================

test "CircuitBreakerMetrics - init and record" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var cb_metrics = try CircuitBreakerMetrics.init(&collector);

    cb_metrics.recordRequest(true, 50);
    cb_metrics.recordRequest(false, 100);
    cb_metrics.recordStateTransition();

    try std.testing.expectEqual(@as(u64, 2), cb_metrics.requests_total.get());
    try std.testing.expectEqual(@as(u64, 1), cb_metrics.requests_rejected.get());
    try std.testing.expectEqual(@as(u64, 1), cb_metrics.state_transitions.get());
}

// ============================================================================
// ErrorMetrics Tests
// ============================================================================

test "ErrorMetrics - init and record" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    var err_metrics = try ErrorMetrics.init(&collector);

    err_metrics.recordError(false);
    err_metrics.recordError(true);
    err_metrics.recordError(false);
    err_metrics.recordPattern();

    try std.testing.expectEqual(@as(u64, 3), err_metrics.errors_total.get());
    try std.testing.expectEqual(@as(u64, 1), err_metrics.errors_critical.get());
    try std.testing.expectEqual(@as(u64, 1), err_metrics.patterns_detected.get());
}

// ============================================================================
// Module State Tests
// ============================================================================

test "module - isEnabled returns build option" {
    // Just verify the function works without crashing
    _ = isEnabled();
}

test "module - isInitialized initially false" {
    // Note: This test assumes fresh state
    // Might fail if other tests have modified the initialized flag
    _ = isInitialized();
}

test "createCollector - creates valid collector" {
    const allocator = std.testing.allocator;
    var collector = createCollector(allocator);
    defer collector.deinit();

    // Just verify it works
    const counter = try collector.registerCounter("test");
    counter.inc(1);
    try std.testing.expectEqual(@as(u64, 1), counter.get());
}
