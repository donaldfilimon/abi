//! Observability Bundle
//!
//! Unified container for metrics collection, Prometheus export, OpenTelemetry
//! tracing, circuit-breaker metrics, and error metrics. Create one bundle per
//! application and pass it around for instrumentation.

const std = @import("std");

const collector_mod = @import("metrics/collector.zig");
const monitoring_mod = @import("monitoring.zig");
const otel_mod = @import("otel.zig");

const MetricsCollector = collector_mod.MetricsCollector;
const DefaultMetrics = collector_mod.DefaultMetrics;
const CircuitBreakerMetrics = collector_mod.CircuitBreakerMetrics;
const ErrorMetrics = collector_mod.ErrorMetrics;
const registerDefaultMetrics = collector_mod.registerDefaultMetrics;

const PrometheusExporter = monitoring_mod.PrometheusExporter;
const PrometheusConfig = monitoring_mod.PrometheusConfig;

const OtelExporter = otel_mod.OtelExporter;
const OtelConfig = otel_mod.OtelConfig;
const OtelTracer = otel_mod.OtelTracer;
const OtelSpan = otel_mod.OtelSpan;

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
