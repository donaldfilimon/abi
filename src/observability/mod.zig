//! Observability Module
//!
//! Unified observability with metrics, tracing, and profiling.
//!
//! ## Features
//! - Metrics collection and export
//! - Distributed tracing
//! - Performance profiling
//! - Circuit breakers

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/monitoring and shared/observability
const features_monitoring = @import("../features/monitoring/mod.zig");
const shared_observability = @import("../shared/observability/mod.zig");

pub const MetricsCollector = features_monitoring.MetricsCollector;
pub const MetricsConfig = features_monitoring.MetricsConfig;
pub const MetricsSummary = features_monitoring.MetricsSummary;
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
