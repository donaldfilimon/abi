//! Monitoring feature facade for metrics collection.
const std = @import("std");
const build_options = @import("build_options");

const observability = @import("../../shared/observability/mod.zig");

pub const MetricsCollector = observability.MetricsCollector;
pub const Counter = observability.Counter;
pub const Histogram = observability.Histogram;

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
