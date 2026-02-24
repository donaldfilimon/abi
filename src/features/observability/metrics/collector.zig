//! Metrics Collector
//!
//! Named metric registration and lifecycle management. These types carry
//! a `name` field for identification within the collector, unlike the
//! lightweight primitives in `services/shared/utils/metric_types.zig`.

const std = @import("std");
const sync = @import("../../../services/shared/sync.zig");
const Mutex = sync.Mutex;

// ============================================================================
// Named Metric Primitives
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

pub const FloatGauge = struct {
    name: []const u8,
    value: f64 = 0.0,
    mutex: Mutex = .{},

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
    mutex: Mutex = .{},

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
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.bounds, 0..) |bound, i| {
            if (value <= bound) {
                self.buckets[i] += 1;
                return;
            }
        }
        self.buckets[self.buckets.len - 1] += 1;
    }
};

// ============================================================================
// Collector
// ============================================================================

pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,
    counters: std.ArrayListUnmanaged(*Counter) = .empty,
    gauges: std.ArrayListUnmanaged(*Gauge) = .empty,
    float_gauges: std.ArrayListUnmanaged(*FloatGauge) = .empty,
    histograms: std.ArrayListUnmanaged(*Histogram) = .empty,

    pub fn init(allocator: std.mem.Allocator) MetricsCollector {
        return .{ .allocator = allocator };
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
        errdefer histogram.deinit(self.allocator);
        self.allocator.free(bounds_copy);
        try self.histograms.append(self.allocator, histogram);
        return histogram;
    }
};

// ============================================================================
// Default Metrics & Helpers
// ============================================================================

pub const DEFAULT_LATENCY_BOUNDS = [_]u64{ 1, 5, 10, 25, 50, 100, 250, 500, 1000 };

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
// Circuit Breaker Metrics
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

// ============================================================================
// Error Metrics
// ============================================================================

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

test {
    std.testing.refAllDecls(@This());
}
