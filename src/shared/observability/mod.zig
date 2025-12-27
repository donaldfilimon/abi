//! Observability primitives for metrics collection and monitoring.
//!
//! Provides thread-safe counters and histograms for collecting performance metrics.
//! Designed for high-throughput scenarios with minimal overhead.
//!
//! Example:
//! ```zig
//! var histogram = try observability.Histogram.init(
//!     allocator,
//!     "latency_ms",
//!     &.{ 1, 5, 10, 25, 50, 100, 250, 500, 1000 },
//! );
//! defer histogram.deinit(allocator);
//!
//! histogram.record(45); // Records in 10-25 bucket
//! ```
const std = @import("std");

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
        histogram.* = try Histogram.init(self.allocator, name, bounds);
        errdefer histogram.deinit(self.allocator);
        try self.histograms.append(self.allocator, histogram);
        return histogram;
    }
};

test "metrics collector registers counters and histograms" {
    var collector = MetricsCollector.init(std.testing.allocator);
    defer collector.deinit();

    const requests = try collector.registerCounter("requests_total");
    const errors = try collector.registerCounter("errors_total");
    const latency = try collector.registerHistogram("latency_ms", &.{ 1, 5, 10 });

    requests.inc(1);
    errors.inc(2);
    latency.record(3);

    try std.testing.expectEqual(@as(u64, 1), requests.get());
    try std.testing.expectEqual(@as(u64, 2), errors.get());
    try std.testing.expectEqual(@as(u64, 1), latency.buckets[1]);
}
