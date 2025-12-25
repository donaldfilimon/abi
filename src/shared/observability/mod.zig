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

    pub fn init(allocator: std.mem.Allocator, name: []const u8, bounds: []const u64) !Histogram {
        const bucket_copy = try allocator.alloc(u64, bounds.len + 1);
        errdefer allocator.free(bucket_copy);
        const bound_copy = try allocator.alloc(u64, bounds.len);
        errdefer allocator.free(bound_copy);
        std.mem.set(u64, bucket_copy, 0);
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
    counters: std.ArrayList(Counter),
    histograms: std.ArrayList(Histogram),

    pub fn init(allocator: std.mem.Allocator) MetricsCollector {
        return .{
            .allocator = allocator,
            .counters = std.ArrayList(Counter).empty,
            .histograms = std.ArrayList(Histogram).empty,
        };
    }

    pub fn deinit(self: *MetricsCollector) void {
        self.counters.deinit(self.allocator);
        for (self.histograms.items) |*histogram| {
            histogram.deinit(self.allocator);
        }
        self.histograms.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn registerCounter(self: *MetricsCollector, name: []const u8) !*Counter {
        try self.counters.append(self.allocator, .{ .name = name });
        return &self.counters.items[self.counters.items.len - 1];
    }

    pub fn registerHistogram(self: *MetricsCollector, name: []const u8, bounds: []const u64) !*Histogram {
        const histogram = try Histogram.init(self.allocator, name, bounds);
        try self.histograms.append(self.allocator, histogram);
        return &self.histograms.items[self.histograms.items.len - 1];
    }
};
