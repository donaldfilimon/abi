//! Shared Metric Primitives
//!
//! Thread-safe metric types for counters, gauges, and histograms.
//! These are infrastructure types used by multiple feature modules
//! (observability, GPU, AI) for consistent metrics collection.

const std = @import("std");
const sync = @import("../sync.zig");

/// Monotonically increasing counter.
pub const Counter = struct {
    value: u64 = 0,

    pub fn inc(self: *Counter) void {
        _ = @atomicRmw(u64, &self.value, .Add, 1, .monotonic);
    }

    pub fn add(self: *Counter, n: u64) void {
        _ = @atomicRmw(u64, &self.value, .Add, n, .monotonic);
    }

    pub fn get(self: *const Counter) u64 {
        return @atomicLoad(u64, &self.value, .monotonic);
    }

    pub fn reset(self: *Counter) void {
        @atomicStore(u64, &self.value, 0, .monotonic);
    }
};

/// Value that can increase or decrease.
pub const Gauge = struct {
    value: i64 = 0,

    pub fn set(self: *Gauge, v: i64) void {
        @atomicStore(i64, &self.value, v, .monotonic);
    }

    pub fn inc(self: *Gauge) void {
        _ = @atomicRmw(i64, &self.value, .Add, 1, .monotonic);
    }

    pub fn dec(self: *Gauge) void {
        _ = @atomicRmw(i64, &self.value, .Sub, 1, .monotonic);
    }

    pub fn add(self: *Gauge, v: i64) void {
        _ = @atomicRmw(i64, &self.value, .Add, v, .monotonic);
    }

    pub fn get(self: *const Gauge) i64 {
        return @atomicLoad(i64, &self.value, .monotonic);
    }
};

/// Float gauge for non-integer measurements (requires mutex).
pub const FloatGauge = struct {
    value: f64 = 0,
    mutex: sync.Mutex = .{},

    pub fn set(self: *FloatGauge, v: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value = v;
    }

    pub fn add(self: *FloatGauge, v: f64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value += v;
    }

    pub fn get(self: *FloatGauge) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.value;
    }
};

/// Standard latency buckets in milliseconds.
pub const default_latency_buckets = [_]f64{ 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000 };

/// Histogram with configurable buckets.
pub fn Histogram(comptime bucket_count: usize) type {
    return struct {
        const Self = @This();

        buckets: [bucket_count]u64 = [_]u64{0} ** bucket_count,
        bucket_bounds: [bucket_count]f64,
        sum: f64 = 0,
        count: u64 = 0,
        mutex: sync.Mutex = .{},

        pub fn init(bounds: [bucket_count]f64) Self {
            return .{ .bucket_bounds = bounds };
        }

        pub fn initDefault() Self {
            comptime {
                if (bucket_count != default_latency_buckets.len) {
                    @compileError("Use Histogram(14) for default latency buckets");
                }
            }
            return .{ .bucket_bounds = default_latency_buckets };
        }

        pub fn observe(self: *Self, value: f64) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.sum += value;
            self.count += 1;
            for (&self.buckets, 0..) |*bucket, i| {
                if (value <= self.bucket_bounds[i]) {
                    bucket.* += 1;
                }
            }
        }

        pub fn mean(self: *Self) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.count == 0) return 0;
            return self.sum / @as(f64, @floatFromInt(self.count));
        }

        pub fn percentile(self: *Self, p: f64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.count == 0) return 0;

            const target = @as(f64, @floatFromInt(self.count)) * p;
            var cumulative: u64 = 0;
            for (self.buckets, 0..) |bucket, i| {
                cumulative += bucket;
                if (@as(f64, @floatFromInt(cumulative)) >= target) {
                    return self.bucket_bounds[i];
                }
            }
            return self.bucket_bounds[bucket_count - 1];
        }

        pub fn getCount(self: *Self) u64 {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.count;
        }

        pub fn reset(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.buckets = [_]u64{0} ** bucket_count;
            self.sum = 0;
            self.count = 0;
        }
    };
}

/// Standard latency histogram with default buckets.
pub const LatencyHistogram = Histogram(default_latency_buckets.len);

test {
    std.testing.refAllDecls(@This());
}
