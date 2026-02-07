//! Core Metrics Primitives (Stub)
//!
//! Stub implementations of thread-safe metric types.
//! All operations are no-ops when observability is disabled.

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");

/// Stub counter - all operations are no-ops.
pub const Counter = struct {
    const Self = @This();
    value: u64 = 0,

    pub fn inc(_: *Self) void {}
    pub fn add(_: *Self, _: u64) void {}
    pub fn get(_: *const Self) u64 {
        return 0;
    }
    pub fn reset(_: *Self) void {}
};

/// Stub gauge - all operations are no-ops.
pub const Gauge = struct {
    const Self = @This();
    value: i64 = 0,

    pub fn set(_: *Self, _: i64) void {}
    pub fn inc(_: *Self) void {}
    pub fn dec(_: *Self) void {}
    pub fn add(_: *Self, _: i64) void {}
    pub fn get(_: *const Self) i64 {
        return 0;
    }
};

/// Stub float gauge - all operations are no-ops.
pub const FloatGauge = struct {
    const Self = @This();
    value: f64 = 0,
    mutex: sync.Mutex = .{},

    pub fn set(_: *Self, _: f64) void {}
    pub fn add(_: *Self, _: f64) void {}
    pub fn get(_: *Self) f64 {
        return 0;
    }
};

/// Standard latency buckets (matches real implementation).
pub const default_latency_buckets = [_]f64{ 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000 };

/// Stub histogram - all operations are no-ops.
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

        pub fn observe(_: *Self, _: f64) void {}
        pub fn mean(_: *Self) f64 {
            return 0;
        }
        pub fn percentile(_: *Self, _: f64) f64 {
            return 0;
        }
        pub fn getCount(_: *Self) u64 {
            return 0;
        }
        pub fn reset(self: *Self) void {
            self.buckets = [_]u64{0} ** bucket_count;
            self.sum = 0;
            self.count = 0;
        }
    };
}

/// Standard latency histogram with default buckets.
pub const LatencyHistogram = Histogram(default_latency_buckets.len);
