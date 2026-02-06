//! Sliding Window Metrics (Stub)
//!
//! Stub implementation of time-windowed metrics.
//! All operations are no-ops when observability is disabled.

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");

/// A sample with timestamp.
pub const TimestampedSample = struct {
    value: f64,
    timestamp_ms: i64,
};

/// Stub sliding window - all operations are no-ops.
pub fn SlidingWindow(comptime max_samples: usize) type {
    return struct {
        const Self = @This();

        samples: [max_samples]TimestampedSample = undefined,
        count: usize = 0,
        head: usize = 0,
        window_ms: i64,
        mutex: sync.Mutex = .{},

        pub fn init(window_ms: i64) Self {
            return .{ .window_ms = window_ms };
        }

        pub fn record(_: *Self, _: f64, _: i64) void {}
        pub fn percentile(_: *Self, _: f64, _: i64) f64 {
            return 0;
        }
        pub fn mean(_: *Self, _: i64) f64 {
            return 0;
        }
        pub fn validCount(_: *Self, _: i64) usize {
            return 0;
        }
        pub fn min(_: *Self, _: i64) f64 {
            return 0;
        }
        pub fn max(_: *Self, _: i64) f64 {
            return 0;
        }
    };
}

/// Standard 1000-sample sliding window.
pub const StandardWindow = SlidingWindow(1000);
