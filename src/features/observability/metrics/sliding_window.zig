//! Sliding Window Metrics
//!
//! Time-windowed metrics that automatically expire old samples.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// A sample with timestamp.
pub const TimestampedSample = struct {
    value: f64,
    timestamp_ms: i64,
};

/// Sliding window for latency tracking with automatic expiration.
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

        pub fn record(self: *Self, value: f64, now_ms: i64) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            // Add new sample
            self.samples[self.head] = .{ .value = value, .timestamp_ms = now_ms };
            self.head = (self.head + 1) % max_samples;
            if (self.count < max_samples) self.count += 1;
        }

        pub fn percentile(self: *Self, p: f64, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;

            // Collect valid samples
            var valid: [max_samples]f64 = undefined;
            var valid_count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    valid[valid_count] = self.samples[idx].value;
                    valid_count += 1;
                }
            }

            if (valid_count == 0) return 0;

            // Sort for percentile calculation
            std.mem.sort(f64, valid[0..valid_count], {}, std.sort.asc(f64));

            const index = @as(usize, @intFromFloat(@as(f64, @floatFromInt(valid_count - 1)) * p));
            return valid[index];
        }

        pub fn mean(self: *Self, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var sum: f64 = 0;
            var count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    sum += self.samples[idx].value;
                    count += 1;
                }
            }

            if (count == 0) return 0;
            return sum / @as(f64, @floatFromInt(count));
        }

        pub fn validCount(self: *Self, now_ms: i64) usize {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var count: usize = 0;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    count += 1;
                }
            }

            return count;
        }

        pub fn min(self: *Self, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var min_val: f64 = std.math.floatMax(f64);
            var found = false;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    min_val = @min(min_val, self.samples[idx].value);
                    found = true;
                }
            }

            return if (found) min_val else 0;
        }

        pub fn max(self: *Self, now_ms: i64) f64 {
            self.mutex.lock();
            defer self.mutex.unlock();

            const cutoff = now_ms - self.window_ms;
            var max_val: f64 = std.math.floatMin(f64);
            var found = false;

            for (0..self.count) |i| {
                const idx = (self.head + max_samples - 1 - i) % max_samples;
                if (self.samples[idx].timestamp_ms >= cutoff) {
                    max_val = @max(max_val, self.samples[idx].value);
                    found = true;
                }
            }

            return if (found) max_val else 0;
        }
    };
}

/// Standard 1000-sample sliding window.
pub const StandardWindow = SlidingWindow(1000);

// ============================================================================
// Tests
// ============================================================================

test "sliding window percentile" {
    var window = SlidingWindow(100).init(60000); // 1 minute window

    // Record some values
    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);
    window.record(40, 4000);
    window.record(50, 5000);

    // P50 should be around 30
    const p50 = window.percentile(0.5, 10000);
    try std.testing.expect(p50 >= 20 and p50 <= 40);
}

test "sliding window mean" {
    var window = SlidingWindow(100).init(60000);

    window.record(10, 1000);
    window.record(20, 2000);
    window.record(30, 3000);

    const avg = window.mean(10000);
    try std.testing.expectApproxEqAbs(@as(f64, 20), avg, 0.001);
}

test "sliding window expiration" {
    var window = SlidingWindow(100).init(5000); // 5 second window

    window.record(100, 1000);
    window.record(200, 3000);
    window.record(300, 8000);

    // At time 10000, only the sample at 8000 should be valid
    const count = window.validCount(10000);
    try std.testing.expectEqual(@as(usize, 1), count);

    const avg = window.mean(10000);
    try std.testing.expectApproxEqAbs(@as(f64, 300), avg, 0.001);
}

test "sliding window min max" {
    var window = SlidingWindow(100).init(60000);

    window.record(50, 1000);
    window.record(10, 2000);
    window.record(90, 3000);
    window.record(30, 4000);

    const min_val = window.min(10000);
    const max_val = window.max(10000);

    try std.testing.expectApproxEqAbs(@as(f64, 10), min_val, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 90), max_val, 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
