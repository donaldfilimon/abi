const std = @import("std");

/// Threshold-based anomaly detector over scalar telemetry streams (design
/// reference: `docs/spec/wdbx-rust-capability-extract.mdx` §6).  Deterministic,
/// no model calls — only basic statistics (mean, deviation, spike-threshold).
pub const IotMonitor = struct {
    allocator: std.mem.Allocator,
    history: std.ArrayListUnmanaged(f64) = .empty,
    z_threshold: f64,

    pub fn init(allocator: std.mem.Allocator) IotMonitor {
        return .{
            .allocator = allocator,
            .history = .empty,
            .z_threshold = 2.5,
        };
    }

    pub fn deinit(self: *IotMonitor) void {
        self.history.deinit(self.allocator);
    }

    /// Feed a scalar reading, record it, and return `true` if the reading
    /// deviates from the running mean by more than `z_threshold` standard
    /// deviations (using Welford's online algorithm for mean and variance).
    pub fn feed(self: *IotMonitor, value: f64) !bool {
        try self.history.append(self.allocator, value);
        if (self.history.items.len < 3) return false;
        const n = self.history.items.len;
        var mean: f64 = 0;
        var m2: f64 = 0;
        for (self.history.items, 1..) |x, i| {
            const delta = x - mean;
            mean += delta / @as(f64, @floatFromInt(i));
            m2 += delta * (x - mean);
        }
        if (n < 2) return false;
        const variance = m2 / @as(f64, @floatFromInt(n - 1));
        const stddev = @sqrt(variance);
        if (stddev < 1e-12) return false;
        const z = @abs(value - mean) / stddev;
        return z >= self.z_threshold;
    }

    /// Number of readings collected so far.
    pub fn count(self: IotMonitor) usize {
        return self.history.items.len;
    }

    /// Reset the monitor — clear history and restore default threshold.
    pub fn reset(self: *IotMonitor) void {
        self.history.clearAndFree(self.allocator);
        self.z_threshold = 2.5;
    }
};

test "IotMonitor returns false for fewer than 3 readings" {
    var monitor = IotMonitor.init(std.testing.allocator);
    defer monitor.deinit();
    try std.testing.expect(!try monitor.feed(10.0));
    try std.testing.expect(!try monitor.feed(10.0));
    try std.testing.expectEqual(@as(usize, 2), monitor.count());
}

test "IotMonitor detects anomaly with z-score threshold" {
    var monitor = IotMonitor.init(std.testing.allocator);
    defer monitor.deinit();
    // 10 stable readings at 100
    for (0..10) |_| {
        _ = try monitor.feed(100.0);
    }
    // feed a spike
    const anomalous = try monitor.feed(300.0);
    try std.testing.expect(anomalous);
}

test "IotMonitor does not flag normal variation" {
    var monitor = IotMonitor.init(std.testing.allocator);
    defer monitor.deinit();
    const readings = [_]f64{ 98, 102, 99, 101, 100, 99, 101, 100, 102, 98 };
    for (readings) |r| {
        _ = try monitor.feed(r);
    }
    const normal = try monitor.feed(100.5);
    try std.testing.expect(!normal);
}

test "IotMonitor reset clears state" {
    var monitor = IotMonitor.init(std.testing.allocator);
    defer monitor.deinit();
    for (0..5) |_| {
        _ = try monitor.feed(1.0);
    }
    try std.testing.expect(monitor.count() > 0);
    monitor.reset();
    try std.testing.expectEqual(@as(usize, 0), monitor.count());
}

test {
    std.testing.refAllDecls(@This());
}
