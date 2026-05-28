//! Metrics stub -- disabled at compile time (opt-in feature).
//!
//! Identical public surface. All operations are no-ops or return
//! FeatureDisabled where appropriate. Used when -Dfeat-metrics=false.

pub const types = @import("types.zig");

const std = @import("std");

pub const MetricsError = types.MetricsError;
pub const Error = types.Error;
pub const CounterSnapshot = types.CounterSnapshot;
pub const GaugeSnapshot = types.GaugeSnapshot;

pub const Metrics = struct {
    allocator: std.mem.Allocator,
    counters: std.StringHashMapUnmanaged(u64),
    gauges: std.StringHashMapUnmanaged(f64),
    enabled: bool = false,

    pub fn init(allocator: std.mem.Allocator) Metrics {
        return .{
            .allocator = allocator,
            .counters = .{},
            .gauges = .{},
        };
    }

    pub fn deinit(self: *Metrics) void {
        _ = self;
    }

    pub fn isEnabled() bool {
        return false;
    }

    pub fn increment(self: *Metrics, name: []const u8, delta: u64) !void {
        _ = self;
        _ = name;
        _ = delta;
        return error.FeatureDisabled;
    }

    pub fn setGauge(self: *Metrics, name: []const u8, value: f64) !void {
        _ = self;
        _ = name;
        _ = value;
        return error.FeatureDisabled;
    }

    pub fn getCounter(self: *const Metrics, name: []const u8) ?u64 {
        _ = self;
        _ = name;
        return null;
    }

    pub fn snapshotCounters(self: *const Metrics, allocator: std.mem.Allocator) ![]CounterSnapshot {
        _ = self;
        _ = allocator;
        return &[_]CounterSnapshot{};
    }

    pub fn snapshotGauges(self: *const Metrics, allocator: std.mem.Allocator) ![]GaugeSnapshot {
        _ = self;
        _ = allocator;
        return &[_]GaugeSnapshot{};
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "metrics stub reports disabled and errors cleanly" {
    var m = Metrics.init(std.testing.allocator);
    defer m.deinit();

    try std.testing.expect(!Metrics.isEnabled());
    try std.testing.expectError(error.FeatureDisabled, m.increment("x", 1));
    try std.testing.expect(m.getCounter("x") == null);
}
