//! Metrics collection and reporting system
//! Updated for Zig 0.16 compatibility

const std = @import("std");
const collections = @import("core/collections.zig");

/// Metrics registry for collecting and reporting metrics
pub const MetricsRegistry = struct {
    const Self = @This();

    counters: collections.StringHashMap(u64),
    gauges: collections.StringHashMap(f64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .counters = collections.StringHashMap(u64).init(allocator),
            .gauges = collections.StringHashMap(f64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.counters.deinit();
        self.gauges.deinit();
    }

    pub fn incrementCounter(self: *Self, name: []const u8) !void {
        const current = self.counters.get(name) orelse 0;
        try self.counters.put(name, current + 1);
    }

    pub fn setGauge(self: *Self, name: []const u8, value: f64) !void {
        try self.gauges.put(name, value);
    }
};

/// Starts a simple HTTP metrics exporter on port 9100.
/// The server listens for incoming connections and replies with a
/// plain‑text payload that follows the Prometheus `/metrics` format.
pub fn startMetricsServer() !void {
    // For compatibility across Zig stdlib changes, the metrics server is
    // currently a no-op that only logs when invoked. Replace with a proper
    // implementation using std.net.StreamServer if needed.
    std.log.info("Metrics server placeholder: not listening (compat shim)", .{});
    return;
}

pub fn main() !void {
    try startMetricsServer();
}

test "metrics - basic operations" {
    const testing = std.testing;
    var metrics = MetricsRegistry.init(testing.allocator);
    defer metrics.deinit();

    try metrics.incrementCounter("test_counter");
    try metrics.setGauge("test_gauge", 42.0);

    try testing.expectEqual(@as(u64, 1), metrics.counters.get("test_counter").?);
    try testing.expectEqual(@as(f64, 42.0), metrics.gauges.get("test_gauge").?);
}
