//! Metrics Feature (opt-in observability)
//!
//! Provides a stateful Metrics registry with counters and gauges.
//! Intended for instrumentation of scheduler, AI training/completion,
//! WDBX operations, and MCP surfaces.

pub const types = @import("types.zig");

const std = @import("std");
const build_options = @import("build_options");

pub const MetricsError = types.MetricsError;
pub const Error = types.Error;
pub const CounterSnapshot = types.CounterSnapshot;
pub const GaugeSnapshot = types.GaugeSnapshot;

pub const Metrics = struct {
    allocator: std.mem.Allocator,
    counters: std.StringHashMapUnmanaged(u64),
    gauges: std.StringHashMapUnmanaged(f64),
    enabled: bool = true,

    pub fn init(allocator: std.mem.Allocator) Metrics {
        return .{
            .allocator = allocator,
            .counters = .{},
            .gauges = .{},
        };
    }

    pub fn deinit(self: *Metrics) void {
        self.counters.deinit(self.allocator);
        self.gauges.deinit(self.allocator);
    }

    pub fn isEnabled() bool {
        return true;
    }

    pub fn increment(self: *Metrics, name: []const u8, delta: u64) !void {
        const gop = try self.counters.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.allocator.dupe(u8, name);
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += delta;
    }

    pub fn setGauge(self: *Metrics, name: []const u8, value: f64) !void {
        const gop = try self.gauges.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.allocator.dupe(u8, name);
        }
        gop.value_ptr.* = value;
    }

    pub fn getCounter(self: *const Metrics, name: []const u8) ?u64 {
        return self.counters.get(name);
    }

    pub fn snapshotCounters(self: *const Metrics, allocator: std.mem.Allocator) ![]CounterSnapshot {
        var list: std.ArrayListUnmanaged(CounterSnapshot) = .empty;
        errdefer list.deinit(allocator);

        var it = self.counters.iterator();
        while (it.next()) |entry| {
            try list.append(allocator, .{
                .name = try allocator.dupe(u8, entry.key_ptr.*),
                .value = entry.value_ptr.*,
            });
        }
        return list.toOwnedSlice(allocator);
    }

    pub fn snapshotGauges(self: *const Metrics, allocator: std.mem.Allocator) ![]GaugeSnapshot {
        _ = self;
        _ = allocator;
        // Placeholder - full impl would mirror counters
        return &[_]GaugeSnapshot{};
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "metrics counter increment and snapshot" {
    var m = Metrics.init(std.testing.allocator);
    defer m.deinit();

    try m.increment("tasks.submitted", 1);
    try m.increment("tasks.submitted", 2);

    const val = m.getCounter("tasks.submitted");
    try std.testing.expectEqual(@as(u64, 3), val.?);

    const snaps = try m.snapshotCounters(std.testing.allocator);
    defer {
        for (snaps) |s| std.testing.allocator.free(s.name);
        std.testing.allocator.free(snaps);
    }
    try std.testing.expectEqual(@as(usize, 1), snaps.len);
}
