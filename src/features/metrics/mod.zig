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
        // Keys are duped on first insert (increment/setGauge), so free them
        // before tearing down the maps to avoid leaking the name strings.
        var counter_keys = self.counters.keyIterator();
        while (counter_keys.next()) |key| self.allocator.free(key.*);
        self.counters.deinit(self.allocator);

        var gauge_keys = self.gauges.keyIterator();
        while (gauge_keys.next()) |key| self.allocator.free(key.*);
        self.gauges.deinit(self.allocator);
    }

    pub fn isEnabled() bool {
        return true;
    }

    pub fn increment(self: *Metrics, name: []const u8, delta: u64) !void {
        const gop = try self.counters.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            errdefer _ = self.counters.remove(name);
            gop.key_ptr.* = try self.allocator.dupe(u8, name);
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += delta;
    }

    pub fn setGauge(self: *Metrics, name: []const u8, value: f64) !void {
        const gop = try self.gauges.getOrPut(self.allocator, name);
        if (!gop.found_existing) {
            errdefer _ = self.gauges.remove(name);
            gop.key_ptr.* = try self.allocator.dupe(u8, name);
        }
        gop.value_ptr.* = value;
    }

    pub fn getCounter(self: *const Metrics, name: []const u8) ?u64 {
        return self.counters.get(name);
    }

    pub fn snapshotCounters(self: *const Metrics, allocator: std.mem.Allocator) ![]CounterSnapshot {
        var list: std.ArrayListUnmanaged(CounterSnapshot) = .empty;
        errdefer {
            for (list.items) |snapshot| allocator.free(snapshot.name);
            list.deinit(allocator);
        }

        var it = self.counters.iterator();
        while (it.next()) |entry| {
            const name = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(name);
            try list.append(allocator, .{
                .name = name,
                .value = entry.value_ptr.*,
            });
        }
        return list.toOwnedSlice(allocator);
    }

    pub fn getGauge(self: *const Metrics, name: []const u8) ?f64 {
        return self.gauges.get(name);
    }

    pub fn snapshotGauges(self: *const Metrics, allocator: std.mem.Allocator) ![]GaugeSnapshot {
        var list: std.ArrayListUnmanaged(GaugeSnapshot) = .empty;
        errdefer {
            for (list.items) |snapshot| allocator.free(snapshot.name);
            list.deinit(allocator);
        }

        var it = self.gauges.iterator();
        while (it.next()) |entry| {
            const name = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(name);
            try list.append(allocator, .{
                .name = name,
                .value = entry.value_ptr.*,
            });
        }
        return list.toOwnedSlice(allocator);
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

test "metrics gauge set, read back, and snapshot" {
    var m = Metrics.init(std.testing.allocator);
    defer m.deinit();

    try m.setGauge("queue.depth", 4.0);
    try m.setGauge("queue.depth", 7.5); // overwrite
    try m.setGauge("mem.mb", 128.0);

    try std.testing.expectEqual(@as(f64, 7.5), m.getGauge("queue.depth").?);
    try std.testing.expect(m.getGauge("missing") == null);

    const snaps = try m.snapshotGauges(std.testing.allocator);
    defer {
        for (snaps) |s| std.testing.allocator.free(s.name);
        std.testing.allocator.free(snaps);
    }
    // Two distinct gauges, regardless of iteration order.
    try std.testing.expectEqual(@as(usize, 2), snaps.len);
}
