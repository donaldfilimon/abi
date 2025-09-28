//! In-process metrics registry with persona aware counters.

const std = @import("std");

pub const MetricsRegistry = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    total_calls: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    total_errors: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    total_latency_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    persona_counts: std.AutoHashMap([]const u8, *std.atomic.Value(u64)),

    pub fn init(allocator: std.mem.Allocator) MetricsRegistry {
        return MetricsRegistry{
            .allocator = allocator,
            .persona_counts = std.AutoHashMap([]const u8, *std.atomic.Value(u64)).init(allocator),
        };
    }

    pub fn deinit(self: *MetricsRegistry) void {
        var it = self.persona_counts.iterator();
        while (it.next()) |entry| {
            self.allocator.destroy(entry.value_ptr.*);
            self.allocator.free(@constCast(entry.key_ptr.*));
        }
        self.persona_counts.deinit();
    }

    pub fn recordCall(self: *MetricsRegistry, persona: []const u8, latency_ns: u64, success: bool) !void {
        _ = self.total_calls.fetchAdd(1, .monotonic);
        _ = self.total_latency_ns.fetchAdd(latency_ns, .monotonic);
        if (!success) {
            _ = self.total_errors.fetchAdd(1, .monotonic);
        }

        const counter = try self.ensurePersonaCounter(persona);
        _ = counter.fetchAdd(1, .monotonic);
    }

    fn ensurePersonaCounter(self: *MetricsRegistry, persona: []const u8) !*std.atomic.Value(u64) {
        if (self.persona_counts.get(persona)) |existing| {
            return existing;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.persona_counts.get(persona)) |existing| {
            return existing;
        }

        const counter = try self.allocator.create(std.atomic.Value(u64));
        counter.* = std.atomic.Value(u64).init(0);
        const key = try std.mem.dupe(self.allocator, u8, persona);
        try self.persona_counts.put(key, counter);
        return counter;
    }

    pub fn snapshot(self: *MetricsRegistry, allocator: std.mem.Allocator) !Snapshot {
        var persona_data = std.ArrayList(Snapshot.PersonaMetric).init(allocator);
        defer persona_data.deinit();

        var it = self.persona_counts.iterator();
        while (it.next()) |entry| {
            try persona_data.append(.{
                .persona = entry.key_ptr.*, // stored slice already duplicated
                .count = entry.value_ptr.*.load(.monotonic),
            });
        }

        return Snapshot{
            .total_calls = self.total_calls.load(.monotonic),
            .total_errors = self.total_errors.load(.monotonic),
            .total_latency_ns = self.total_latency_ns.load(.monotonic),
            .persona = try persona_data.toOwnedSlice(),
        };
    }
};

pub const Snapshot = struct {
    total_calls: u64,
    total_errors: u64,
    total_latency_ns: u64,
    persona: []Snapshot.PersonaMetric,

    pub fn averageLatencyNs(self: Snapshot) u64 {
        if (self.total_calls == 0) return 0;
        return @intCast(self.total_latency_ns / self.total_calls);
    }

    pub const PersonaMetric = struct {
        persona: []const u8,
        count: u64,
    };
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "registry tracks persona counts" {
    var registry = MetricsRegistry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.recordCall("adaptive", 100, true);
    try registry.recordCall("adaptive", 200, false);
    try registry.recordCall("technical", 300, true);

    const snapshot = try registry.snapshot(std.testing.allocator);
    defer std.testing.allocator.free(snapshot.persona);

    try std.testing.expectEqual(@as(u64, 3), snapshot.total_calls);
    try std.testing.expectEqual(@as(u64, 1), snapshot.total_errors);
    try std.testing.expectEqual(@as(u64, 200), snapshot.averageLatencyNs());
}

