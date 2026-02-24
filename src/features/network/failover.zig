//! Automatic Failover Manager
//!
//! Monitors node health and triggers automatic failover when primary fails.

const std = @import("std");
const loadbalancer = @import("loadbalancer.zig");
const platform_time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

pub const FailoverConfig = struct {
    /// Health check interval in milliseconds.
    health_check_interval_ms: u64 = 5000,
    /// Number of failed checks before triggering failover.
    failure_threshold: u32 = 3,
    /// Timeout for health checks in milliseconds.
    health_check_timeout_ms: u64 = 2000,
    /// Enable automatic failover.
    auto_failover: bool = true,
};

pub const FailoverState = enum {
    normal,
    monitoring,
    failing_over,
    failed_over,
    recovering,
};

pub const FailoverEvent = struct {
    timestamp_ms: i64,
    event_type: EventType,
    node_id: []const u8,
    details: ?[]const u8,

    pub const EventType = enum {
        health_check_failed,
        failover_started,
        failover_completed,
        recovery_started,
        recovery_completed,
    };
};

pub const FailoverManager = struct {
    allocator: std.mem.Allocator,
    config: FailoverConfig,
    state: FailoverState,
    primary_node: ?[]const u8,
    secondary_nodes: std.ArrayListUnmanaged([]const u8),
    failure_counts: std.StringHashMapUnmanaged(u32),
    event_log: std.ArrayListUnmanaged(FailoverEvent),
    mutex: sync.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: FailoverConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .normal,
            .primary_node = null,
            .secondary_nodes = .empty,
            .failure_counts = .empty,
            .event_log = .empty,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.secondary_nodes.items) |node| {
            self.allocator.free(node);
        }
        self.secondary_nodes.deinit(self.allocator);

        if (self.primary_node) |node| {
            self.allocator.free(node);
        }

        var it = self.failure_counts.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.failure_counts.deinit(self.allocator);

        for (self.event_log.items) |event| {
            self.allocator.free(event.node_id);
            if (event.details) |d| self.allocator.free(d);
        }
        self.event_log.deinit(self.allocator);
    }

    pub fn setPrimary(self: *Self, node_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.primary_node) |old| self.allocator.free(old);
        self.primary_node = try self.allocator.dupe(u8, node_id);
    }

    pub fn addSecondary(self: *Self, node_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const id = try self.allocator.dupe(u8, node_id);
        try self.secondary_nodes.append(self.allocator, id);
    }

    pub fn recordHealthCheckResult(self: *Self, node_id: []const u8, success: bool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (success) {
            if (self.failure_counts.fetchRemove(node_id)) |entry| {
                self.allocator.free(entry.key);
            }
            return;
        }

        // Increment failure count
        const entry = try self.failure_counts.getOrPut(self.allocator, node_id);
        if (entry.found_existing) {
            entry.value_ptr.* += 1;
        } else {
            entry.key_ptr.* = try self.allocator.dupe(u8, node_id);
            entry.value_ptr.* = 1;
        }

        // Check if failover needed
        if (entry.value_ptr.* >= self.config.failure_threshold) {
            if (self.config.auto_failover and
                self.primary_node != null and
                std.mem.eql(u8, self.primary_node.?, node_id))
            {
                try self.triggerFailover(node_id);
            }
        }
    }

    fn getTimestampMs() i64 {
        return @intCast(platform_time.timestampMs());
    }

    fn triggerFailover(self: *Self, failed_node: []const u8) !void {
        self.state = .failing_over;

        try self.logEvent(.{
            .timestamp_ms = getTimestampMs(),
            .event_type = .failover_started,
            .node_id = failed_node,
            .details = "Primary node failed, initiating failover",
        });

        // Promote first available secondary
        if (self.secondary_nodes.items.len > 0) {
            const new_primary = self.secondary_nodes.orderedRemove(0);

            if (self.primary_node) |old| self.allocator.free(old);
            self.primary_node = new_primary;
            self.state = .failed_over;

            try self.logEvent(.{
                .timestamp_ms = getTimestampMs(),
                .event_type = .failover_completed,
                .node_id = new_primary,
                .details = "Promoted to primary",
            });
        } else {
            self.state = .normal; // No secondaries available
        }
    }

    fn logEvent(self: *Self, event: FailoverEvent) !void {
        const node_id_copy = try self.allocator.dupe(u8, event.node_id);
        errdefer self.allocator.free(node_id_copy);
        const details_copy = if (event.details) |d| try self.allocator.dupe(u8, d) else null;
        errdefer if (details_copy) |d| self.allocator.free(d);

        var e = event;
        e.node_id = node_id_copy;
        e.details = details_copy;

        try self.event_log.append(self.allocator, e);
    }

    pub fn getState(self: *Self) FailoverState {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.state;
    }

    pub fn getEventLog(self: *Self) []const FailoverEvent {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.event_log.items;
    }
};

test "FailoverManager basic operations" {
    var fm = FailoverManager.init(std.testing.allocator, .{
        .failure_threshold = 2,
        .auto_failover = true,
    });
    defer fm.deinit();

    try fm.setPrimary("node-1");
    try fm.addSecondary("node-2");

    // First failure - no failover yet
    try fm.recordHealthCheckResult("node-1", false);
    try std.testing.expectEqual(FailoverState.normal, fm.getState());

    // Second failure - triggers failover
    try fm.recordHealthCheckResult("node-1", false);
    try std.testing.expectEqual(FailoverState.failed_over, fm.getState());
}

test {
    std.testing.refAllDecls(@This());
}
