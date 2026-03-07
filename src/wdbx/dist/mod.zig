//! Distributed coordination: node membership, heartbeat state machine, shard map.
//!
//! Allows growth from single-node to clustered deployment. Node health is
//! driven by heartbeat timeouts (stale → failed) and optional shard balancing.

const std = @import("std");

/// Binary RPC codec for node-to-node messages (heartbeat, block sync).
pub const rpc = @import("rpc.zig");
/// In-process block sync path (request/response/chunk stream); see runRequesterPath.
pub const replication = @import("replication.zig");

/// Health state derived from last_seen vs configured timeouts.
pub const HealthState = enum {
    joining,
    active,
    stale,
    failed,
};

/// Configurable timeouts (seconds) for the heartbeat state machine.
pub const HeartbeatConfig = struct {
    /// Mark node stale after this many seconds without heartbeat.
    stale_after_s: i64 = 30,
    /// Mark node failed after this many seconds without heartbeat.
    fail_after_s: i64 = 90,
};

pub const NodeState = struct {
    id: u32,
    address: []const u8,
    /// Derived from last_seen and HeartbeatConfig; do not set directly.
    is_healthy: bool = true,
    /// Last heartbeat timestamp (unix seconds; pass from your time source).
    last_seen: i64 = 0,
    load_factor: f32 = 0.0,
    /// Cached health state; updated by Coordinator.tick().
    health_state: HealthState = .joining,
};

/// Optional callback when a node's health state changes (for trace logs). Signature: (node_id, old_state, new_state).
pub const TraceStateChangeFn = *const fn (node_id: u32, old: HealthState, new: HealthState) void;

pub const Coordinator = struct {
    nodes: std.ArrayListUnmanaged(NodeState) = .empty,
    shard_map: std.AutoHashMapUnmanaged(u32, u32) = .empty, // shard_id -> node_id
    config: HeartbeatConfig = .{},
    /// When set, called on each health_state transition in tick() for trace logging.
    trace_state_change: ?TraceStateChangeFn = null,

    pub fn deinit(self: *Coordinator, allocator: std.mem.Allocator) void {
        for (self.nodes.items) |node| {
            allocator.free(node.address);
        }
        self.nodes.deinit(allocator);
        self.shard_map.deinit(allocator);
    }

    /// Configure heartbeat timeouts. Call before or after registering nodes.
    pub fn setHeartbeatConfig(self: *Coordinator, config: HeartbeatConfig) void {
        self.config = config;
    }

    /// Register a node; pass current unix time in seconds (e.g. from your time module).
    pub fn registerNode(self: *Coordinator, allocator: std.mem.Allocator, address: []const u8, now: i64) !u32 {
        if (address.len > 256) return error.AddressTooLong;
        const id = @as(u32, @truncate(self.nodes.items.len));
        try self.nodes.append(allocator, .{
            .id = id,
            .address = try allocator.dupe(u8, address),
            .last_seen = now,
            .health_state = .active,
        });
        return id;
    }

    /// Update heartbeat for a node; pass current unix time in seconds.
    pub fn updateHeartbeat(self: *Coordinator, node_id: u32, load_factor: f32, now: i64) void {
        if (node_id < self.nodes.items.len) {
            var node = &self.nodes.items[node_id];
            node.last_seen = now;
            node.load_factor = load_factor;
            node.health_state = .active;
            node.is_healthy = true;
        }
    }

    /// Recompute health_state and is_healthy for all nodes from last_seen.
    /// Call periodically (e.g. every few seconds) to mark stale/failed nodes.
    /// If trace_state_change is set, it is called for each node whose health_state changed.
    pub fn tick(self: *Coordinator, now: i64) void {
        for (self.nodes.items) |*node| {
            const elapsed = if (now < node.last_seen) 0 else now - node.last_seen;
            const old = node.health_state;
            if (elapsed >= self.config.fail_after_s) {
                node.health_state = .failed;
                node.is_healthy = false;
            } else if (elapsed >= self.config.stale_after_s) {
                node.health_state = .stale;
                node.is_healthy = false;
            } else {
                node.health_state = .active;
                node.is_healthy = true;
            }
            if (self.trace_state_change) |trace| {
                if (old != node.health_state) trace(node.id, old, node.health_state);
            }
        }
    }

    /// Call tick(now) with current unix time from your time source (e.g. platform time module).
    pub fn tickNow(self: *Coordinator, now: i64) void {
        self.tick(now);
    }

    /// Number of nodes currently considered healthy (active).
    pub fn healthyCount(self: *const Coordinator) u32 {
        var n: u32 = 0;
        for (self.nodes.items) |node| {
            if (node.is_healthy) n += 1;
        }
        return n;
    }

    /// Assign a shard to a node. Overwrites any existing assignment for shard_id.
    pub fn assignShard(self: *Coordinator, allocator: std.mem.Allocator, shard_id: u32, node_id: u32) !void {
        try self.shard_map.put(allocator, shard_id, node_id);
    }

    /// Return the node_id that owns the shard, or null if unassigned.
    pub fn getShardOwner(self: *const Coordinator, shard_id: u32) ?u32 {
        return self.shard_map.get(shard_id);
    }

    /// Remove shard assignments for a node (e.g. when node becomes failed).
    /// Call this from your trace_state_change callback when new == .failed, or after tick(); then optionally reassign shards to healthy nodes via assignShard.
    pub fn unassignShardsForNode(self: *Coordinator, allocator: std.mem.Allocator, node_id: u32) !void {
        var to_remove = std.ArrayListUnmanaged(u32){};
        defer to_remove.deinit(allocator);
        var it = self.shard_map.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* == node_id) try to_remove.append(allocator, entry.key_ptr.*);
        }
        for (to_remove.items) |shard_id| _ = self.shard_map.remove(shard_id);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

var test_trace_buf: [4]struct { id: u32, old: HealthState, new: HealthState } = undefined;
var test_trace_count: usize = 0;

var test_rebalance_coord: ?*Coordinator = null;
var test_rebalance_allocator: ?std.mem.Allocator = null;

fn testTraceStateChange(id: u32, old: HealthState, new: HealthState) void {
    if (test_trace_count < 4) {
        test_trace_buf[test_trace_count] = .{ .id = id, .old = old, .new = new };
        test_trace_count += 1;
    }
}

test "Coordinator: register and heartbeat" {
    const allocator = std.testing.allocator;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);

    const now: i64 = 1000000;
    const id = try coord.registerNode(allocator, "127.0.0.1:9200", now);
    try std.testing.expectEqual(@as(u32, 0), id);
    try std.testing.expect(coord.nodes.items[0].is_healthy);
    try std.testing.expectEqual(HealthState.active, coord.nodes.items[0].health_state);
}

test "Coordinator: heartbeat state machine stale and failed" {
    const allocator = std.testing.allocator;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);

    coord.config = .{ .stale_after_s = 5, .fail_after_s = 10 };
    const base: i64 = 1000000;
    _ = try coord.registerNode(allocator, "a", base);
    _ = try coord.registerNode(allocator, "b", base);

    coord.tick(base + 3);
    try std.testing.expect(coord.nodes.items[0].is_healthy);
    try std.testing.expectEqual(HealthState.active, coord.nodes.items[0].health_state);

    coord.tick(base + 6);
    try std.testing.expect(!coord.nodes.items[0].is_healthy);
    try std.testing.expectEqual(HealthState.stale, coord.nodes.items[0].health_state);

    coord.tick(base + 11);
    try std.testing.expectEqual(HealthState.failed, coord.nodes.items[0].health_state);
    try std.testing.expectEqual(@as(u32, 0), coord.healthyCount());

    coord.updateHeartbeat(0, 0.5, base + 11);
    coord.tick(base + 12);
    try std.testing.expect(coord.nodes.items[0].is_healthy);
    try std.testing.expectEqual(@as(u32, 1), coord.healthyCount());
}

test "Coordinator: trace_state_change called on transitions" {
    const allocator = std.testing.allocator;
    test_trace_count = 0;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);
    coord.config = .{ .stale_after_s = 5, .fail_after_s = 10 };
    coord.trace_state_change = testTraceStateChange;

    const base: i64 = 2000000;
    _ = try coord.registerNode(allocator, "a", base);
    coord.tick(base + 1);
    try std.testing.expectEqual(@as(usize, 0), test_trace_count);

    coord.tick(base + 6);
    try std.testing.expectEqual(@as(usize, 1), test_trace_count);
    try std.testing.expectEqual(HealthState.active, test_trace_buf[0].old);
    try std.testing.expectEqual(HealthState.stale, test_trace_buf[0].new);

    coord.tick(base + 11);
    try std.testing.expectEqual(@as(usize, 2), test_trace_count);
    try std.testing.expectEqual(HealthState.stale, test_trace_buf[1].old);
    try std.testing.expectEqual(HealthState.failed, test_trace_buf[1].new);
}

test "Coordinator: assignShard, getShardOwner, unassignShardsForNode" {
    const allocator = std.testing.allocator;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);
    const base: i64 = 3000000;
    _ = try coord.registerNode(allocator, "n0", base);
    _ = try coord.registerNode(allocator, "n1", base);

    try coord.assignShard(allocator, 0, 0);
    try coord.assignShard(allocator, 1, 1);
    try coord.assignShard(allocator, 2, 0);
    try std.testing.expectEqual(@as(u32, 0), coord.getShardOwner(0).?);
    try std.testing.expectEqual(@as(u32, 1), coord.getShardOwner(1).?);
    try std.testing.expectEqual(@as(u32, 0), coord.getShardOwner(2).?);
    try std.testing.expect(coord.getShardOwner(99) == null);

    try coord.unassignShardsForNode(allocator, 0);
    try std.testing.expect(coord.getShardOwner(0) == null);
    try std.testing.expectEqual(@as(u32, 1), coord.getShardOwner(1).?);
    try std.testing.expect(coord.getShardOwner(2) == null);
}

test "Coordinator: rebalance on fail via trace_state_change" {
    const allocator = std.testing.allocator;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);
    test_rebalance_coord = &coord;
    test_rebalance_allocator = allocator;
    defer {
        test_rebalance_coord = null;
        test_rebalance_allocator = null;
    }
    coord.config = .{ .stale_after_s = 5, .fail_after_s = 10 };
    coord.trace_state_change = struct {
        fn onTransition(id: u32, _: HealthState, new: HealthState) void {
            if (new == .failed) {
                const c = test_rebalance_coord orelse return;
                const a = test_rebalance_allocator orelse return;
                c.unassignShardsForNode(a, id) catch return;
            }
        }
    }.onTransition;

    const base: i64 = 4000000;
    _ = try coord.registerNode(allocator, "n0", base);
    _ = try coord.registerNode(allocator, "n1", base);
    try coord.assignShard(allocator, 0, 0);
    try coord.assignShard(allocator, 1, 0);
    try coord.assignShard(allocator, 2, 1);

    coord.tick(base + 11);
    try std.testing.expectEqual(HealthState.failed, coord.nodes.items[0].health_state);
    try std.testing.expect(coord.getShardOwner(0) == null);
    try std.testing.expect(coord.getShardOwner(1) == null);
    try std.testing.expectEqual(@as(u32, 1), coord.getShardOwner(2).?);

    try coord.assignShard(allocator, 0, 1);
    try coord.assignShard(allocator, 1, 1);
    try std.testing.expectEqual(@as(u32, 1), coord.getShardOwner(0).?);
    try std.testing.expectEqual(@as(u32, 1), coord.getShardOwner(1).?);
}

test "Coordinator: many nodes and shards (stress)" {
    const allocator = std.testing.allocator;
    var coord: Coordinator = .{};
    defer coord.deinit(allocator);
    const base: i64 = 5000000;
    const n_nodes: u32 = 20;
    const n_shards: u32 = 50;
    for (0..n_nodes) |i| {
        var buf: [16]u8 = undefined;
        const addr = std.fmt.bufPrint(&buf, "n{d}", .{i}) catch break;
        _ = try coord.registerNode(allocator, addr, base);
    }
    for (0..n_shards) |i| {
        try coord.assignShard(allocator, @intCast(i), @intCast(i % n_nodes));
    }
    try std.testing.expectEqual(n_nodes, @as(u32, @intCast(coord.nodes.items.len)));
    for (0..n_shards) |i| {
        try std.testing.expect(coord.getShardOwner(@intCast(i)) != null);
    }
    try coord.unassignShardsForNode(allocator, 0);
    var unassigned: u32 = 0;
    for (0..n_shards) |i| {
        if (coord.getShardOwner(@intCast(i)) == null) unassigned += 1;
    }
    try std.testing.expect(unassigned >= 1 and unassigned <= n_shards);
}

test {
    std.testing.refAllDecls(@This());
}
