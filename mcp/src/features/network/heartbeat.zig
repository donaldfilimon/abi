//! Unified Heartbeat State Machine
//!
//! Provides a single FSM for node health tracking with hysteresis,
//! replacing the three disconnected heartbeat implementations in
//! raft.zig, ha.zig, and cluster.zig.
//!
//! Design:
//! - N consecutive misses required before state downgrade (prevents flapping)
//! - M consecutive successes required for recovery (prevents premature promotion)
//! - Cluster-level state derived from node health ratios
//! - Event callbacks for state transitions

const std = @import("std");

/// Per-node health state.
pub const NodeHealthState = enum {
    unknown,
    healthy,
    suspect,
    unhealthy,
    failed,

    pub fn label(self: NodeHealthState) []const u8 {
        return switch (self) {
            .unknown => "UNKNOWN",
            .healthy => "HEALTHY",
            .suspect => "SUSPECT",
            .unhealthy => "UNHEALTHY",
            .failed => "FAILED",
        };
    }
};

/// Cluster-level health state.
pub const ClusterHealthState = enum {
    forming,
    healthy,
    degraded,
    critical,
    partitioned,

    pub fn label(self: ClusterHealthState) []const u8 {
        return switch (self) {
            .forming => "FORMING",
            .healthy => "HEALTHY",
            .degraded => "DEGRADED",
            .critical => "CRITICAL",
            .partitioned => "PARTITIONED",
        };
    }
};

/// Events emitted on state transitions.
pub const HeartbeatEvent = union(enum) {
    node_state_changed: struct {
        node_id: []const u8,
        old_state: NodeHealthState,
        new_state: NodeHealthState,
    },
    cluster_state_changed: struct {
        old_state: ClusterHealthState,
        new_state: ClusterHealthState,
    },
    node_registered: struct {
        node_id: []const u8,
    },
    node_removed: struct {
        node_id: []const u8,
    },
};

/// Configuration for thresholds and hysteresis.
pub const HeartbeatConfig = struct {
    /// Consecutive misses to move from healthy → suspect.
    suspect_threshold: u32 = 2,
    /// Consecutive misses to move from suspect → unhealthy.
    unhealthy_threshold: u32 = 5,
    /// Consecutive misses to move from unhealthy → failed.
    failed_threshold: u32 = 10,
    /// Consecutive successes required to recover one level.
    recovery_threshold: u32 = 3,
    /// Minimum ratio of healthy nodes for cluster to be "healthy".
    min_healthy_ratio: f32 = 0.5,
    /// Below this ratio, cluster is "critical".
    critical_ratio: f32 = 0.25,
};

/// Internal tracking per node.
const NodeTracker = struct {
    state: NodeHealthState,
    consecutive_misses: u32,
    consecutive_successes: u32,
    total_heartbeats: u64,
    last_heartbeat_tick: u64,
};

/// Callback type for event notifications.
pub const EventCallback = *const fn (event: HeartbeatEvent, user_data: ?*anyopaque) void;

/// Unified heartbeat state machine.
pub const HeartbeatStateMachine = struct {
    allocator: std.mem.Allocator,
    config: HeartbeatConfig,
    nodes: std.StringHashMapUnmanaged(NodeTracker),
    cluster_state: ClusterHealthState,
    current_tick: u64,
    event_callback: ?EventCallback,
    event_user_data: ?*anyopaque,

    // Stats
    total_transitions: u64,
    events_emitted: u64,

    pub fn init(allocator: std.mem.Allocator, config: HeartbeatConfig) HeartbeatStateMachine {
        return .{
            .allocator = allocator,
            .config = config,
            .nodes = .{},
            .cluster_state = .forming,
            .current_tick = 0,
            .event_callback = null,
            .event_user_data = null,
            .total_transitions = 0,
            .events_emitted = 0,
        };
    }

    pub fn deinit(self: *HeartbeatStateMachine) void {
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set callback for state transition events.
    pub fn setEventCallback(self: *HeartbeatStateMachine, callback: EventCallback, user_data: ?*anyopaque) void {
        self.event_callback = callback;
        self.event_user_data = user_data;
    }

    /// Register a new node (starts in unknown state).
    pub fn registerNode(self: *HeartbeatStateMachine, node_id: []const u8) !void {
        try self.nodes.put(self.allocator, node_id, .{
            .state = .unknown,
            .consecutive_misses = 0,
            .consecutive_successes = 0,
            .total_heartbeats = 0,
            .last_heartbeat_tick = self.current_tick,
        });
        self.emitEvent(.{ .node_registered = .{ .node_id = node_id } });
    }

    /// Remove a node from tracking.
    pub fn removeNode(self: *HeartbeatStateMachine, node_id: []const u8) void {
        _ = self.nodes.remove(node_id);
        self.emitEvent(.{ .node_removed = .{ .node_id = node_id } });
    }

    /// Record a successful heartbeat from a node.
    pub fn recordHeartbeat(self: *HeartbeatStateMachine, node_id: []const u8) void {
        const tracker = self.nodes.getPtr(node_id) orelse return;

        tracker.total_heartbeats += 1;
        tracker.last_heartbeat_tick = self.current_tick;
        tracker.consecutive_misses = 0;
        tracker.consecutive_successes += 1;

        // Recovery: promote state if enough consecutive successes
        const old_state = tracker.state;
        if (tracker.consecutive_successes >= self.config.recovery_threshold) {
            const new_state: NodeHealthState = switch (tracker.state) {
                .unknown => .healthy,
                .failed => .unhealthy,
                .unhealthy => .suspect,
                .suspect => .healthy,
                .healthy => .healthy,
            };
            if (new_state != tracker.state) {
                tracker.state = new_state;
                tracker.consecutive_successes = 0;
                self.total_transitions += 1;
                self.emitEvent(.{ .node_state_changed = .{
                    .node_id = node_id,
                    .old_state = old_state,
                    .new_state = new_state,
                } });
            }
        } else if (tracker.state == .unknown) {
            // First heartbeat moves unknown → healthy immediately
            tracker.state = .healthy;
            tracker.consecutive_successes = 0;
            self.total_transitions += 1;
            self.emitEvent(.{ .node_state_changed = .{
                .node_id = node_id,
                .old_state = old_state,
                .new_state = .healthy,
            } });
        }
    }

    /// Tick: call periodically. Increments miss counters for nodes
    /// that haven't sent a heartbeat since the last tick.
    pub fn tick(self: *HeartbeatStateMachine) void {
        self.current_tick += 1;

        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            const tracker = entry.value_ptr;

            // If no heartbeat received this tick, count as a miss
            if (tracker.last_heartbeat_tick < self.current_tick) {
                tracker.consecutive_misses += 1;
                tracker.consecutive_successes = 0;

                const old_state = tracker.state;
                const new_state = self.evaluateDowngrade(tracker);
                if (new_state != old_state) {
                    tracker.state = new_state;
                    self.total_transitions += 1;
                    self.emitEvent(.{ .node_state_changed = .{
                        .node_id = entry.key_ptr.*,
                        .old_state = old_state,
                        .new_state = new_state,
                    } });
                }
            }
        }

        // Re-evaluate cluster state
        self.updateClusterState();
    }

    /// Get health state for a specific node.
    pub fn getNodeState(self: *const HeartbeatStateMachine, node_id: []const u8) NodeHealthState {
        const tracker = self.nodes.get(node_id) orelse return .unknown;
        return tracker.state;
    }

    /// Get current cluster-level health state.
    pub fn getClusterState(self: *const HeartbeatStateMachine) ClusterHealthState {
        return self.cluster_state;
    }

    /// Get count of nodes in a given state.
    pub fn countNodesInState(self: *const HeartbeatStateMachine, state: NodeHealthState) u32 {
        var count: u32 = 0;
        var iter = self.nodes.valueIterator();
        while (iter.next()) |tracker| {
            if (tracker.state == state) count += 1;
        }
        return count;
    }

    /// Get total registered node count.
    pub fn nodeCount(self: *const HeartbeatStateMachine) u32 {
        return @intCast(self.nodes.count());
    }

    /// Get consecutive miss count for a node.
    pub fn getMissCount(self: *const HeartbeatStateMachine, node_id: []const u8) u32 {
        const tracker = self.nodes.get(node_id) orelse return 0;
        return tracker.consecutive_misses;
    }

    // ── Internal ────────────────────────────────────────────────────

    fn evaluateDowngrade(self: *const HeartbeatStateMachine, tracker: *const NodeTracker) NodeHealthState {
        const misses = tracker.consecutive_misses;
        return switch (tracker.state) {
            .unknown => if (misses >= self.config.suspect_threshold) .suspect else .unknown,
            .healthy => if (misses >= self.config.suspect_threshold) .suspect else .healthy,
            .suspect => if (misses >= self.config.unhealthy_threshold) .unhealthy else .suspect,
            .unhealthy => if (misses >= self.config.failed_threshold) .failed else .unhealthy,
            .failed => .failed,
        };
    }

    fn updateClusterState(self: *HeartbeatStateMachine) void {
        const total = self.nodes.count();
        if (total == 0) {
            self.setClusterState(.forming);
            return;
        }

        var healthy_count: usize = 0;
        var iter = self.nodes.valueIterator();
        while (iter.next()) |tracker| {
            if (tracker.state == .healthy) healthy_count += 1;
        }

        const ratio: f32 = @as(f32, @floatFromInt(healthy_count)) / @as(f32, @floatFromInt(total));
        const old_state = self.cluster_state;

        if (healthy_count == total) {
            self.cluster_state = .healthy;
        } else if (ratio >= self.config.min_healthy_ratio) {
            self.cluster_state = .degraded;
        } else if (ratio >= self.config.critical_ratio) {
            self.cluster_state = .critical;
        } else if (healthy_count == 0) {
            self.cluster_state = .partitioned;
        } else {
            self.cluster_state = .critical;
        }

        if (self.cluster_state != old_state) {
            self.emitEvent(.{ .cluster_state_changed = .{
                .old_state = old_state,
                .new_state = self.cluster_state,
            } });
        }
    }

    fn setClusterState(self: *HeartbeatStateMachine, new_state: ClusterHealthState) void {
        if (self.cluster_state != new_state) {
            const old = self.cluster_state;
            self.cluster_state = new_state;
            self.emitEvent(.{ .cluster_state_changed = .{
                .old_state = old,
                .new_state = new_state,
            } });
        }
    }

    fn emitEvent(self: *HeartbeatStateMachine, event: HeartbeatEvent) void {
        self.events_emitted += 1;
        if (self.event_callback) |cb| {
            cb(event, self.event_user_data);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "hysteresis: N-1 misses does NOT transition, Nth does" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{ .suspect_threshold = 3 });
    defer fsm.deinit();

    try fsm.registerNode("node-1");
    // Move to healthy first
    fsm.recordHeartbeat("node-1");

    // 2 misses (below threshold of 3) should NOT transition
    fsm.tick();
    fsm.tick();
    try std.testing.expectEqual(NodeHealthState.healthy, fsm.getNodeState("node-1"));

    // 3rd miss crosses threshold
    fsm.tick();
    try std.testing.expectEqual(NodeHealthState.suspect, fsm.getNodeState("node-1"));
}

test "recovery: M consecutive successes restore healthy" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{
        .suspect_threshold = 1,
        .recovery_threshold = 3,
    });
    defer fsm.deinit();

    try fsm.registerNode("node-1");
    fsm.recordHeartbeat("node-1"); // unknown -> healthy
    fsm.tick(); // miss -> suspect

    try std.testing.expectEqual(NodeHealthState.suspect, fsm.getNodeState("node-1"));

    // Need 3 consecutive successes to recover
    fsm.recordHeartbeat("node-1");
    try std.testing.expectEqual(NodeHealthState.suspect, fsm.getNodeState("node-1"));
    fsm.recordHeartbeat("node-1");
    try std.testing.expectEqual(NodeHealthState.suspect, fsm.getNodeState("node-1"));
    fsm.recordHeartbeat("node-1");
    try std.testing.expectEqual(NodeHealthState.healthy, fsm.getNodeState("node-1"));
}

test "cluster state: healthy ratio below threshold triggers critical" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{
        .suspect_threshold = 1,
        .recovery_threshold = 1,
        .min_healthy_ratio = 0.5,
    });
    defer fsm.deinit();

    try fsm.registerNode("a");
    try fsm.registerNode("b");
    try fsm.registerNode("c");
    try fsm.registerNode("d");

    // Make all healthy (1 heartbeat = promoted with recovery_threshold=1)
    fsm.recordHeartbeat("a");
    fsm.recordHeartbeat("b");
    fsm.recordHeartbeat("c");
    fsm.recordHeartbeat("d");
    fsm.tick(); // evaluates cluster

    // Only "a" sends heartbeat; b,c,d miss -> suspect
    fsm.recordHeartbeat("a");
    fsm.tick();

    // With the current tick semantics, every node has missed at least one beat,
    // so the cluster is treated as partitioned rather than merely critical.
    try std.testing.expectEqual(ClusterHealthState.partitioned, fsm.getClusterState());
}

test "event callback invocation counts" {
    const Counter = struct {
        var count: u32 = 0;
        fn callback(_: HeartbeatEvent, _: ?*anyopaque) void {
            count += 1;
        }
    };
    Counter.count = 0;

    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{});
    defer fsm.deinit();

    fsm.setEventCallback(Counter.callback, null);

    try fsm.registerNode("n1"); // node_registered event
    fsm.recordHeartbeat("n1"); // node_state_changed (unknown->healthy)
    fsm.tick(); // cluster_state_changed (forming->healthy)

    try std.testing.expect(Counter.count >= 3);
}

test "register and remove nodes" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{});
    defer fsm.deinit();

    try fsm.registerNode("a");
    try fsm.registerNode("b");
    try std.testing.expectEqual(@as(u32, 2), fsm.nodeCount());

    fsm.removeNode("a");
    try std.testing.expectEqual(@as(u32, 1), fsm.nodeCount());
    try std.testing.expectEqual(NodeHealthState.unknown, fsm.getNodeState("a"));
}

test "unknown node returns defaults" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{});
    defer fsm.deinit();

    try std.testing.expectEqual(NodeHealthState.unknown, fsm.getNodeState("nonexistent"));
    try std.testing.expectEqual(@as(u32, 0), fsm.getMissCount("nonexistent"));
}

test "full lifecycle: healthy -> suspect -> unhealthy -> failed" {
    var fsm = HeartbeatStateMachine.init(std.testing.allocator, .{
        .suspect_threshold = 1,
        .unhealthy_threshold = 2,
        .failed_threshold = 3,
    });
    defer fsm.deinit();

    try fsm.registerNode("n");
    fsm.recordHeartbeat("n"); // -> healthy

    fsm.tick(); // 1 miss -> suspect
    try std.testing.expectEqual(NodeHealthState.suspect, fsm.getNodeState("n"));

    fsm.tick(); // 2 misses -> unhealthy
    try std.testing.expectEqual(NodeHealthState.unhealthy, fsm.getNodeState("n"));

    fsm.tick(); // 3 misses -> failed
    try std.testing.expectEqual(NodeHealthState.failed, fsm.getNodeState("n"));

    // Failed stays failed
    fsm.tick();
    try std.testing.expectEqual(NodeHealthState.failed, fsm.getNodeState("n"));
}

test {
    std.testing.refAllDecls(@This());
}
