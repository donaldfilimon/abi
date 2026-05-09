//! Replica membership: addReplica, removeReplica, failoverTo, heartbeat, state management.

const std = @import("std");
const state = @import("state.zig");
const sync_impl = @import("sync.zig");
const time = @import("../../../foundation/mod.zig").time;

const sync = @import("../../../foundation/mod.zig").sync;
const Mutex = sync.Mutex;

const ReplicationConfig = state.ReplicationConfig;
const ReplicationState = state.ReplicationState;
const ReplicationEvent = state.ReplicationEvent;
const DisconnectReason = state.DisconnectReason;
const ReplicaNode = state.ReplicaNode;
const ReplicationQueue = state.ReplicationQueue;

/// Replication manager
pub const ReplicationManager = struct {
    allocator: std.mem.Allocator,
    config: ReplicationConfig,

    // Node tracking
    local_node_id: u64,
    is_leader: bool,
    leader_node_id: u64,
    replicas: std.AutoHashMapUnmanaged(u64, ReplicaNode),

    // Replication state
    state: ReplicationState,
    current_sequence: u64,

    // Async replication queue
    queue: ReplicationQueue,

    // Synchronization
    mutex: Mutex,

    /// Initialize the replication manager
    pub fn init(allocator: std.mem.Allocator, config: ReplicationConfig) ReplicationManager {
        const seed = time.timestampNs();
        var prng = std.Random.DefaultPrng.init(seed);

        return .{
            .allocator = allocator,
            .config = config,
            .local_node_id = prng.random().int(u64),
            .is_leader = true, // Start as leader until election
            .leader_node_id = 0,
            .replicas = .{},
            .state = .initializing,
            .current_sequence = 0,
            .queue = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the replication manager
    pub fn deinit(self: *ReplicationManager) void {
        self.replicas.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a replica node
    pub fn addReplica(
        self: *ReplicationManager,
        node_id: u64,
        region: []const u8,
        address: []const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const node = ReplicaNode{
            .node_id = node_id,
            .region = region,
            .address = address,
            .state = .connecting,
            .last_heartbeat = time.timestampSec(),
            .replication_lag_ms = 0,
            .sequence_number = 0,
        };

        try self.replicas.put(self.allocator, node_id, node);

        self.emitEvent(.{ .replica_connected = .{
            .node_id = node_id,
            .region = region,
        } });

        self.updateState();
    }

    /// Remove a replica node
    pub fn removeReplica(self: *ReplicationManager, node_id: u64, reason: DisconnectReason) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        _ = self.replicas.remove(node_id);

        self.emitEvent(.{ .replica_disconnected = .{
            .node_id = node_id,
            .reason = reason,
        } });

        self.updateState();
    }

    /// Get current replication state
    pub fn getState(self: *ReplicationManager) ReplicationState {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.state;
    }

    /// Check if this node is the leader
    pub fn isLeader(self: *ReplicationManager) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.is_leader;
    }

    /// Get the current replication sequence number
    pub fn getCurrentSequence(self: *ReplicationManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_sequence;
    }

    /// Get the leader node ID
    pub fn getLeaderNodeId(self: *ReplicationManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.leader_node_id;
    }

    /// Get replica count
    pub fn getReplicaCount(self: *ReplicationManager) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return @as(u32, self.replicas.count());
    }

    /// Get maximum replication lag
    pub fn getMaxLag(self: *ReplicationManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var max_lag: u64 = 0;
        var it = self.replicas.valueIterator();
        while (it.next()) |node| {
            if (node.replication_lag_ms > max_lag) {
                max_lag = node.replication_lag_ms;
            }
        }
        return max_lag;
    }

    /// Replicate a write operation
    pub fn replicate(
        self: *ReplicationManager,
        key: []const u8,
        value: []const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.is_leader) {
            std.log.warn("Rejecting write: node is not leader (local_node_id: {d}, current_leader: {d})", .{ self.local_node_id, self.leader_node_id });
            return error.NotLeader;
        }

        self.current_sequence += 1;
        const sequence = self.current_sequence;

        // In real implementation, send to replicas based on mode
        switch (self.config.mode) {
            .sync => {
                // Wait for all replicas
                try sync_impl.replicateSync(self, key, value, sequence);
            },
            .async_fire_forget => {
                // Queue for async replication
                sync_impl.queueReplication(self, key, value, sequence);
            },
            .async_with_ack => {
                // Wait for quorum acknowledgment
                try sync_impl.replicateQuorum(self, key, value, sequence);
            },
        }
    }

    /// Drain the replication queue and apply entries to all replicas.
    /// Returns the number of entries processed.
    pub fn processQueue(self: *ReplicationManager) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var processed: u64 = 0;
        while (self.queue.pop()) |entry| {
            var it = self.replicas.iterator();
            while (it.next()) |replica| {
                if (replica.value_ptr.state != .disconnected) {
                    if (entry.sequence > replica.value_ptr.sequence_number) {
                        replica.value_ptr.sequence_number = entry.sequence;
                    }
                }
            }
            processed += 1;
        }
        return processed;
    }

    /// Promote a replica to primary
    pub fn promoteToPrimary(self: *ReplicationManager, node_id: u64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (node_id == self.local_node_id) {
            self.is_leader = true;
            self.leader_node_id = node_id;
        } else {
            self.is_leader = false;
            self.leader_node_id = node_id;
        }

        self.emitEvent(.{ .leader_elected = .{ .node_id = node_id } });
    }

    /// Process heartbeat from replica
    pub fn processHeartbeat(self: *ReplicationManager, node_id: u64, sequence: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.replicas.getPtr(node_id)) |node| {
            node.last_heartbeat = time.timestampSec();
            node.sequence_number = sequence;

            // Calculate lag
            if (self.current_sequence > sequence) {
                node.replication_lag_ms = (self.current_sequence - sequence) * 10; // Estimate
                if (node.replication_lag_ms > self.config.max_lag_ms) {
                    node.state = .lagging;
                    self.emitEvent(.{ .replication_lag = .{
                        .node_id = node_id,
                        .lag_ms = node.replication_lag_ms,
                    } });
                } else {
                    node.state = .active;
                }
            } else {
                node.replication_lag_ms = 0;
                node.state = .active;
            }
        }
    }

    fn updateState(self: *ReplicationManager) void {
        const count = self.replicas.count();
        const quorum = sync_impl.calculateQuorum(self);
        const prev_state = self.state;

        if (count == 0) {
            // No replicas connected
            if (prev_state == .degraded or prev_state == .healthy or prev_state == .syncing) {
                self.state = .failed;
            } else if (prev_state == .initializing) {
                // Stay initializing until at least one replica connects
                self.state = .initializing;
            }
            // .failed stays .failed
            return;
        }

        // Count active (non-lagging) replicas
        var caught_up: u32 = 0;
        var it = self.replicas.valueIterator();
        while (it.next()) |node| {
            if (node.replication_lag_ms <= self.config.max_lag_ms and node.state != .disconnected) {
                caught_up += 1;
            }
        }

        // +1 for self (leader)
        const total_caught_up = caught_up + 1;

        if (prev_state == .initializing) {
            self.state = .syncing;
        } else if (total_caught_up >= quorum) {
            if (prev_state != .healthy) {
                if (prev_state == .degraded) {
                    self.emitEvent(.quorum_restored);
                }
            }
            self.state = .healthy;
        } else {
            // Have replicas but below quorum
            if (prev_state == .healthy) {
                self.emitEvent(.quorum_lost);
            }
            self.state = .degraded;
        }
    }

    fn emitEvent(self: *ReplicationManager, event: ReplicationEvent) void {
        if (self.config.on_event) |callback| {
            callback(event);
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "ReplicationManager initialization" {
    const allocator = std.testing.allocator;

    var manager = ReplicationManager.init(allocator, .{
        .replication_factor = 3,
    });
    defer manager.deinit();

    try std.testing.expectEqual(ReplicationState.initializing, manager.state);
    try std.testing.expect(manager.is_leader);
}

test "ReplicationManager add/remove replica" {
    const allocator = std.testing.allocator;

    var manager = ReplicationManager.init(allocator, .{});
    defer manager.deinit();

    try manager.addReplica(12345, "us-east-1", "10.0.0.2:5432");
    try std.testing.expectEqual(@as(u32, 1), manager.getReplicaCount());

    manager.removeReplica(12345, .node_shutdown);
    try std.testing.expectEqual(@as(u32, 0), manager.getReplicaCount());
}

test "queueReplication and processQueue" {
    const allocator = std.testing.allocator;

    var manager = ReplicationManager.init(allocator, .{ .mode = .async_fire_forget });
    defer manager.deinit();

    try manager.addReplica(100, "us-east-1", "10.0.0.2:5432");

    // Replicate puts entries in the queue
    try manager.replicate("key1", "val1");
    try manager.replicate("key2", "val2");
    try std.testing.expectEqual(@as(usize, 2), manager.queue.len);

    // processQueue drains and applies to replicas
    const processed = manager.processQueue();
    try std.testing.expectEqual(@as(u64, 2), processed);
    try std.testing.expect(manager.queue.isEmpty());

    // Replica should have been advanced
    if (manager.replicas.getPtr(100)) |replica| {
        try std.testing.expectEqual(@as(u64, 2), replica.sequence_number);
    }
}

test "updateState transitions" {
    const allocator = std.testing.allocator;

    var manager = ReplicationManager.init(allocator, .{ .replication_factor = 3, .write_quorum = 2 });
    defer manager.deinit();

    // Starts as initializing
    try std.testing.expectEqual(ReplicationState.initializing, manager.state);

    // Adding first replica: initializing -> syncing
    try manager.addReplica(1, "us-east-1", "10.0.0.1:5432");
    try std.testing.expectEqual(ReplicationState.syncing, manager.state);

    // Adding second replica with quorum=2: syncing -> healthy (self + 2 replicas >= 2)
    try manager.addReplica(2, "us-west-2", "10.0.0.2:5432");
    try std.testing.expectEqual(ReplicationState.healthy, manager.state);

    // Remove all replicas: healthy -> degraded -> failed
    manager.removeReplica(2, .node_shutdown);
    manager.removeReplica(1, .node_shutdown);
    try std.testing.expectEqual(ReplicationState.failed, manager.state);
}

test "updateState degraded when below quorum" {
    const allocator = std.testing.allocator;

    // write_quorum=3 means we need 3 caught-up nodes (including self)
    var manager = ReplicationManager.init(allocator, .{ .write_quorum = 3 });
    defer manager.deinit();

    // Add one replica: initializing -> syncing
    try manager.addReplica(1, "us-east-1", "10.0.0.1:5432");
    try std.testing.expectEqual(ReplicationState.syncing, manager.state);

    // Simulate a heartbeat to make the replica active, then re-check
    manager.processHeartbeat(1, manager.current_sequence);
    // processHeartbeat doesn't call updateState, so manually transition via addReplica
    try manager.addReplica(2, "us-west-2", "10.0.0.2:5432");
    // Now self + 2 replicas = 3 >= quorum of 3 -> healthy
    try std.testing.expectEqual(ReplicationState.healthy, manager.state);

    // Remove one -> self + 1 = 2 < 3 -> degraded
    manager.removeReplica(2, .timeout);
    try std.testing.expectEqual(ReplicationState.degraded, manager.state);
}

test {
    std.testing.refAllDecls(@This());
}
