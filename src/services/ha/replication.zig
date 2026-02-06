//! Multi-Region Replication Manager
//!
//! Handles data replication across multiple nodes and regions:
//! - Synchronous and asynchronous replication modes
//! - Automatic leader election
//! - Conflict resolution
//! - Lag monitoring

const std = @import("std");
const time = @import("../shared/utils.zig");
const platform_time = @import("../shared/time.zig");

// Zig 0.16 compatibility: Simple spinlock Mutex
const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

/// Replication configuration
pub const ReplicationConfig = struct {
    /// Number of replicas to maintain
    replication_factor: u8 = 3,
    /// Replication mode
    mode: ReplicationMode = .async_with_ack,
    /// Maximum replication lag before warning (ms)
    max_lag_ms: u64 = 5000,
    /// Quorum size for writes (0 = majority)
    write_quorum: u8 = 0,
    /// Timeout for replica acknowledgment (ms)
    ack_timeout_ms: u64 = 1000,
    /// Enable automatic failover
    auto_failover: bool = true,
    /// Heartbeat interval (ms)
    heartbeat_interval_ms: u64 = 1000,
    /// Callback for replication events
    on_event: ?*const fn (ReplicationEvent) void = null,
};

/// Replication mode
pub const ReplicationMode = enum {
    /// Synchronous - wait for all replicas
    sync,
    /// Asynchronous - fire and forget
    async_fire_forget,
    /// Async with acknowledgment from quorum
    async_with_ack,
};

/// Replication state
pub const ReplicationState = enum {
    initializing,
    syncing,
    healthy,
    degraded,
    failed,
};

/// Replication events
pub const ReplicationEvent = union(enum) {
    replica_connected: struct { node_id: u64, region: []const u8 },
    replica_disconnected: struct { node_id: u64, reason: DisconnectReason },
    replication_lag: struct { node_id: u64, lag_ms: u64 },
    quorum_lost: void,
    quorum_restored: void,
    leader_elected: struct { node_id: u64 },
    conflict_detected: struct { key: []const u8, resolution: ConflictResolution },
};

/// Disconnect reasons
pub const DisconnectReason = enum {
    timeout,
    network_error,
    node_shutdown,
    kicked,
};

/// Conflict resolution strategies
pub const ConflictResolution = enum {
    last_write_wins,
    first_write_wins,
    manual_resolution_required,
};

/// Replica node information
const ReplicaNode = struct {
    node_id: u64,
    region: []const u8,
    address: []const u8,
    state: NodeState,
    last_heartbeat: u64,
    replication_lag_ms: u64,
    sequence_number: u64,

    const NodeState = enum {
        connecting,
        syncing,
        active,
        lagging,
        disconnected,
    };
};

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

    // Synchronization
    mutex: Mutex,

    /// Initialize the replication manager
    pub fn init(allocator: std.mem.Allocator, config: ReplicationConfig) ReplicationManager {
        const seed = platform_time.timestampNs();
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
            .last_heartbeat = time.time.timestampSec(),
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
                try self.replicateSync(key, value, sequence);
            },
            .async_fire_forget => {
                // Queue for async replication
                self.queueReplication(key, value, sequence);
            },
            .async_with_ack => {
                // Wait for quorum acknowledgment
                try self.replicateQuorum(key, value, sequence);
            },
        }
    }

    fn replicateSync(
        self: *ReplicationManager,
        key: []const u8,
        value: []const u8,
        sequence: u64,
    ) !void {
        _ = key;
        _ = value;
        // Send to all replicas and wait for all acks
        var it = self.replicas.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.sequence_number = sequence;
        }
    }

    fn replicateQuorum(
        self: *ReplicationManager,
        key: []const u8,
        value: []const u8,
        sequence: u64,
    ) !void {
        _ = key;
        _ = value;
        // Send to all replicas and wait for quorum
        const quorum = self.calculateQuorum();
        var acks: u32 = 1; // Self counts

        var it = self.replicas.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.sequence_number = sequence;
            acks += 1;
            if (acks >= quorum) break;
        }

        if (acks < quorum) {
            std.log.warn("Quorum not reached for replication (acks: {d}, required: {d}, sequence: {d}, replica_count: {d})", .{ acks, quorum, sequence, self.replicas.count() });
            return error.QuorumNotReached;
        }
    }

    fn queueReplication(
        self: *ReplicationManager,
        key: []const u8,
        value: []const u8,
        sequence: u64,
    ) void {
        _ = self;
        _ = key;
        _ = value;
        _ = sequence;
        // In real implementation, add to replication queue
    }

    fn calculateQuorum(self: *ReplicationManager) u32 {
        if (self.config.write_quorum > 0) {
            return self.config.write_quorum;
        }
        // Default to majority
        const total = @as(u32, self.replicas.count()) + 1;
        return (total / 2) + 1;
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
            node.last_heartbeat = time.time.timestampSec();
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
        const quorum = self.calculateQuorum();

        if (count + 1 >= quorum) {
            self.state = .healthy;
        } else if (count > 0) {
            self.state = .degraded;
            self.emitEvent(.quorum_lost);
        } else {
            self.state = .degraded;
        }
    }

    fn emitEvent(self: *ReplicationManager, event: ReplicationEvent) void {
        if (self.config.on_event) |callback| {
            callback(event);
        }
    }
};

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
