//! Replication state types, transitions, and status tracking.

const std = @import("std");

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
pub const ReplicaNode = struct {
    node_id: u64,
    region: []const u8,
    address: []const u8,
    state: NodeState,
    last_heartbeat: u64,
    replication_lag_ms: u64,
    sequence_number: u64,

    pub const NodeState = enum {
        connecting,
        syncing,
        active,
        lagging,
        disconnected,
    };
};

/// Entry in the replication queue for async fire-and-forget mode.
pub const QueueEntry = struct {
    key: []const u8,
    value: []const u8,
    sequence: u64,
    timestamp: u64,
    retry_count: u8,
};

/// Bounded circular buffer for replication queue entries.
pub const ReplicationQueue = struct {
    pub const capacity = 256;

    buffer: [capacity]QueueEntry = undefined,
    head: usize = 0,
    tail: usize = 0,
    len: usize = 0,
    dropped: u64 = 0,

    pub fn push(self: *ReplicationQueue, entry: QueueEntry) void {
        if (self.len == capacity) {
            // Drop oldest entry
            self.head = (self.head + 1) % capacity;
            self.len -= 1;
            self.dropped += 1;
        }
        self.buffer[self.tail] = entry;
        self.tail = (self.tail + 1) % capacity;
        self.len += 1;
    }

    pub fn pop(self: *ReplicationQueue) ?QueueEntry {
        if (self.len == 0) return null;
        const entry = self.buffer[self.head];
        self.head = (self.head + 1) % capacity;
        self.len -= 1;
        return entry;
    }

    pub fn isEmpty(self: *const ReplicationQueue) bool {
        return self.len == 0;
    }

    pub fn isFull(self: *const ReplicationQueue) bool {
        return self.len == capacity;
    }
};

test "ReplicationQueue push and pop" {
    var q = ReplicationQueue{};

    try std.testing.expect(q.isEmpty());
    try std.testing.expect(!q.isFull());

    q.push(.{ .key = "k1", .value = "v1", .sequence = 1, .timestamp = 100, .retry_count = 0 });
    q.push(.{ .key = "k2", .value = "v2", .sequence = 2, .timestamp = 101, .retry_count = 0 });

    try std.testing.expectEqual(@as(usize, 2), q.len);

    const first = q.pop().?;
    try std.testing.expectEqual(@as(u64, 1), first.sequence);

    const second = q.pop().?;
    try std.testing.expectEqual(@as(u64, 2), second.sequence);

    try std.testing.expect(q.pop() == null);
    try std.testing.expect(q.isEmpty());
}

test "ReplicationQueue overflow drops oldest" {
    var q = ReplicationQueue{};

    for (0..ReplicationQueue.capacity) |i| {
        q.push(.{ .key = "k", .value = "v", .sequence = @intCast(i), .timestamp = 0, .retry_count = 0 });
    }
    try std.testing.expect(q.isFull());
    try std.testing.expectEqual(@as(u64, 0), q.dropped);

    // Push one more, should drop oldest (sequence 0)
    q.push(.{ .key = "k", .value = "v", .sequence = 999, .timestamp = 0, .retry_count = 0 });
    try std.testing.expectEqual(@as(u64, 1), q.dropped);
    try std.testing.expectEqual(@as(usize, ReplicationQueue.capacity), q.len);

    // First entry should now be sequence 1 (0 was dropped)
    const first = q.pop().?;
    try std.testing.expectEqual(@as(u64, 1), first.sequence);
}

test {
    std.testing.refAllDecls(@This());
}
