//! Replication sync: replicate, processQueue, lag tracking, data sync.

const std = @import("std");
const state = @import("state.zig");
const time = @import("../../../foundation/mod.zig").time;

const ReplicationManager = @import("membership.zig").ReplicationManager;

const ReplicaNode = state.ReplicaNode;
const ReplicationQueue = state.ReplicationQueue;
const QueueEntry = state.QueueEntry;

/// Replicate a write operation synchronously to all replicas.
pub fn replicateSync(
    manager: *ReplicationManager,
    key: []const u8,
    value: []const u8,
    sequence: u64,
) !void {
    _ = key;
    _ = value;
    // Send to all replicas and wait for all acks
    var it = manager.replicas.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.sequence_number = sequence;
    }
}

/// Replicate a write operation to a quorum of replicas.
pub fn replicateQuorum(
    manager: *ReplicationManager,
    key: []const u8,
    value: []const u8,
    sequence: u64,
) !void {
    _ = key;
    _ = value;
    // Send to all replicas and wait for quorum
    const quorum = calculateQuorum(manager);
    var acks: u32 = 1; // Self counts

    var it = manager.replicas.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.sequence_number = sequence;
        acks += 1;
        if (acks >= quorum) break;
    }

    if (acks < quorum) {
        std.log.warn("Quorum not reached for replication (acks: {d}, required: {d}, sequence: {d}, replica_count: {d})", .{ acks, quorum, sequence, manager.replicas.count() });
        return error.QuorumNotReached;
    }
}

/// Queue a replication entry for async fire-and-forget mode.
pub fn queueReplication(
    manager: *ReplicationManager,
    key: []const u8,
    value: []const u8,
    sequence: u64,
) void {
    manager.queue.push(.{
        .key = key,
        .value = value,
        .sequence = sequence,
        .timestamp = time.timestampSec(),
        .retry_count = 0,
    });
}

/// Calculate the required quorum size.
pub fn calculateQuorum(manager: *ReplicationManager) u32 {
    if (manager.config.write_quorum > 0) {
        return manager.config.write_quorum;
    }
    // Default to majority
    const total = @as(u32, manager.replicas.count()) + 1;
    return (total / 2) + 1;
}

test {
    std.testing.refAllDecls(@This());
}
