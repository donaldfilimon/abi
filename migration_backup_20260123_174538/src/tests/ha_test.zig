//! High Availability Module Integration Tests
//!
//! Tests for the HA module components:
//! - HaManager integration
//! - ReplicationManager (replica management, heartbeats)
//! - BackupOrchestrator (full/incremental, retention)
//! - PitrManager (checkpoints, recovery points)

const std = @import("std");
const abi = @import("abi");
const ha = abi.ha;

// ============================================================================
// HaManager Integration Tests
// ============================================================================

test "HaManager initialization with default config" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
    try std.testing.expectEqual(@as(u32, 0), status.replica_count);
}

test "HaManager initialization with custom replication factor" {
    const allocator = std.testing.allocator;

    const config = ha.HaConfig{
        .replication_factor = 5,
        .auto_failover = false,
    };

    var manager = ha.HaManager.init(allocator, config);
    defer manager.deinit();

    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
}

// ============================================================================
// ReplicationManager Tests
// ============================================================================

test "ReplicationManager initialization" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try std.testing.expectEqual(@as(u32, 0), rm.getReplicaCount());
}

test "ReplicationManager add and remove replica" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    // Add a replica (node_id, region, address)
    try rm.addReplica(1, "primary", "127.0.0.1:5001");
    try std.testing.expectEqual(@as(u32, 1), rm.getReplicaCount());

    // Add another replica
    try rm.addReplica(2, "primary", "127.0.0.1:5002");
    try std.testing.expectEqual(@as(u32, 2), rm.getReplicaCount());

    // Remove a replica
    rm.removeReplica(1, .node_shutdown);
    try std.testing.expectEqual(@as(u32, 1), rm.getReplicaCount());
}

test "ReplicationManager max lag tracking" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    // Initially no lag (no replicas)
    try std.testing.expectEqual(@as(u64, 0), rm.getMaxLag());
}

// ============================================================================
// BackupOrchestrator Tests
// ============================================================================

test "BackupOrchestrator initialization" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    try std.testing.expectEqual(ha.backup.BackupState.idle, bo.getState());
}

test "BackupOrchestrator backup due check" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    // Check if backup is due (depends on interval config)
    _ = bo.isBackupDue();
}

test "BackupOrchestrator list backups" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const backups = bo.listBackups();
    try std.testing.expectEqual(@as(usize, 0), backups.len);
}

// ============================================================================
// PitrManager Tests
// ============================================================================

test "PitrManager initialization" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expectEqual(@as(u64, 0), pm.getCurrentSequence());
}

test "PitrManager recovery points" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    const points = pm.getRecoveryPoints();
    try std.testing.expectEqual(@as(usize, 0), points.len);
}

test "PitrManager create checkpoint" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    // Need to capture some operations before creating a checkpoint
    try pm.captureOperation(.insert, "key1", "value1", null);
    try pm.captureOperation(.insert, "key2", "value2", null);

    const checkpoint_id = try pm.createCheckpoint();
    try std.testing.expect(checkpoint_id > 0);

    // Should now have at least one recovery point
    const points = pm.getRecoveryPoints();
    try std.testing.expect(points.len > 0);
}

test "PitrManager find nearest recovery point" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    // Capture some operations and create a checkpoint
    try pm.captureOperation(.insert, "key1", "value1", null);
    const seq = try pm.createCheckpoint();
    try std.testing.expect(seq > 0);

    // Verify checkpoint was created
    const points = pm.getRecoveryPoints();
    try std.testing.expect(points.len > 0);

    // Find nearest recovery point - use a timestamp far in the future (year ~2100)
    // but not so large it could cause overflow issues
    const future_time: i64 = 4_000_000_000; // ~2096
    const point = pm.findNearestRecoveryPoint(future_time);
    try std.testing.expect(point != null);
}
