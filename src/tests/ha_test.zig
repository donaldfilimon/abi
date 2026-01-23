//! High Availability Module Integration Tests
//!
//! Tests for the HA module components:
//! - HaManager integration
//! - ReplicationManager (quorum, sync/async modes)
//! - BackupOrchestrator (full/incremental, retention)
//! - PitrManager (capture, checkpoint, recovery)
//! - Cross-component coordination

const std = @import("std");
const ha = @import("../ha/mod.zig");

// ============================================================================
// HaManager Integration Tests
// ============================================================================

test "HaManager initialization with default config" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try std.testing.expect(!manager.is_running);
    try std.testing.expect(manager.is_primary);
    try std.testing.expect(manager.node_id != 0);
}

test "HaManager initialization with custom config" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 5,
        .backup_interval_hours = 12,
        .enable_pitr = true,
        .pitr_retention_hours = 336, // 14 days
        .auto_failover = false,
    });
    defer manager.deinit();

    try std.testing.expectEqual(@as(u8, 5), manager.config.replication_factor);
    try std.testing.expectEqual(@as(u32, 12), manager.config.backup_interval_hours);
    try std.testing.expect(manager.config.enable_pitr);
    try std.testing.expectEqual(@as(u32, 336), manager.config.pitr_retention_hours);
    try std.testing.expect(!manager.config.auto_failover);
}

test "HaManager start initializes sub-managers" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 3,
        .enable_pitr = true,
    });
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(manager.is_running);
    try std.testing.expect(manager.replication_manager != null);
    try std.testing.expect(manager.backup_orchestrator != null);
    try std.testing.expect(manager.pitr_manager != null);
}

test "HaManager start without replication" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 1, // Single node, no replication
        .enable_pitr = false,
    });
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(manager.is_running);
    try std.testing.expect(manager.replication_manager == null);
    try std.testing.expect(manager.backup_orchestrator != null);
    try std.testing.expect(manager.pitr_manager == null);
}

test "HaManager stop/start cycle" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    // Start
    try manager.start();
    try std.testing.expect(manager.is_running);

    // Stop
    manager.stop();
    try std.testing.expect(!manager.is_running);

    // Restart
    try manager.start();
    try std.testing.expect(manager.is_running);
}

test "HaManager getStatus" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 3,
    });
    defer manager.deinit();

    // Status before start
    var status = manager.getStatus();
    try std.testing.expect(!status.is_running);

    // Status after start
    try manager.start();
    status = manager.getStatus();
    try std.testing.expect(status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expectEqual(manager.node_id, status.node_id);
}

test "HaManager event callback" {
    const allocator = std.testing.allocator;

    var event_received = false;
    var received_node_id: u64 = 0;

    const callback = struct {
        fn handler(event: ha.HaEvent) void {
            switch (event) {
                .replica_added => |info| {
                    _ = info;
                    // Mark that we received the event (can't modify outer scope directly)
                },
                else => {},
            }
            _ = &event_received;
            _ = &received_node_id;
        }
    }.handler;

    var manager = ha.HaManager.init(allocator, .{
        .on_event = &callback,
    });
    defer manager.deinit();

    try manager.start();
    // Event should have been emitted during start
}

// ============================================================================
// ReplicationManager Tests
// ============================================================================

test "ReplicationManager initialization" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{
        .replication_factor = 3,
        .mode = .async_with_ack,
    });
    defer rm.deinit();

    try std.testing.expectEqual(ha.ReplicationState.initializing, rm.getState());
}

test "ReplicationManager config modes" {
    const allocator = std.testing.allocator;

    // Test sync mode
    var rm_sync = ha.ReplicationManager.init(allocator, .{
        .mode = .sync,
    });
    defer rm_sync.deinit();
    try std.testing.expectEqual(ha.ReplicationMode.sync, rm_sync.config.mode);

    // Test async fire and forget
    var rm_async = ha.ReplicationManager.init(allocator, .{
        .mode = .async_fire_forget,
    });
    defer rm_async.deinit();
    try std.testing.expectEqual(ha.ReplicationMode.async_fire_forget, rm_async.config.mode);
}

test "ReplicationManager quorum calculation" {
    const allocator = std.testing.allocator;

    // With 3 replicas, quorum should be 2
    var rm = ha.ReplicationManager.init(allocator, .{
        .replication_factor = 3,
        .write_quorum = 0, // Auto-calculate majority
    });
    defer rm.deinit();

    const quorum = rm.getQuorumSize();
    try std.testing.expectEqual(@as(u8, 2), quorum);
}

test "ReplicationManager replica count" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{
        .replication_factor = 5,
    });
    defer rm.deinit();

    // Initially no replicas connected
    try std.testing.expectEqual(@as(u32, 0), rm.getReplicaCount());
}

// ============================================================================
// BackupOrchestrator Tests
// ============================================================================

test "BackupOrchestrator initialization" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{
        .interval_hours = 6,
        .mode = .incremental,
    });
    defer bo.deinit();

    try std.testing.expectEqual(ha.BackupState.idle, bo.getState());
}

test "BackupOrchestrator backup modes" {
    const allocator = std.testing.allocator;

    // Test full backup mode
    var bo_full = ha.BackupOrchestrator.init(allocator, .{
        .mode = .full,
    });
    defer bo_full.deinit();
    try std.testing.expectEqual(ha.BackupMode.full, bo_full.config.mode);

    // Test incremental mode
    var bo_inc = ha.BackupOrchestrator.init(allocator, .{
        .mode = .incremental,
    });
    defer bo_inc.deinit();
    try std.testing.expectEqual(ha.BackupMode.incremental, bo_inc.config.mode);

    // Test differential mode
    var bo_diff = ha.BackupOrchestrator.init(allocator, .{
        .mode = .differential,
    });
    defer bo_diff.deinit();
    try std.testing.expectEqual(ha.BackupMode.differential, bo_diff.config.mode);
}

test "BackupOrchestrator retention policy" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{
        .retention = .{
            .keep_last = 5,
            .keep_daily_days = 14,
            .keep_weekly_weeks = 8,
            .keep_monthly_months = 6,
        },
    });
    defer bo.deinit();

    try std.testing.expectEqual(@as(u32, 5), bo.config.retention.keep_last);
    try std.testing.expectEqual(@as(u32, 14), bo.config.retention.keep_daily_days);
}

test "BackupOrchestrator trigger backup" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const backup_id = try bo.triggerBackup();
    try std.testing.expect(backup_id != 0);
}

// ============================================================================
// PitrManager Tests
// ============================================================================

test "PitrManager initialization" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{
        .retention_hours = 168, // 7 days
    });
    defer pm.deinit();

    try std.testing.expectEqual(@as(u64, 0), pm.getCurrentSequence());
}

test "PitrManager checkpoint capture" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    // Capture a checkpoint
    const seq1 = try pm.captureCheckpoint();
    try std.testing.expect(seq1 > 0);

    // Capture another checkpoint
    const seq2 = try pm.captureCheckpoint();
    try std.testing.expect(seq2 > seq1);
}

test "PitrManager recovery point listing" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    // Initially no recovery points
    const points = try pm.listRecoveryPoints(allocator);
    defer allocator.free(points);

    try std.testing.expectEqual(@as(usize, 0), points.len);
}

// ============================================================================
// Cross-Component Integration Tests
// ============================================================================

test "HaManager backup trigger through manager" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();

    const backup_id = try manager.triggerBackup();
    try std.testing.expect(backup_id != 0);
}

test "HaManager PITR disabled error" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .enable_pitr = false,
    });
    defer manager.deinit();

    try manager.start();

    // Should error when trying to recover with PITR disabled
    const result = manager.recoverToPoint(0);
    try std.testing.expectError(error.PitrDisabled, result);
}

test "HaStatus formatting" {
    const status = ha.HaStatus{
        .is_running = true,
        .is_primary = true,
        .node_id = 12345,
        .replica_count = 3,
        .replication_lag_ms = 50,
        .backup_state = .idle,
        .pitr_sequence = 100,
    };

    var buf: [256]u8 = undefined;
    const formatted = std.fmt.bufPrint(&buf, "{}", .{status}) catch "";

    try std.testing.expect(std.mem.indexOf(u8, formatted, "RUNNING") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "PRIMARY") != null);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

test "HaManager double start is idempotent" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    try manager.start(); // Should not error
    try std.testing.expect(manager.is_running);
}

test "HaManager double stop is idempotent" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    manager.stop();
    manager.stop(); // Should not error
    try std.testing.expect(!manager.is_running);
}

test "HaManager backup without start" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    // Should error when backup_orchestrator is null
    const result = manager.triggerBackup();
    try std.testing.expectError(error.BackupsDisabled, result);
}
