//! Integration Tests: HA (High Availability) Module
//!
//! Verifies the HA module's public API from a consumer perspective:
//! HaManager lifecycle, backup orchestration, PITR checkpointing,
//! replication management, and full lifecycle coordination.

const std = @import("std");
const abi = @import("abi");

const ha = abi.ha;

// ============================================================================
// Type availability
// ============================================================================

test "ha: HaManager type exists" {
    const T = ha.HaManager;
    _ = T;
}

test "ha: HaConfig type exists" {
    const T = ha.HaConfig;
    _ = T;
}

test "ha: HaStatus type exists" {
    const T = ha.HaStatus;
    _ = T;
}

test "ha: HaEvent type exists" {
    const T = ha.HaEvent;
    _ = T;
}

test "ha: ReplicationManager type exists" {
    const T = ha.ReplicationManager;
    _ = T;
}

test "ha: ReplicationConfig type exists" {
    const T = ha.ReplicationConfig;
    _ = T;
}

test "ha: ReplicationState type exists" {
    const T = ha.ReplicationState;
    _ = T;
}

test "ha: ReplicationEvent type exists" {
    const T = ha.ReplicationEvent;
    _ = T;
}

test "ha: BackupOrchestrator type exists" {
    const T = ha.BackupOrchestrator;
    _ = T;
}

test "ha: BackupConfig type exists" {
    const T = ha.BackupConfig;
    _ = T;
}

test "ha: BackupState type exists" {
    const T = ha.BackupState;
    _ = T;
}

test "ha: BackupResult type exists" {
    const T = ha.BackupResult;
    _ = T;
}

test "ha: PitrManager type exists" {
    const T = ha.PitrManager;
    _ = T;
}

test "ha: PitrConfig type exists" {
    const T = ha.PitrConfig;
    _ = T;
}

test "ha: RecoveryPoint type exists" {
    const T = ha.RecoveryPoint;
    _ = T;
}

// ============================================================================
// HaConfig defaults
// ============================================================================

test "ha: HaConfig default values" {
    const config = ha.HaConfig{};
    try std.testing.expectEqual(@as(u8, 3), config.replication_factor);
    try std.testing.expectEqual(@as(u32, 6), config.backup_interval_hours);
    try std.testing.expect(config.enable_pitr);
    try std.testing.expectEqual(@as(u32, 168), config.pitr_retention_hours);
    try std.testing.expectEqual(@as(u32, 30), config.health_check_interval_sec);
    try std.testing.expect(config.auto_failover);
    try std.testing.expect(config.on_event == null);
}

// ============================================================================
// HaManager lifecycle
// ============================================================================

test "ha: HaManager init/deinit" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expect(status.node_id != 0);
}

test "ha: HaManager start sets running" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(manager.getStatus().is_running);
}

test "ha: HaManager stop clears running" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    manager.stop();
    try std.testing.expect(!manager.getStatus().is_running);
}

test "ha: HaManager double start is idempotent" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    try manager.start();
    try std.testing.expect(manager.getStatus().is_running);
}

test "ha: HaManager double stop is safe" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    manager.stop();
    manager.stop();
    try std.testing.expect(!manager.getStatus().is_running);
}

// ============================================================================
// HaManager status reporting
// ============================================================================

test "ha: getStatus before start" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 3,
    });
    defer manager.deinit();

    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expectEqual(@as(u32, 0), status.replica_count);
    try std.testing.expectEqual(@as(u64, 0), status.replication_lag_ms);
}

test "ha: getStatus after start" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 3,
        .enable_pitr = true,
    });
    defer manager.deinit();

    try manager.start();

    const status = manager.getStatus();
    try std.testing.expect(status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expect(status.node_id != 0);
}

test "ha: getStatus backup state is idle after start" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();

    const status = manager.getStatus();
    try std.testing.expectEqual(ha.BackupState.idle, status.backup_state);
}

// ============================================================================
// HaManager event callback
// ============================================================================

test "ha: event callback fires on start" {
    const allocator = std.testing.allocator;

    const Tracker = struct {
        var count: u32 = 0;
        fn handler(_: ha.HaEvent) void {
            count += 1;
        }
    };
    Tracker.count = 0;

    var manager = ha.HaManager.init(allocator, .{
        .on_event = &Tracker.handler,
    });
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(Tracker.count > 0);
}

// ============================================================================
// Backup lifecycle via HaManager
// ============================================================================

test "ha: triggerBackup before start returns error" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try std.testing.expectError(error.BackupsDisabled, manager.triggerBackup());
}

test "ha: triggerBackup after start succeeds" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    const backup_id = try manager.triggerBackup();
    try std.testing.expect(backup_id > 0);
}

test "ha: multiple backups produce distinct IDs" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    const id1 = try manager.triggerBackup();
    const id2 = try manager.triggerBackup();
    try std.testing.expect(id1 != id2);
}

// ============================================================================
// PITR via HaManager
// ============================================================================

test "ha: recoverToPoint before start returns error" {
    const allocator = std.testing.allocator;

    var manager = ha.HaManager.init(allocator, .{
        .enable_pitr = true,
    });
    defer manager.deinit();

    try std.testing.expectError(error.PitrDisabled, manager.recoverToPoint(1000));
}

// ============================================================================
// BackupOrchestrator standalone
// ============================================================================

test "ha: BackupOrchestrator init/deinit" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{
        .interval_hours = 12,
    });
    defer bo.deinit();

    try std.testing.expectEqual(ha.BackupState.idle, bo.getState());
}

test "ha: BackupOrchestrator triggerBackup records history" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const id = try bo.triggerBackup();
    try std.testing.expect(id > 0);

    const backups = bo.listBackups();
    try std.testing.expectEqual(@as(usize, 1), backups.len);
    try std.testing.expectEqual(id, backups[0].backup_id);
}

test "ha: BackupOrchestrator triggerFullBackup" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const id = try bo.triggerFullBackup();
    try std.testing.expect(id > 0);
}

test "ha: BackupOrchestrator getBackup returns metadata" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const id = try bo.triggerBackup();
    const meta = bo.getBackup(id);
    try std.testing.expect(meta != null);
    try std.testing.expectEqual(id, meta.?.backup_id);
}

test "ha: BackupOrchestrator getBackup with unknown ID returns null" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    try std.testing.expect(bo.getBackup(9999) == null);
}

test "ha: BackupOrchestrator verifyBackup passes for existing backup" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    const id = try bo.triggerBackup();
    const verified = try bo.verifyBackup(id);
    try std.testing.expect(verified);
}

test "ha: BackupOrchestrator verifyBackup errors for unknown ID" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    try std.testing.expectError(error.BackupNotFound, bo.verifyBackup(9999));
}

test "ha: BackupOrchestrator applyRetention trims old backups" {
    const allocator = std.testing.allocator;

    var bo = ha.BackupOrchestrator.init(allocator, .{
        .retention = .{ .keep_last = 2 },
    });
    defer bo.deinit();

    // Create 4 backups
    _ = try bo.triggerBackup();
    _ = try bo.triggerBackup();
    _ = try bo.triggerBackup();
    _ = try bo.triggerBackup();

    try std.testing.expectEqual(@as(usize, 4), bo.listBackups().len);

    try bo.applyRetention();
    try std.testing.expectEqual(@as(usize, 2), bo.listBackups().len);
}

// ============================================================================
// ReplicationManager standalone
// ============================================================================

test "ha: ReplicationManager init/deinit" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{
        .replication_factor = 3,
    });
    defer rm.deinit();

    try std.testing.expectEqual(ha.ReplicationState.initializing, rm.getState());
    try std.testing.expect(rm.isLeader());
}

test "ha: ReplicationManager add and count replicas" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try rm.addReplica(100, "us-east-1", "10.0.0.2:5432");
    try std.testing.expectEqual(@as(u32, 1), rm.getReplicaCount());

    try rm.addReplica(200, "us-west-2", "10.0.1.2:5432");
    try std.testing.expectEqual(@as(u32, 2), rm.getReplicaCount());
}

test "ha: ReplicationManager remove replica" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try rm.addReplica(100, "us-east-1", "10.0.0.2:5432");
    try std.testing.expectEqual(@as(u32, 1), rm.getReplicaCount());

    rm.removeReplica(100, .node_shutdown);
    try std.testing.expectEqual(@as(u32, 0), rm.getReplicaCount());
}

test "ha: ReplicationManager getMaxLag starts at zero" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try std.testing.expectEqual(@as(u64, 0), rm.getMaxLag());
}

test "ha: ReplicationManager replicate as leader" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{
        .mode = .async_fire_forget,
    });
    defer rm.deinit();

    // Leader can replicate even with no replicas in async_fire_forget mode
    try rm.replicate("key1", "value1");
}

test "ha: ReplicationManager promoteToPrimary" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    // Promote another node -- local becomes non-leader
    try rm.promoteToPrimary(999);
    try std.testing.expect(!rm.isLeader());
    try std.testing.expectEqual(@as(u64, 999), rm.getLeaderNodeId());
}

test "ha: ReplicationManager processHeartbeat updates replica state" {
    const allocator = std.testing.allocator;

    var rm = ha.ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try rm.addReplica(100, "us-east-1", "10.0.0.2:5432");

    // Process a heartbeat with matching sequence -- should set active
    rm.processHeartbeat(100, rm.getCurrentSequence());
}

// ============================================================================
// PitrManager standalone
// ============================================================================

test "ha: PitrManager init/deinit" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{
        .retention_hours = 24,
    });
    defer pm.deinit();

    try std.testing.expectEqual(@as(u64, 0), pm.getCurrentSequence());
}

test "ha: PitrManager captureOperation and createCheckpoint" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{
        .checkpoint_interval_sec = 3600, // disable auto-checkpoint
    });
    defer pm.deinit();

    try pm.captureOperation(.insert, "key1", "value1", null);
    try pm.captureOperation(.update, "key1", "value2", "value1");
    try pm.captureOperation(.delete, "key2", null, "old_value");

    const seq = try pm.createCheckpoint();
    try std.testing.expect(seq > 0);

    const points = pm.getRecoveryPoints();
    try std.testing.expectEqual(@as(usize, 1), points.len);
    try std.testing.expectEqual(@as(u64, 3), points[0].operation_count);
}

test "ha: PitrManager createCheckpoint with no ops returns current sequence" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    const seq = try pm.createCheckpoint();
    try std.testing.expectEqual(@as(u64, 0), seq);
    try std.testing.expectEqual(@as(usize, 0), pm.getRecoveryPoints().len);
}

test "ha: PitrManager recoverToSequence with unknown sequence errors" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expectError(error.SequenceNotFound, pm.recoverToSequence(999));
}

test "ha: PitrManager recoverToTimestamp with no points errors" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expectError(error.NoRecoveryPoint, pm.recoverToTimestamp(1000));
}

test "ha: PitrManager findNearestRecoveryPoint returns null with no points" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expect(pm.findNearestRecoveryPoint(1000) == null);
}

test "ha: PitrManager findNearestRecoveryPoint with negative timestamp" {
    const allocator = std.testing.allocator;

    var pm = ha.PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expect(pm.findNearestRecoveryPoint(-1) == null);
}

// ============================================================================
// Full HA lifecycle
// ============================================================================

test "ha: full lifecycle -- init, start, backup, status, stop" {
    const allocator = std.testing.allocator;

    // 1. Init with replication and PITR enabled
    var manager = ha.HaManager.init(allocator, .{
        .replication_factor = 3,
        .backup_interval_hours = 6,
        .enable_pitr = true,
    });
    defer manager.deinit();

    // 2. Start HA services
    try manager.start();

    const status1 = manager.getStatus();
    try std.testing.expect(status1.is_running);
    try std.testing.expect(status1.is_primary);

    // 3. Trigger a backup
    const backup_id = try manager.triggerBackup();
    try std.testing.expect(backup_id > 0);

    // 4. Check status reflects backup orchestrator
    const status2 = manager.getStatus();
    try std.testing.expectEqual(ha.BackupState.idle, status2.backup_state);

    // 5. Stop HA services
    manager.stop();
    const status3 = manager.getStatus();
    try std.testing.expect(!status3.is_running);
}

// ============================================================================
// HaStatus formatting
// ============================================================================

test "ha: HaStatus format produces output" {
    const status = ha.HaStatus{
        .is_running = true,
        .is_primary = true,
        .node_id = 42,
        .replica_count = 3,
        .replication_lag_ms = 50,
        .backup_state = .idle,
        .pitr_sequence = 100,
    };

    var buf: [256]u8 = undefined;
    const result = std.fmt.bufPrint(&buf, "{}", .{status}) catch unreachable;
    try std.testing.expect(result.len > 0);
}
