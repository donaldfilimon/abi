//! High Availability Module
//!
//! Provides comprehensive high-availability features for production deployments:
//! - Multi-region replication
//! - Automated backup orchestration
//! - Point-in-time recovery (PITR)
//! - Health monitoring and automatic failover
//!
//! ## Quick Start
//!
//! ```zig
//! const ha = @import("ha");
//!
//! var manager = ha.HaManager.init(allocator, .{
//!     .replication_factor = 3,
//!     .backup_interval_hours = 6,
//!     .enable_pitr = true,
//! });
//! defer manager.deinit();
//!
//! // Start HA services
//! try manager.start();
//! ```

const std = @import("std");
const build_options = @import("build_options");

// Internal submodule imports (const, not pub const, to avoid parity failures)
const ha_types = @import("types.zig");
const ha_state = @import("state.zig");

pub const replication = @import("replication.zig");
pub const backup = @import("backup.zig");
pub const pitr = @import("pitr.zig");

// Re-export main types from submodules
pub const ReplicationManager = replication.ReplicationManager;
pub const ReplicationConfig = replication.ReplicationConfig;
pub const ReplicationState = replication.ReplicationState;
pub const ReplicationEvent = replication.ReplicationEvent;

pub const BackupOrchestrator = backup.BackupOrchestrator;
pub const BackupConfig = backup.BackupConfig;
pub const BackupState = backup.BackupState;
pub const BackupResult = backup.BackupResult;

pub const PitrManager = pitr.PitrManager;
pub const PitrConfig = pitr.PitrConfig;
pub const RecoveryPoint = pitr.RecoveryPoint;

// Re-export HA types from types.zig
pub const HaConfig = ha_types.HaConfig;
pub const HaEvent = ha_types.HaEvent;
pub const HaStatus = ha_types.HaStatus;

// Re-export HaManager from state.zig
pub const HaManager = ha_state.HaManager;

// ── Module Lifecycle ──────────────────────────────────────────────────

var initialized = std.atomic.Value(bool).init(false);

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return error.FeatureDisabled;
    if (initialized.load(.acquire)) return;
    initialized.store(true, .release);
}

pub fn deinit() void {
    initialized.store(false, .release);
}

pub fn isEnabled() bool {
    return build_options.feat_ha;
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
}

// ── Tests ─────────────────────────────────────────────────────────────

test "HaManager initialization" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{
        .replication_factor = 3,
        .backup_interval_hours = 6,
    });
    defer manager.deinit();

    try std.testing.expect(!manager.is_running);
    try std.testing.expect(manager.is_primary);
}

test "HaManager start/stop" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(manager.is_running);

    manager.stop();
    try std.testing.expect(!manager.is_running);
}

test "HaManager double start is idempotent" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    try manager.start(); // second start should be no-op
    try std.testing.expect(manager.is_running);
}

test "HaManager double stop is safe" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    manager.stop();
    manager.stop(); // second stop should be safe
    try std.testing.expect(!manager.is_running);
}

test "HaManager getStatus before start" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{
        .replication_factor = 3,
    });
    defer manager.deinit();

    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expectEqual(@as(u32, 0), status.replica_count);
}

test "HaManager getStatus after start" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{
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

test "HaManager triggerBackup" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    try manager.start();
    const backup_id = try manager.triggerBackup();
    try std.testing.expect(backup_id != 0);
}

test "HaManager triggerBackup before start errors" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    // Before start, backup_orchestrator is null
    try std.testing.expectError(error.BackupsDisabled, manager.triggerBackup());
}

test "HaManager recoverToPoint before start errors" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{
        .enable_pitr = true,
    });
    defer manager.deinit();

    // Before start, pitr_manager is null
    try std.testing.expectError(error.PitrDisabled, manager.recoverToPoint(1000));
}

test "HaManager event callback invoked on start" {
    const allocator = std.testing.allocator;

    const EventTracker = struct {
        var event_count: u32 = 0;
        fn handler(_: HaEvent) void {
            event_count += 1;
        }
    };
    EventTracker.event_count = 0;

    var manager = HaManager.init(allocator, .{
        .on_event = &EventTracker.handler,
    });
    defer manager.deinit();

    try manager.start();
    try std.testing.expect(EventTracker.event_count > 0);
}

test "HaManager node IDs are non-zero" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{});
    defer manager.deinit();

    try std.testing.expect(manager.node_id != 0);
}

test "HaStatus fields" {
    const status = HaStatus{
        .is_running = true,
        .is_primary = true,
        .node_id = 42,
        .replica_count = 3,
        .replication_lag_ms = 50,
        .backup_state = .idle,
        .pitr_sequence = 100,
    };

    try std.testing.expect(status.is_running);
    try std.testing.expect(status.is_primary);
    try std.testing.expectEqual(@as(u64, 42), status.node_id);
    try std.testing.expectEqual(@as(u32, 3), status.replica_count);
    try std.testing.expectEqual(@as(u64, 50), status.replication_lag_ms);
    try std.testing.expectEqual(@as(u64, 100), status.pitr_sequence);
}

test "HaConfig default values" {
    const config = HaConfig{};
    try std.testing.expectEqual(@as(u8, 3), config.replication_factor);
    try std.testing.expectEqual(@as(u32, 6), config.backup_interval_hours);
    try std.testing.expect(config.enable_pitr);
    try std.testing.expectEqual(@as(u32, 168), config.pitr_retention_hours);
    try std.testing.expectEqual(@as(u32, 30), config.health_check_interval_sec);
    try std.testing.expect(config.auto_failover);
}

test {
    std.testing.refAllDecls(@This());
}
