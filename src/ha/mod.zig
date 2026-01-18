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
//! const ha = @import("ha/mod.zig");
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
const platform_time = @import("../shared/time.zig");

pub const replication = @import("replication.zig");
pub const backup = @import("backup.zig");
pub const pitr = @import("pitr.zig");

// Re-export main types
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

/// High Availability configuration
pub const HaConfig = struct {
    /// Number of replicas to maintain
    replication_factor: u8 = 3,
    /// Backup interval in hours
    backup_interval_hours: u32 = 6,
    /// Enable point-in-time recovery
    enable_pitr: bool = true,
    /// PITR retention in hours
    pitr_retention_hours: u32 = 168, // 7 days
    /// Health check interval in seconds
    health_check_interval_sec: u32 = 30,
    /// Maximum lag before triggering failover (milliseconds)
    max_replication_lag_ms: u64 = 5000,
    /// Enable automatic failover
    auto_failover: bool = true,
    /// Regions for multi-region deployment
    regions: []const []const u8 = &.{"primary"},
    /// Callback for HA events
    on_event: ?*const fn (HaEvent) void = null,
};

/// High Availability events
pub const HaEvent = union(enum) {
    replica_added: struct { region: []const u8, node_id: u64 },
    replica_removed: struct { region: []const u8, node_id: u64 },
    replication_lag_warning: struct { node_id: u64, lag_ms: u64 },
    failover_started: struct { from_node: u64, to_node: u64 },
    failover_completed: struct { new_primary: u64, duration_ms: u64 },
    backup_started: struct { backup_id: u64 },
    backup_completed: struct { backup_id: u64, size_bytes: u64 },
    backup_failed: struct { backup_id: u64, reason: []const u8 },
    pitr_checkpoint: struct { sequence: u64, timestamp: i64 },
    health_check_failed: struct { node_id: u64, consecutive_failures: u32 },
};

/// High Availability manager
pub const HaManager = struct {
    allocator: std.mem.Allocator,
    config: HaConfig,

    // Sub-managers
    replication_manager: ?ReplicationManager,
    backup_orchestrator: ?BackupOrchestrator,
    pitr_manager: ?PitrManager,

    // State
    is_running: bool,
    is_primary: bool,
    node_id: u64,
    mutex: std.Thread.Mutex,

    /// Initialize the HA manager
    pub fn init(allocator: std.mem.Allocator, config: HaConfig) HaManager {
        return .{
            .allocator = allocator,
            .config = config,
            .replication_manager = null,
            .backup_orchestrator = null,
            .pitr_manager = null,
            .is_running = false,
            .is_primary = true,
            .node_id = generateNodeId(),
            .mutex = .{},
        };
    }

    /// Deinitialize the HA manager
    pub fn deinit(self: *HaManager) void {
        self.stop();

        if (self.replication_manager) |*rm| {
            rm.deinit();
        }
        if (self.backup_orchestrator) |*bo| {
            bo.deinit();
        }
        if (self.pitr_manager) |*pm| {
            pm.deinit();
        }

        self.* = undefined;
    }

    /// Start HA services
    pub fn start(self: *HaManager) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.is_running) {
            return;
        }

        // Initialize replication
        if (self.config.replication_factor > 1) {
            self.replication_manager = ReplicationManager.init(self.allocator, .{
                .replication_factor = self.config.replication_factor,
                .max_lag_ms = self.config.max_replication_lag_ms,
            });
        }

        // Initialize backup orchestrator
        self.backup_orchestrator = BackupOrchestrator.init(self.allocator, .{
            .interval_hours = self.config.backup_interval_hours,
        });

        // Initialize PITR
        if (self.config.enable_pitr) {
            self.pitr_manager = PitrManager.init(self.allocator, .{
                .retention_hours = self.config.pitr_retention_hours,
            });
        }

        self.is_running = true;
        self.emitEvent(.{ .replica_added = .{
            .region = "primary",
            .node_id = self.node_id,
        } });
    }

    /// Stop HA services
    pub fn stop(self: *HaManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.is_running) {
            return;
        }

        self.is_running = false;
    }

    /// Get cluster status
    pub fn getStatus(self: *HaManager) HaStatus {
        self.mutex.lock();
        defer self.mutex.unlock();

        var replica_count: u32 = 0;
        var replication_lag_ms: u64 = 0;

        if (self.replication_manager) |*rm| {
            replica_count = rm.getReplicaCount();
            replication_lag_ms = rm.getMaxLag();
        }

        var backup_state: BackupState = .idle;
        if (self.backup_orchestrator) |*bo| {
            backup_state = bo.getState();
        }

        var pitr_sequence: u64 = 0;
        if (self.pitr_manager) |*pm| {
            pitr_sequence = pm.getCurrentSequence();
        }

        return .{
            .is_running = self.is_running,
            .is_primary = self.is_primary,
            .node_id = self.node_id,
            .replica_count = replica_count,
            .replication_lag_ms = replication_lag_ms,
            .backup_state = backup_state,
            .pitr_sequence = pitr_sequence,
        };
    }

    /// Trigger manual backup
    pub fn triggerBackup(self: *HaManager) !u64 {
        if (self.backup_orchestrator) |*bo| {
            return bo.triggerBackup();
        }
        return error.BackupsDisabled;
    }

    /// Recover to a specific point in time
    pub fn recoverToPoint(self: *HaManager, timestamp: i64) !void {
        if (self.pitr_manager) |*pm| {
            return pm.recoverToTimestamp(timestamp);
        }
        return error.PitrDisabled;
    }

    /// Manual failover to specific node
    pub fn failoverTo(self: *HaManager, target_node_id: u64) !void {
        if (self.replication_manager) |*rm| {
            const old_primary = self.node_id;
            try rm.promoteToPrimary(target_node_id);

            self.emitEvent(.{ .failover_started = .{
                .from_node = old_primary,
                .to_node = target_node_id,
            } });

            self.is_primary = (target_node_id == self.node_id);
        }
    }

    fn emitEvent(self: *HaManager, event: HaEvent) void {
        if (self.config.on_event) |callback| {
            callback(event);
        }
    }

    fn generateNodeId() u64 {
        // Use platform-aware time for entropy seed
        const seed = platform_time.timestampNs();
        var prng = std.Random.DefaultPrng.init(seed);
        return prng.random().int(u64);
    }
};

/// High Availability status summary
pub const HaStatus = struct {
    is_running: bool,
    is_primary: bool,
    node_id: u64,
    replica_count: u32,
    replication_lag_ms: u64,
    backup_state: BackupState,
    pitr_sequence: u64,

    pub fn format(
        self: HaStatus,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const role = if (self.is_primary) "PRIMARY" else "REPLICA";
        const status = if (self.is_running) "RUNNING" else "STOPPED";
        try writer.print("HA Status: {s} ({s}), Replicas: {d}, Lag: {d}ms", .{
            status,
            role,
            self.replica_count,
            self.replication_lag_ms,
        });
    }
};

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
