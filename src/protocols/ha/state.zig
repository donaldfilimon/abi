//! HA Manager state and lifecycle.
//!
//! Contains the HaManager struct: initialization, start/stop, status,
//! backup triggers, recovery, and failover orchestration.

const std = @import("std");
const platform_time = @import("../../foundation/mod.zig").time;
const sync = @import("../../foundation/mod.zig").sync;
const Mutex = sync.Mutex;

const types = @import("types.zig");
const replication = @import("replication.zig");
const backup = @import("backup.zig");
const pitr_mod = @import("pitr.zig");

pub const HaConfig = types.HaConfig;
pub const HaEvent = types.HaEvent;
pub const HaStatus = types.HaStatus;
const BackupState = types.BackupState;
const ReplicationManager = replication.ReplicationManager;
const BackupOrchestrator = backup.BackupOrchestrator;
const PitrManager = pitr_mod.PitrManager;

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
    mutex: Mutex,

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

            // Attempt to load persisted state from prior run
            if (self.pitr_manager) |*pm| {
                if (self.config.pitr_log_path.len > 0) {
                    pm.loadOperationLog(self.config.pitr_log_path) catch {
                        self.emitEvent(.{ .pitr_checkpoint = .{ .sequence = 0, .timestamp = 0 } });
                    };
                }
                if (self.config.pitr_checkpoint_path.len > 0) {
                    pm.loadCheckpoints(self.config.pitr_checkpoint_path) catch {
                        self.emitEvent(.{ .pitr_checkpoint = .{ .sequence = 0, .timestamp = 0 } });
                    };
                }
            }
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
            var result = try pm.recoverToTimestamp(timestamp);
            result.deinit();
            return;
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
        // Use platform-aware unique ID (crypto random on WASM, timestamp-based on native)
        return platform_time.getUniqueId();
    }
};
