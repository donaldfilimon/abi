//! HA protocol shared types.
//!
//! Types used by both the real implementation (mod.zig) and the stub (stub.zig).

const std = @import("std");
const backup = @import("backup.zig");

pub const BackupState = backup.BackupState;

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
    /// Path for PITR operation log persistence (empty = no persistence)
    pitr_log_path: []const u8 = "",
    /// Path for PITR checkpoint persistence (empty = no persistence)
    pitr_checkpoint_path: []const u8 = "",
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
        try std.fmt.format(writer, "HA Status: {s} ({s}), Replicas: {d}, Lag: {d}ms", .{
            status,
            role,
            self.replica_count,
            self.replication_lag_ms,
        });
    }
};

test {
    std.testing.refAllDecls(@This());
}
