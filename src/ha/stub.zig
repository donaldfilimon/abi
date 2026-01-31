//! Stub for High Availability module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.HaDisabled for all operations.

const std = @import("std");

/// HA module errors.
pub const Error = error{
    HaDisabled,
    BackupsDisabled,
    PitrDisabled,
    BackupInProgress,
    BackupNotFound,
    NoRecoveryPoint,
    SequenceNotFound,
    NotLeader,
    QuorumNotReached,
    OutOfMemory,
};

pub const HaError = Error;

// =============================================================================
// Replication Types
// =============================================================================

/// Replication configuration.
pub const ReplicationConfig = struct {
    replication_factor: u8 = 3,
    mode: ReplicationMode = .async_with_ack,
    max_lag_ms: u64 = 5000,
    write_quorum: u8 = 0,
    ack_timeout_ms: u64 = 1000,
    auto_failover: bool = true,
    heartbeat_interval_ms: u64 = 1000,
    on_event: ?*const fn (ReplicationEvent) void = null,
};

/// Replication mode.
pub const ReplicationMode = enum {
    sync,
    async_fire_forget,
    async_with_ack,
};

/// Replication state.
pub const ReplicationState = enum {
    initializing,
    syncing,
    healthy,
    degraded,
    failed,
};

/// Disconnect reasons.
pub const DisconnectReason = enum {
    timeout,
    network_error,
    node_shutdown,
    kicked,
};

/// Conflict resolution strategies.
pub const ConflictResolution = enum {
    last_write_wins,
    first_write_wins,
    manual_resolution_required,
};

/// Replication events.
pub const ReplicationEvent = union(enum) {
    replica_connected: struct { node_id: u64, region: []const u8 },
    replica_disconnected: struct { node_id: u64, reason: DisconnectReason },
    replication_lag: struct { node_id: u64, lag_ms: u64 },
    quorum_lost: void,
    quorum_restored: void,
    leader_elected: struct { node_id: u64 },
    conflict_detected: struct { key: []const u8, resolution: ConflictResolution },
};

/// Replication manager stub.
pub const ReplicationManager = struct {
    allocator: std.mem.Allocator,
    config: ReplicationConfig,

    pub fn init(allocator: std.mem.Allocator, config: ReplicationConfig) ReplicationManager {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *ReplicationManager) void {
        _ = self;
    }

    pub fn addReplica(
        self: *ReplicationManager,
        node_id: u64,
        region: []const u8,
        address: []const u8,
    ) Error!void {
        _ = self;
        _ = node_id;
        _ = region;
        _ = address;
        return Error.HaDisabled;
    }

    pub fn removeReplica(self: *ReplicationManager, node_id: u64, reason: DisconnectReason) void {
        _ = self;
        _ = node_id;
        _ = reason;
    }

    pub fn getReplicaCount(self: *ReplicationManager) u32 {
        _ = self;
        return 0;
    }

    pub fn getMaxLag(self: *ReplicationManager) u64 {
        _ = self;
        return 0;
    }

    pub fn replicate(self: *ReplicationManager, key: []const u8, value: []const u8) Error!void {
        _ = self;
        _ = key;
        _ = value;
        return Error.HaDisabled;
    }

    pub fn promoteToPrimary(self: *ReplicationManager, node_id: u64) Error!void {
        _ = self;
        _ = node_id;
        return Error.HaDisabled;
    }

    pub fn processHeartbeat(self: *ReplicationManager, node_id: u64, sequence: u64) void {
        _ = self;
        _ = node_id;
        _ = sequence;
    }
};

// =============================================================================
// Backup Types
// =============================================================================

/// Backup configuration.
pub const BackupConfig = struct {
    interval_hours: u32 = 6,
    mode: BackupMode = .incremental,
    retention: RetentionPolicy = .{},
    compression: bool = true,
    compression_level: u8 = 6,
    encryption: bool = false,
    destinations: []const Destination = &.{.{ .type = .local, .path = "backups/" }},
    on_event: ?*const fn (BackupEvent) void = null,
};

/// Backup mode.
pub const BackupMode = enum {
    full,
    incremental,
    differential,
};

/// Retention policy.
pub const RetentionPolicy = struct {
    keep_last: u32 = 10,
    keep_daily_days: u32 = 7,
    keep_weekly_weeks: u32 = 4,
    keep_monthly_months: u32 = 12,
};

/// Backup destination.
pub const Destination = struct {
    type: DestinationType,
    path: []const u8,
    bucket: ?[]const u8 = null,
    region: ?[]const u8 = null,
    credentials: ?[]const u8 = null,
};

/// Destination types.
pub const DestinationType = enum {
    local,
    s3,
    gcs,
    azure_blob,
};

/// Backup state.
pub const BackupState = enum {
    idle,
    preparing,
    backing_up,
    compressing,
    encrypting,
    uploading,
    verifying,
    completed,
    failed,
};

/// Backup events.
pub const BackupEvent = union(enum) {
    backup_started: struct { backup_id: u64, mode: BackupMode },
    backup_progress: struct { backup_id: u64, percent: u8 },
    backup_completed: struct { backup_id: u64, size_bytes: u64, duration_ms: u64 },
    backup_failed: struct { backup_id: u64, reason: []const u8 },
    upload_started: struct { backup_id: u64, destination: DestinationType },
    upload_completed: struct { backup_id: u64, destination: DestinationType },
    verification_passed: struct { backup_id: u64 },
    verification_failed: struct { backup_id: u64, reason: []const u8 },
    retention_cleanup: struct { deleted_count: u32, freed_bytes: u64 },
};

/// Backup result.
pub const BackupResult = struct {
    backup_id: u64,
    timestamp: u64,
    mode: BackupMode,
    size_bytes: u64,
    size_compressed: u64,
    duration_ms: u64,
    checksum: [32]u8,
    destinations_succeeded: u32,
    destinations_failed: u32,
};

/// Backup metadata (internal).
pub const BackupMetadata = struct {
    backup_id: u64,
    timestamp: u64,
    mode: BackupMode,
    size_bytes: u64,
    checksum: [32]u8,
    base_backup_id: ?u64,
    sequence_number: u64,
};

/// Backup orchestrator stub.
pub const BackupOrchestrator = struct {
    allocator: std.mem.Allocator,
    config: BackupConfig,
    state: BackupState,

    pub fn init(allocator: std.mem.Allocator, config: BackupConfig) BackupOrchestrator {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .idle,
        };
    }

    pub fn deinit(self: *BackupOrchestrator) void {
        _ = self;
    }

    pub fn getState(self: *BackupOrchestrator) BackupState {
        _ = self;
        return .idle;
    }

    pub fn isBackupDue(self: *BackupOrchestrator) bool {
        _ = self;
        return false;
    }

    pub fn triggerBackup(self: *BackupOrchestrator) Error!u64 {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn triggerFullBackup(self: *BackupOrchestrator) Error!u64 {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn listBackups(self: *BackupOrchestrator) []const BackupMetadata {
        _ = self;
        return &.{};
    }

    pub fn getBackup(self: *BackupOrchestrator, backup_id: u64) ?BackupMetadata {
        _ = self;
        _ = backup_id;
        return null;
    }

    pub fn applyRetention(self: *BackupOrchestrator) Error!void {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn verifyBackup(self: *BackupOrchestrator, backup_id: u64) Error!bool {
        _ = self;
        _ = backup_id;
        return Error.HaDisabled;
    }
};

// =============================================================================
// PITR Types
// =============================================================================

/// PITR configuration.
pub const PitrConfig = struct {
    retention_hours: u32 = 168,
    checkpoint_interval_sec: u32 = 60,
    max_checkpoint_size: u64 = 10 * 1024 * 1024,
    compression: bool = true,
    storage_path: []const u8 = "pitr/",
    on_event: ?*const fn (PitrEvent) void = null,
};

/// PITR events.
pub const PitrEvent = union(enum) {
    checkpoint_created: struct { sequence: u64, size_bytes: u64 },
    checkpoint_pruned: struct { sequence: u64 },
    recovery_started: struct { target_timestamp: i64 },
    recovery_completed: struct { target_timestamp: i64, operations_replayed: u64 },
    recovery_failed: struct { reason: []const u8 },
    retention_applied: struct { pruned_count: u32, freed_bytes: u64 },
};

/// Recovery point information.
pub const RecoveryPoint = struct {
    sequence: u64,
    timestamp: u64,
    size_bytes: u64,
    operation_count: u64,
    checksum: [32]u8,
};

/// Checkpoint header.
pub const CheckpointHeader = extern struct {
    magic: u32 = 0x50495452,
    version: u16 = 1,
    flags: u16 = 0,
    sequence: u64,
    timestamp: i64,
    operation_count: u64,
    data_size: u64,
    checksum: [32]u8,
    reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Operation types for change capture.
pub const OperationType = enum(u8) {
    insert = 1,
    update = 2,
    delete = 3,
    truncate = 4,
};

/// Captured operation.
pub const Operation = struct {
    type: OperationType,
    timestamp: i64,
    key: []const u8,
    value: ?[]const u8,
    previous_value: ?[]const u8,
};

/// PITR manager stub.
pub const PitrManager = struct {
    allocator: std.mem.Allocator,
    config: PitrConfig,

    pub fn init(allocator: std.mem.Allocator, config: PitrConfig) PitrManager {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *PitrManager) void {
        _ = self;
    }

    pub fn getCurrentSequence(self: *PitrManager) u64 {
        _ = self;
        return 0;
    }

    pub fn captureOperation(
        self: *PitrManager,
        op_type: OperationType,
        key: []const u8,
        value: ?[]const u8,
        previous_value: ?[]const u8,
    ) Error!void {
        _ = self;
        _ = op_type;
        _ = key;
        _ = value;
        _ = previous_value;
        return Error.HaDisabled;
    }

    pub fn createCheckpoint(self: *PitrManager) Error!u64 {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn getRecoveryPoints(self: *PitrManager) []const RecoveryPoint {
        _ = self;
        return &.{};
    }

    pub fn findNearestRecoveryPoint(self: *PitrManager, timestamp: i64) ?RecoveryPoint {
        _ = self;
        _ = timestamp;
        return null;
    }

    pub fn recoverToTimestamp(self: *PitrManager, timestamp: i64) Error!void {
        _ = self;
        _ = timestamp;
        return Error.HaDisabled;
    }

    pub fn recoverToSequence(self: *PitrManager, sequence: u64) Error!void {
        _ = self;
        _ = sequence;
        return Error.HaDisabled;
    }

    pub fn applyRetention(self: *PitrManager) Error!void {
        _ = self;
        return Error.HaDisabled;
    }
};

// =============================================================================
// HA Manager Types
// =============================================================================

/// High Availability configuration.
pub const HaConfig = struct {
    replication_factor: u8 = 3,
    backup_interval_hours: u32 = 6,
    enable_pitr: bool = true,
    pitr_retention_hours: u32 = 168,
    health_check_interval_sec: u32 = 30,
    max_replication_lag_ms: u64 = 5000,
    auto_failover: bool = true,
    regions: []const []const u8 = &.{"primary"},
    on_event: ?*const fn (HaEvent) void = null,
};

/// High Availability events.
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

/// High Availability status summary.
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

/// High Availability manager stub.
pub const HaManager = struct {
    allocator: std.mem.Allocator,
    config: HaConfig,

    pub fn init(allocator: std.mem.Allocator, config: HaConfig) HaManager {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *HaManager) void {
        _ = self;
    }

    pub fn start(self: *HaManager) Error!void {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn stop(self: *HaManager) void {
        _ = self;
    }

    pub fn getStatus(self: *HaManager) HaStatus {
        _ = self;
        return .{
            .is_running = false,
            .is_primary = false,
            .node_id = 0,
            .replica_count = 0,
            .replication_lag_ms = 0,
            .backup_state = .idle,
            .pitr_sequence = 0,
        };
    }

    pub fn triggerBackup(self: *HaManager) Error!u64 {
        _ = self;
        return Error.HaDisabled;
    }

    pub fn recoverToPoint(self: *HaManager, timestamp: i64) Error!void {
        _ = self;
        _ = timestamp;
        return Error.HaDisabled;
    }

    pub fn failoverTo(self: *HaManager, target_node_id: u64) Error!void {
        _ = self;
        _ = target_node_id;
        return Error.HaDisabled;
    }
};

// =============================================================================
// Re-export sub-modules as namespaces for API compatibility
// =============================================================================

pub const replication = struct {
    pub const ReplicationManager = @This().ReplicationManager;
    pub const ReplicationConfig = @This().ReplicationConfig;
    pub const ReplicationState = @This().ReplicationState;
    pub const ReplicationEvent = @This().ReplicationEvent;
    pub const ReplicationMode = @This().ReplicationMode;
    pub const DisconnectReason = @This().DisconnectReason;
    pub const ConflictResolution = @This().ConflictResolution;
};

pub const backup = struct {
    pub const BackupOrchestrator = @This().BackupOrchestrator;
    pub const BackupConfig = @This().BackupConfig;
    pub const BackupState = @This().BackupState;
    pub const BackupResult = @This().BackupResult;
    pub const BackupEvent = @This().BackupEvent;
    pub const BackupMode = @This().BackupMode;
    pub const BackupMetadata = @This().BackupMetadata;
    pub const RetentionPolicy = @This().RetentionPolicy;
    pub const Destination = @This().Destination;
    pub const DestinationType = @This().DestinationType;
};

pub const pitr = struct {
    pub const PitrManager = @This().PitrManager;
    pub const PitrConfig = @This().PitrConfig;
    pub const RecoveryPoint = @This().RecoveryPoint;
    pub const PitrEvent = @This().PitrEvent;
    pub const CheckpointHeader = @This().CheckpointHeader;
    pub const OperationType = @This().OperationType;
    pub const Operation = @This().Operation;
};

// =============================================================================
// Module Lifecycle
// =============================================================================

var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) Error!void {
    _ = allocator;
    return Error.HaDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

// =============================================================================
// Tests
// =============================================================================

test "HaManager stub initialization" {
    const allocator = std.testing.allocator;

    var manager = HaManager.init(allocator, .{
        .replication_factor = 3,
        .backup_interval_hours = 6,
    });
    defer manager.deinit();

    // Start should return HaDisabled
    try std.testing.expectError(Error.HaDisabled, manager.start());

    // getStatus should return default values
    const status = manager.getStatus();
    try std.testing.expect(!status.is_running);
    try std.testing.expect(!status.is_primary);
    try std.testing.expectEqual(@as(u64, 0), status.node_id);
}

test "ReplicationManager stub operations" {
    const allocator = std.testing.allocator;

    var rm = ReplicationManager.init(allocator, .{});
    defer rm.deinit();

    try std.testing.expectEqual(@as(u32, 0), rm.getReplicaCount());
    try std.testing.expectEqual(@as(u64, 0), rm.getMaxLag());
    try std.testing.expectError(Error.HaDisabled, rm.addReplica(1, "us-east-1", "10.0.0.1:5432"));
    try std.testing.expectError(Error.HaDisabled, rm.replicate("key", "value"));
}

test "BackupOrchestrator stub operations" {
    const allocator = std.testing.allocator;

    var bo = BackupOrchestrator.init(allocator, .{});
    defer bo.deinit();

    try std.testing.expectEqual(BackupState.idle, bo.getState());
    try std.testing.expect(!bo.isBackupDue());
    try std.testing.expectError(Error.HaDisabled, bo.triggerBackup());
    try std.testing.expectEqual(@as(usize, 0), bo.listBackups().len);
}

test "PitrManager stub operations" {
    const allocator = std.testing.allocator;

    var pm = PitrManager.init(allocator, .{});
    defer pm.deinit();

    try std.testing.expectEqual(@as(u64, 0), pm.getCurrentSequence());
    try std.testing.expectError(Error.HaDisabled, pm.createCheckpoint());
    try std.testing.expectEqual(@as(usize, 0), pm.getRecoveryPoints().len);
    try std.testing.expect(pm.findNearestRecoveryPoint(12345) == null);
}
