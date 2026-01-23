# High Availability API Reference
> **Codebase Status:** Synced with repository as of 2026-01-22.

**Source:** `src/ha/mod.zig`

The HA module provides comprehensive high-availability features for production deployments including multi-region replication, automated backups, and point-in-time recovery.

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize HA manager
    var manager = abi.ha.HaManager.init(allocator, .{
        .replication_factor = 3,
        .backup_interval_hours = 6,
        .enable_pitr = true,
    });
    defer manager.deinit();

    // Start HA services
    try manager.start();

    // Check status
    const status = manager.getStatus();
    std.debug.print("HA Status: {}\n", .{status});

    // Trigger manual backup
    const backup_id = try manager.triggerBackup();
    std.debug.print("Backup started: {d}\n", .{backup_id});
}
```

---

## Core Types

### `HaManager`

Central coordinator for all HA features.

```zig
pub const HaManager = struct {
    pub fn init(allocator: Allocator, config: HaConfig) HaManager;
    pub fn deinit(self: *HaManager) void;
    pub fn start(self: *HaManager) !void;
    pub fn stop(self: *HaManager) void;
    pub fn getStatus(self: *HaManager) HaStatus;
    pub fn triggerBackup(self: *HaManager) !u64;
    pub fn recoverToPoint(self: *HaManager, timestamp: i64) !void;
    pub fn failoverTo(self: *HaManager, target_node_id: u64) !void;
};
```

### `HaConfig`

Configuration for the HA manager.

```zig
pub const HaConfig = struct {
    /// Number of replicas to maintain (default: 3)
    replication_factor: u8 = 3,
    /// Backup interval in hours (default: 6)
    backup_interval_hours: u32 = 6,
    /// Enable point-in-time recovery (default: true)
    enable_pitr: bool = true,
    /// PITR retention in hours (default: 168 = 7 days)
    pitr_retention_hours: u32 = 168,
    /// Health check interval in seconds (default: 30)
    health_check_interval_sec: u32 = 30,
    /// Max replication lag before failover (ms) (default: 5000)
    max_replication_lag_ms: u64 = 5000,
    /// Enable automatic failover (default: true)
    auto_failover: bool = true,
    /// Regions for multi-region deployment
    regions: []const []const u8 = &.{"primary"},
    /// Callback for HA events
    on_event: ?*const fn (HaEvent) void = null,
};
```

### `HaStatus`

Status summary for the HA cluster.

```zig
pub const HaStatus = struct {
    is_running: bool,
    is_primary: bool,
    node_id: u64,
    replica_count: u32,
    replication_lag_ms: u64,
    backup_state: BackupState,
    pitr_sequence: u64,
};
```

### `HaEvent`

Events emitted by the HA system.

```zig
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
```

---

## Replication

### `ReplicationManager`

Manages data replication across nodes.

```zig
pub const ReplicationManager = struct {
    pub fn init(allocator: Allocator, config: ReplicationConfig) ReplicationManager;
    pub fn deinit(self: *ReplicationManager) void;
    pub fn getState(self: *ReplicationManager) ReplicationState;
    pub fn getReplicaCount(self: *ReplicationManager) u32;
    pub fn getMaxLag(self: *ReplicationManager) u64;
    pub fn getQuorumSize(self: *ReplicationManager) u8;
    pub fn promoteToPrimary(self: *ReplicationManager, node_id: u64) !void;
};
```

### `ReplicationConfig`

```zig
pub const ReplicationConfig = struct {
    /// Number of replicas (default: 3)
    replication_factor: u8 = 3,
    /// Replication mode (default: async_with_ack)
    mode: ReplicationMode = .async_with_ack,
    /// Max lag before warning (ms) (default: 5000)
    max_lag_ms: u64 = 5000,
    /// Write quorum (0 = majority) (default: 0)
    write_quorum: u8 = 0,
    /// Ack timeout (ms) (default: 1000)
    ack_timeout_ms: u64 = 1000,
    /// Auto failover (default: true)
    auto_failover: bool = true,
    /// Heartbeat interval (ms) (default: 1000)
    heartbeat_interval_ms: u64 = 1000,
};
```

### `ReplicationMode`

```zig
pub const ReplicationMode = enum {
    sync,              // Wait for all replicas
    async_fire_forget, // Fire and forget
    async_with_ack,    // Async with quorum acknowledgment
};
```

### `ReplicationState`

```zig
pub const ReplicationState = enum {
    initializing,
    syncing,
    healthy,
    degraded,
    failed,
};
```

---

## Backup

### `BackupOrchestrator`

Manages automated backup operations.

```zig
pub const BackupOrchestrator = struct {
    pub fn init(allocator: Allocator, config: BackupConfig) BackupOrchestrator;
    pub fn deinit(self: *BackupOrchestrator) void;
    pub fn getState(self: *BackupOrchestrator) BackupState;
    pub fn triggerBackup(self: *BackupOrchestrator) !u64;
    pub fn getLastBackup(self: *BackupOrchestrator) ?BackupResult;
    pub fn listBackups(self: *BackupOrchestrator, allocator: Allocator) ![]BackupResult;
};
```

### `BackupConfig`

```zig
pub const BackupConfig = struct {
    /// Backup interval in hours (default: 6)
    interval_hours: u32 = 6,
    /// Backup mode (default: incremental)
    mode: BackupMode = .incremental,
    /// Retention policy
    retention: RetentionPolicy = .{},
    /// Enable compression (default: true)
    compression: bool = true,
    /// Compression level 1-9 (default: 6)
    compression_level: u8 = 6,
    /// Enable encryption (default: false)
    encryption: bool = false,
    /// Backup destinations
    destinations: []const Destination = &.{.{ .type = .local, .path = "backups/" }},
};
```

### `BackupMode`

```zig
pub const BackupMode = enum {
    full,         // Full backup every time
    incremental,  // Incremental with periodic full
    differential, // Differential from last full
};
```

### `BackupState`

```zig
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
```

### `RetentionPolicy`

```zig
pub const RetentionPolicy = struct {
    keep_last: u32 = 10,           // Keep last N backups
    keep_daily_days: u32 = 7,      // Keep daily for N days
    keep_weekly_weeks: u32 = 4,    // Keep weekly for N weeks
    keep_monthly_months: u32 = 12, // Keep monthly for N months
};
```

---

## Point-in-Time Recovery (PITR)

### `PitrManager`

Manages point-in-time recovery capabilities.

```zig
pub const PitrManager = struct {
    pub fn init(allocator: Allocator, config: PitrConfig) PitrManager;
    pub fn deinit(self: *PitrManager) void;
    pub fn getCurrentSequence(self: *PitrManager) u64;
    pub fn captureCheckpoint(self: *PitrManager) !u64;
    pub fn listRecoveryPoints(self: *PitrManager, allocator: Allocator) ![]RecoveryPoint;
    pub fn recoverToTimestamp(self: *PitrManager, timestamp: i64) !void;
    pub fn recoverToSequence(self: *PitrManager, sequence: u64) !void;
};
```

### `PitrConfig`

```zig
pub const PitrConfig = struct {
    /// Retention in hours (default: 168 = 7 days)
    retention_hours: u32 = 168,
    /// Checkpoint interval in seconds (default: 60)
    checkpoint_interval_sec: u32 = 60,
    /// Max WAL size before checkpoint (bytes) (default: 64MB)
    max_wal_size: u64 = 64 * 1024 * 1024,
};
```

### `RecoveryPoint`

```zig
pub const RecoveryPoint = struct {
    sequence: u64,
    timestamp: i64,
    size_bytes: u64,
    is_consistent: bool,
};
```

---

## Usage Patterns

### Basic HA Setup

```zig
var manager = ha.HaManager.init(allocator, .{
    .replication_factor = 3,
    .backup_interval_hours = 6,
    .enable_pitr = true,
});
defer manager.deinit();

try manager.start();
```

### Event Monitoring

```zig
fn handleHaEvent(event: ha.HaEvent) void {
    switch (event) {
        .failover_started => |info| {
            std.log.warn("Failover: {} -> {}", .{ info.from_node, info.to_node });
        },
        .backup_completed => |info| {
            std.log.info("Backup {} complete: {} bytes", .{ info.backup_id, info.size_bytes });
        },
        else => {},
    }
}

var manager = ha.HaManager.init(allocator, .{
    .on_event = &handleHaEvent,
});
```

### Manual Backup

```zig
const backup_id = try manager.triggerBackup();
std.debug.print("Backup started with ID: {d}\n", .{backup_id});
```

### Point-in-Time Recovery

```zig
// Recover to a specific timestamp
const target_time = std.time.timestamp() - 3600; // 1 hour ago
try manager.recoverToPoint(target_time);
```

---

## Related Documentation

- [Replication Guide](replication.md)
- [Backup Strategy](backup.md)
- [Disaster Recovery](disaster_recovery.md)
- [API Reference](API_REFERENCE.md)
