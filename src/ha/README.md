# High Availability (HA) Module

The HA module provides production-ready high availability features for the ABI framework, ensuring system resilience and data durability through replication, automated backups, and point-in-time recovery.

## Features

- **Multi-Region Replication**: Maintain synchronized copies of data across multiple regions with configurable replication factor
- **Automated Backup Orchestration**: Schedule and manage backups at regular intervals with failure recovery
- **Point-in-Time Recovery (PITR)**: Recover data to any point within a retention window (default: 7 days)
- **Health Monitoring**: Continuous health checks of replica nodes with automatic failover capabilities
- **Failover Management**: Automatic or manual promotion of replica nodes to primary

## Architecture

The HA module is composed of three main sub-systems:

| Component | File | Purpose |
|-----------|------|---------|
| **Replication Manager** | `replication.zig` | Manages multi-region data replication and synchronization |
| **Backup Orchestrator** | `backup.zig` | Orchestrates scheduled backups and recovery operations |
| **PITR Manager** | `pitr.zig` | Manages point-in-time recovery checkpoints and replay logs |

## Quick Start

```zig
const abi = @import("abi");
const ha = abi.ha;

var manager = ha.HaManager.init(allocator, .{
    .replication_factor = 3,
    .backup_interval_hours = 6,
    .enable_pitr = true,
    .pitr_retention_hours = 168,  // 7 days
});
defer manager.deinit();

// Start HA services
try manager.start();

// Get cluster status
const status = manager.getStatus();
std.debug.print("Status: {}\n", .{status});

// Trigger manual backup
const backup_id = try manager.triggerBackup();

// Failover to specific node
try manager.failoverTo(target_node_id);

// Recover to a specific timestamp
try manager.recoverToPoint(recovery_timestamp);

// Stop HA services
manager.stop();
```

## Configuration Options

The `HaConfig` struct provides fine-grained control over HA behavior:

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

    /// Maximum acceptable replication lag in milliseconds (default: 5000)
    max_replication_lag_ms: u64 = 5000,

    /// Enable automatic failover (default: true)
    auto_failover: bool = true,

    /// Regions for multi-region deployment (default: ["primary"])
    regions: []const []const u8 = &.{"primary"},

    /// Optional callback for HA events
    on_event: ?*const fn (HaEvent) void = null,
};
```

## API Reference

### Core Methods

**`HaManager.init(allocator, config) HaManager`**
- Initialize the HA manager with the given configuration
- Returns an unstarted manager that must be cleaned up with `deinit()`

**`HaManager.start() !void`**
- Start all HA services (replication, backup, PITR)
- Safe to call multiple times (idempotent)

**`HaManager.stop() void`**
- Stop all HA services gracefully

**`HaManager.getStatus() HaStatus`**
- Get current cluster status including replica count, replication lag, and backup state
- Thread-safe

**`HaManager.triggerBackup() !u64`**
- Manually trigger an immediate backup
- Returns the backup ID on success
- Returns `error.BackupsDisabled` if backups are not configured

**`HaManager.recoverToPoint(timestamp: i64) !void`**
- Recover data to a specific point in time
- Returns `error.PitrDisabled` if PITR is not enabled
- Timestamp should be in milliseconds since epoch

**`HaManager.failoverTo(target_node_id: u64) !void`**
- Manually promote a replica node to primary
- Useful for maintenance or emergency failover scenarios

### Status Reporting

**`HaStatus`** contains:
- `is_running`: Whether HA services are active
- `is_primary`: Whether this node is the primary
- `node_id`: Unique identifier for this node
- `replica_count`: Number of active replicas
- `replication_lag_ms`: Maximum lag between primary and replicas
- `backup_state`: Current state (idle, running, or failed)
- `pitr_sequence`: Current PITR checkpoint sequence number

### Event Handling

The HA module emits events for significant state changes:

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

Register an event callback in the config:

```zig
var manager = ha.HaManager.init(allocator, .{
    .on_event = myEventHandler,
});

fn myEventHandler(event: ha.HaEvent) void {
    switch (event) {
        .failover_completed => |data| {
            std.debug.print("Failover to node {} completed in {} ms\n",
                .{data.new_primary, data.duration_ms});
        },
        .backup_completed => |data| {
            std.debug.print("Backup {} completed: {} bytes\n",
                .{data.backup_id, data.size_bytes});
        },
        else => {},
    }
}
```

## Sub-Modules

### Replication (`replication.zig`)
Manages data replication across multiple nodes and regions. Provides:
- Replica node management and health monitoring
- Synchronous/asynchronous replication modes
- Automatic failover when leader becomes unhealthy
- Replication lag tracking and alerts

### Backup (`backup.zig`)
Orchestrates scheduled and on-demand backups. Provides:
- Configurable backup schedules (interval-based)
- Backup state tracking and result reporting
- Backup retry and failure recovery logic

### PITR (`pitr.zig`)
Manages point-in-time recovery capabilities. Provides:
- Recovery point creation and management
- Retention window enforcement (garbage collection)
- Recovery point metadata and filtering

## Common Use Cases

### Basic HA Setup
```zig
var manager = ha.HaManager.init(allocator, .{
    .replication_factor = 3,
    .backup_interval_hours = 6,
    .enable_pitr = true,
});
try manager.start();
```

### Multi-Region Deployment
```zig
var manager = ha.HaManager.init(allocator, .{
    .replication_factor = 2,
    .regions = &.{"us-east", "eu-west", "ap-south"},
    .backup_interval_hours = 1,
});
try manager.start();
```

### Recovery Scenario
```zig
// Get current status
const status = manager.getStatus();
if (!status.is_running) {
    try manager.start();
}

// Recover to last known good state
const recovery_time = try getLastGoodTimestamp();
try manager.recoverToPoint(recovery_time);
```

## Failure Modes and Recovery

| Scenario | Behavior | Recovery |
|----------|----------|----------|
| Replica lag exceeds threshold | Warning event emitted | Automatic catchup or failover |
| Backup fails | `backup_failed` event | Retry on next interval |
| Primary node fails | Automatic failover (if enabled) | Healthiest replica promoted |
| PITR retention expired | Checkpoint deleted | Older recovery points unavailable |

## Thread Safety

All public methods on `HaManager` are thread-safe and use internal mutex for synchronization.

## See Also

- [docs/deployment.md](/docs/deployment.md) - Production deployment guide
- [docs/plans/](docs/plans/) - Implementation plans and design decisions
