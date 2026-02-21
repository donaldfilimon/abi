# ha

> High availability (replication, backup, PITR).

**Source:** [`src/services/ha/mod.zig`](../../src/services/ha/mod.zig)

**Availability:** Always enabled

---

High Availability Module

Provides comprehensive high-availability features for production deployments:
- Multi-region replication
- Automated backup orchestration
- Point-in-time recovery (PITR)
- Health monitoring and automatic failover

## Quick Start

```zig
const ha = @import("ha/mod.zig");

var manager = ha.HaManager.init(allocator, .{
.replication_factor = 3,
.backup_interval_hours = 6,
.enable_pitr = true,
});
defer manager.deinit();

// Start HA services
try manager.start();
```

---

## API

### `pub const HaConfig`

<sup>**type**</sup>

High Availability configuration

### `pub const HaEvent`

<sup>**type**</sup>

High Availability events

### `pub const HaManager`

<sup>**type**</sup>

High Availability manager

### `pub fn init(allocator: std.mem.Allocator, config: HaConfig) HaManager`

<sup>**fn**</sup>

Initialize the HA manager

### `pub fn deinit(self: *HaManager) void`

<sup>**fn**</sup>

Deinitialize the HA manager

### `pub fn start(self: *HaManager) !void`

<sup>**fn**</sup>

Start HA services

### `pub fn stop(self: *HaManager) void`

<sup>**fn**</sup>

Stop HA services

### `pub fn getStatus(self: *HaManager) HaStatus`

<sup>**fn**</sup>

Get cluster status

### `pub fn triggerBackup(self: *HaManager) !u64`

<sup>**fn**</sup>

Trigger manual backup

### `pub fn recoverToPoint(self: *HaManager, timestamp: i64) !void`

<sup>**fn**</sup>

Recover to a specific point in time

### `pub fn failoverTo(self: *HaManager, target_node_id: u64) !void`

<sup>**fn**</sup>

Manual failover to specific node

### `pub const HaStatus`

<sup>**type**</sup>

High Availability status summary

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
