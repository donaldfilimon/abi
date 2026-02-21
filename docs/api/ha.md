# ha

> High availability (replication, backup, PITR).

**Source:** [`src/services/ha/mod.zig`](../../src/services/ha/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-haconfig"></a>`pub const HaConfig`

<sup>**const**</sup> | [source](../../src/services/ha/mod.zig#L51)

High Availability configuration

### <a id="pub-const-haevent"></a>`pub const HaEvent`

<sup>**const**</sup> | [source](../../src/services/ha/mod.zig#L73)

High Availability events

### <a id="pub-const-hamanager"></a>`pub const HaManager`

<sup>**const**</sup> | [source](../../src/services/ha/mod.zig#L87)

High Availability manager

### <a id="pub-fn-init-allocator-std-mem-allocator-config-haconfig-hamanager"></a>`pub fn init(allocator: std.mem.Allocator, config: HaConfig) HaManager`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L103)

Initialize the HA manager

### <a id="pub-fn-deinit-self-hamanager-void"></a>`pub fn deinit(self: *HaManager) void`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L118)

Deinitialize the HA manager

### <a id="pub-fn-start-self-hamanager-void"></a>`pub fn start(self: *HaManager) !void`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L135)

Start HA services

### <a id="pub-fn-stop-self-hamanager-void"></a>`pub fn stop(self: *HaManager) void`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L171)

Stop HA services

### <a id="pub-fn-getstatus-self-hamanager-hastatus"></a>`pub fn getStatus(self: *HaManager) HaStatus`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L183)

Get cluster status

### <a id="pub-fn-triggerbackup-self-hamanager-u64"></a>`pub fn triggerBackup(self: *HaManager) !u64`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L217)

Trigger manual backup

### <a id="pub-fn-recovertopoint-self-hamanager-timestamp-i64-void"></a>`pub fn recoverToPoint(self: *HaManager, timestamp: i64) !void`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L225)

Recover to a specific point in time

### <a id="pub-fn-failoverto-self-hamanager-target-node-id-u64-void"></a>`pub fn failoverTo(self: *HaManager, target_node_id: u64) !void`

<sup>**fn**</sup> | [source](../../src/services/ha/mod.zig#L233)

Manual failover to specific node

### <a id="pub-const-hastatus"></a>`pub const HaStatus`

<sup>**const**</sup> | [source](../../src/services/ha/mod.zig#L260)

High Availability status summary

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
