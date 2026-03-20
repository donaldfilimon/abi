---
title: mobile API
purpose: Generated API reference for mobile
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# mobile

> Mobile Module

Platform lifecycle, sensors, notifications, permissions, and device info.
Provides simulated mobile platform behavior for development and testing.

**Source:** [`src/features/mobile/mod.zig`](../../src/features/mobile/mod.zig)

**Build flag:** `-Dfeat_mobile=true`

---

## API

### <a id="pub-fn-readsensor-self-context-sensor-type-sensortype-mobileerror-sensordata"></a>`pub fn readSensor(self: *Context, sensor_type: SensorType) MobileError!SensorData`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L47)

Read a simulated sensor value based on the sensor type.

### <a id="pub-fn-sendnotification-self-context-title-const-u8-body-text-const-u8-priority-notification-priority-mobileerror-void"></a>`pub fn sendNotification( self: *Context, title: []const u8, body_text: []const u8, priority: Notification.Priority, ) MobileError!void`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L71)

Send a notification and track it in the log.

### <a id="pub-fn-getnotificationcount-self-const-context-usize"></a>`pub fn getNotificationCount(self: *const Context) usize`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L97)

Return the number of tracked notifications.

### <a id="pub-fn-clearnotifications-self-context-void"></a>`pub fn clearNotifications(self: *Context) void`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L102)

Clear all tracked notifications.

### <a id="pub-fn-checkpermission-self-const-context-perm-permission-permissionstatus"></a>`pub fn checkPermission(self: *const Context, perm: Permission) PermissionStatus`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L107)

Check the current status of a permission.

### <a id="pub-fn-requestpermission-self-context-perm-permission-permissionstatus"></a>`pub fn requestPermission(self: *Context, perm: Permission) PermissionStatus`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L112)

Request a permission (simulated: always grants).

### <a id="pub-fn-revokepermission-self-context-perm-permission-void"></a>`pub fn revokePermission(self: *Context, perm: Permission) void`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L118)

Revoke a previously granted permission.

### <a id="pub-fn-getdeviceinfo-self-const-context-deviceinfo"></a>`pub fn getDeviceInfo(self: *const Context) DeviceInfo`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L123)

Return simulated device information based on the configured platform.

### <a id="pub-fn-readsensor-const-u8-mobileerror-sensordata"></a>`pub fn readSensor(_: []const u8) MobileError!SensorData`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L174)

Legacy module-level readSensor (string-based, returns default data).

### <a id="pub-fn-sendnotification-const-u8-const-u8-mobileerror-void"></a>`pub fn sendNotification(_: []const u8, _: []const u8) MobileError!void`

<sup>**fn**</sup> | [source](../../src/features/mobile/mod.zig#L179)

Legacy module-level sendNotification (no tracking).



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
