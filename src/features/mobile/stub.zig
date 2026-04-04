//! Mobile Stub Module — no-op when mobile is disabled (default).

const std = @import("std");
const stub_helpers = @import("../core/stub_helpers.zig");
pub const types = @import("types.zig");

// Submodule stubs — empty structs matching mod.zig's public surface
pub const sensors = struct {};
pub const notifications = struct {};
pub const permissions = struct {};
pub const device = struct {};

pub const MobileConfig = types.MobileConfig;
pub const MobilePlatform = types.MobilePlatform;
pub const MobileError = types.MobileError;
pub const Error = MobileError;
pub const LifecycleState = types.LifecycleState;
pub const SensorType = types.SensorType;
pub const SensorData = types.SensorData;
pub const Notification = types.Notification;
pub const NotificationEntry = types.NotificationEntry;
pub const Permission = types.Permission;
pub const PermissionStatus = types.PermissionStatus;
pub const DeviceInfo = types.DeviceInfo;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MobileConfig,
    state: LifecycleState = .active,
    permissions: [types.permission_count]PermissionStatus = @splat(.not_requested),
    notification_log: std.ArrayListUnmanaged(NotificationEntry) = .empty,

    pub fn init(_: std.mem.Allocator, _: MobileConfig) !*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn readSensor(_: *Context, _: SensorType) MobileError!SensorData {
        return error.FeatureDisabled;
    }
    pub fn sendNotification(_: *Context, _: []const u8, _: []const u8, _: Notification.Priority) MobileError!void {
        return error.FeatureDisabled;
    }
    pub fn getNotificationCount(_: *const Context) usize {
        return 0;
    }
    pub fn clearNotifications(_: *Context) void {}
    pub fn checkPermission(_: *const Context, _: Permission) PermissionStatus {
        return .not_requested;
    }
    pub fn requestPermission(_: *Context, _: Permission) PermissionStatus {
        return .denied;
    }
    pub fn revokePermission(_: *Context, _: Permission) void {}
    pub fn getDeviceInfo(_: *const Context) DeviceInfo {
        return .{
            .platform = .auto,
            .os_version = "unknown",
            .device_model = "unknown",
            .screen_width = 0,
            .screen_height = 0,
            .battery_level = 0.0,
            .is_charging = false,
        };
    }
};

// Module-level lifecycle — delegate to StubFeature helpers (Context above is custom).
const _Stub = stub_helpers.StubFeature(MobileConfig, MobileError);
pub const init = _Stub.init;
pub const deinit = _Stub.deinit;
pub const isEnabled = _Stub.isEnabled;
pub const isInitialized = _Stub.isInitialized;
pub fn getLifecycleState() LifecycleState {
    return .terminated;
}
pub fn readSensor(_: []const u8) MobileError!SensorData {
    return error.FeatureDisabled;
}
pub fn sendNotification(_: []const u8, _: []const u8) MobileError!void {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
