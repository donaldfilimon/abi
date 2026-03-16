//! Mobile Stub Module — no-op when mobile is disabled (default).

const std = @import("std");
const mobile_config = @import("../../core/config/platform.zig");

pub const MobileConfig = mobile_config.MobileConfig;
pub const MobilePlatform = mobile_config.MobileConfig.Platform;
pub const MobileError = error{
    FeatureDisabled,
    PlatformNotSupported,
    SensorUnavailable,
    NotificationFailed,
    OutOfMemory,
    PermissionDenied,
};
pub const LifecycleState = enum { active, background, suspended, terminated };
pub const SensorData = struct { timestamp_ms: u64 = 0, values: [3]f32 = .{ 0, 0, 0 } };

pub const SensorType = enum {
    accelerometer,
    gyroscope,
    magnetometer,
    gps,
    barometer,
    proximity,
    light,
};

pub const Notification = struct {
    title: []const u8,
    body: []const u8,
    priority: Priority = .normal,
    sent_at: i64 = 0,

    pub const Priority = enum { low, normal, high, critical };
};

pub const NotificationEntry = struct {
    title_buf: [256]u8 = @splat(0),
    title_len: u8 = 0,
    body_buf: [512]u8 = @splat(0),
    body_len: u16 = 0,
    priority: Notification.Priority = .normal,
    sent_at: i64 = 0,

    pub fn title(self: *const NotificationEntry) []const u8 {
        return self.title_buf[0..self.title_len];
    }

    pub fn body(self: *const NotificationEntry) []const u8 {
        return self.body_buf[0..self.body_len];
    }
};

pub const Permission = enum {
    camera,
    microphone,
    location,
    notifications,
    storage,
    contacts,
    bluetooth,
};

pub const PermissionStatus = enum {
    granted,
    denied,
    not_requested,
};

pub const DeviceInfo = struct {
    platform: MobilePlatform,
    os_version: []const u8,
    device_model: []const u8,
    screen_width: u32,
    screen_height: u32,
    battery_level: f32,
    is_charging: bool,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MobileConfig,
    state: LifecycleState = .active,
    permissions: [7]PermissionStatus = @splat(.not_requested),
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

pub fn init(_: std.mem.Allocator, _: MobileConfig) MobileError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}
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
