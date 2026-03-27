//! Shared types for the mobile feature (mod + stub).

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

pub const LifecycleState = enum {
    active,
    background,
    suspended,
    terminated,
};

// ============================================================================
// Sensor Types
// ============================================================================

pub const SensorType = enum {
    accelerometer,
    gyroscope,
    magnetometer,
    gps,
    barometer,
    proximity,
    light,
};

pub const SensorData = struct {
    timestamp_ms: u64 = 0,
    values: [3]f32 = .{ 0, 0, 0 },
};

// ============================================================================
// Notification Types
// ============================================================================

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

// ============================================================================
// Permission Types
// ============================================================================

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

pub const permission_count = @typeInfo(Permission).@"enum".fields.len;

// ============================================================================
// Device Info
// ============================================================================

pub const DeviceInfo = struct {
    platform: MobilePlatform,
    os_version: []const u8,
    device_model: []const u8,
    screen_width: u32,
    screen_height: u32,
    battery_level: f32,
    is_charging: bool,
};
