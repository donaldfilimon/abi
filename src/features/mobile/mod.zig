//! Mobile Module
//!
//! Platform lifecycle, sensors, notifications, permissions, and device info.
//! Provides simulated mobile platform behavior for development and testing.

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

const permission_count = @typeInfo(Permission).@"enum".fields.len;

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

// ============================================================================
// Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MobileConfig,
    state: LifecycleState = .active,
    permissions: [permission_count]PermissionStatus = @splat(.not_requested),
    notification_log: std.ArrayListUnmanaged(NotificationEntry) = .empty,

    pub fn init(allocator: std.mem.Allocator, config: MobileConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.notification_log.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Read a simulated sensor value based on the sensor type.
    pub fn readSensor(self: *Context, sensor_type: SensorType) MobileError!SensorData {
        _ = self;
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const timestamp_ms: u64 = @intCast(@as(i64, @intCast(ts.sec)) * 1000 +
            @divTrunc(@as(i64, ts.nsec), 1_000_000));

        const values: [3]f32 = switch (sensor_type) {
            .accelerometer => .{ 0.0, 0.0, 9.81 },
            .gyroscope => .{ 0.0, 0.0, 0.0 },
            .magnetometer => .{ 25.0, 0.0, 45.0 },
            .gps => .{ 37.7749, -122.4194, 0.0 },
            .barometer => .{ 1013.25, 0.0, 0.0 },
            .proximity => .{ 1.0, 0.0, 0.0 },
            .light => .{ 500.0, 0.0, 0.0 },
        };

        return .{
            .timestamp_ms = timestamp_ms,
            .values = values,
        };
    }

    /// Send a notification and track it in the log.
    pub fn sendNotification(
        self: *Context,
        title: []const u8,
        body_text: []const u8,
        priority: Notification.Priority,
    ) MobileError!void {
        var entry: NotificationEntry = .{
            .priority = priority,
        };

        const t_len: u8 = @intCast(@min(title.len, entry.title_buf.len));
        @memcpy(entry.title_buf[0..t_len], title[0..t_len]);
        entry.title_len = t_len;

        const b_len: u16 = @intCast(@min(body_text.len, entry.body_buf.len));
        @memcpy(entry.body_buf[0..b_len], body_text[0..b_len]);
        entry.body_len = b_len;

        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        entry.sent_at = @intCast(ts.sec);

        self.notification_log.append(self.allocator, entry) catch return error.OutOfMemory;
    }

    /// Return the number of tracked notifications.
    pub fn getNotificationCount(self: *const Context) usize {
        return self.notification_log.items.len;
    }

    /// Clear all tracked notifications.
    pub fn clearNotifications(self: *Context) void {
        self.notification_log.clearRetainingCapacity();
    }

    /// Check the current status of a permission.
    pub fn checkPermission(self: *const Context, perm: Permission) PermissionStatus {
        return self.permissions[@intFromEnum(perm)];
    }

    /// Request a permission (simulated: always grants).
    pub fn requestPermission(self: *Context, perm: Permission) PermissionStatus {
        self.permissions[@intFromEnum(perm)] = .granted;
        return .granted;
    }

    /// Revoke a previously granted permission.
    pub fn revokePermission(self: *Context, perm: Permission) void {
        self.permissions[@intFromEnum(perm)] = .denied;
    }

    /// Return simulated device information based on the configured platform.
    pub fn getDeviceInfo(self: *const Context) DeviceInfo {
        return switch (self.config.platform) {
            .ios => .{
                .platform = .ios,
                .os_version = "17.4.1",
                .device_model = "iPhone 15 Pro",
                .screen_width = 1179,
                .screen_height = 2556,
                .battery_level = 0.85,
                .is_charging = false,
            },
            .android => .{
                .platform = .android,
                .os_version = "14.0",
                .device_model = "Pixel 8 Pro",
                .screen_width = 1344,
                .screen_height = 2992,
                .battery_level = 0.72,
                .is_charging = true,
            },
            .auto => .{
                .platform = .auto,
                .os_version = "1.0.0",
                .device_model = "Simulator",
                .screen_width = 1080,
                .screen_height = 1920,
                .battery_level = 1.0,
                .is_charging = true,
            },
        };
    }
};

// ============================================================================
// Module-level convenience functions (backwards compatible)
// ============================================================================

pub fn init(_: std.mem.Allocator, _: MobileConfig) MobileError!void {}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return true;
}
pub fn isInitialized() bool {
    return true;
}

pub fn getLifecycleState() LifecycleState {
    return .active;
}

/// Legacy module-level readSensor (string-based, returns default data).
pub fn readSensor(_: []const u8) MobileError!SensorData {
    return .{};
}

/// Legacy module-level sendNotification (no tracking).
pub fn sendNotification(_: []const u8, _: []const u8) MobileError!void {}

// ============================================================================
// Tests
// ============================================================================

test "Context - init and deinit" {
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();
    try std.testing.expectEqual(LifecycleState.active, ctx.state);
    try std.testing.expectEqual(MobilePlatform.auto, ctx.config.platform);
}

test "Context - init with custom config" {
    const ctx = try Context.init(std.testing.allocator, .{
        .platform = .ios,
        .enable_sensors = true,
        .enable_notifications = true,
    });
    defer ctx.deinit();
    try std.testing.expectEqual(MobilePlatform.ios, ctx.config.platform);
    try std.testing.expect(ctx.config.enable_sensors);
    try std.testing.expect(ctx.config.enable_notifications);
}

test "isEnabled returns true" {
    try std.testing.expect(isEnabled());
}

test "isInitialized returns true" {
    try std.testing.expect(isInitialized());
}

test "getLifecycleState returns active" {
    try std.testing.expectEqual(LifecycleState.active, getLifecycleState());
}

test "readSensor returns default SensorData" {
    const data = try readSensor("accelerometer");
    try std.testing.expectEqual(@as(u64, 0), data.timestamp_ms);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.values[0], 0.001);
}

test "sendNotification succeeds" {
    try sendNotification("Test Title", "Test body");
}

test "init and deinit module lifecycle" {
    try init(std.testing.allocator, .{});
    deinit();
}

test "LifecycleState enum variants" {
    const states = [_]LifecycleState{ .active, .background, .suspended, .terminated };
    try std.testing.expectEqual(@as(usize, 4), states.len);
}

test "SensorData default values" {
    const data = SensorData{};
    try std.testing.expectEqual(@as(u64, 0), data.timestamp_ms);
    try std.testing.expectEqual(@as(usize, 3), data.values.len);
}

test "MobileConfig default values" {
    const config = MobileConfig{};
    try std.testing.expectEqual(MobilePlatform.auto, config.platform);
    try std.testing.expect(!config.enable_sensors);
    try std.testing.expect(!config.enable_notifications);
}

test "MobileConfig.defaults matches zero-init" {
    const a = MobileConfig{};
    const b = MobileConfig.defaults();
    try std.testing.expectEqual(a.platform, b.platform);
    try std.testing.expectEqual(a.enable_sensors, b.enable_sensors);
    try std.testing.expectEqual(a.enable_notifications, b.enable_notifications);
}

test "MobilePlatform enum coverage" {
    const platforms = [_]MobilePlatform{ .auto, .ios, .android };
    try std.testing.expectEqual(@as(usize, 3), platforms.len);
    // Verify they are distinct
    try std.testing.expect(platforms[0] != platforms[1]);
    try std.testing.expect(platforms[1] != platforms[2]);
    try std.testing.expect(platforms[0] != platforms[2]);
}

test "Context default state is active" {
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();
    try std.testing.expectEqual(LifecycleState.active, ctx.state);
}

test "Context with android platform" {
    const ctx = try Context.init(std.testing.allocator, .{
        .platform = .android,
        .enable_sensors = false,
        .enable_notifications = true,
    });
    defer ctx.deinit();
    try std.testing.expectEqual(MobilePlatform.android, ctx.config.platform);
    try std.testing.expect(!ctx.config.enable_sensors);
    try std.testing.expect(ctx.config.enable_notifications);
}

test "SensorData custom values" {
    const data = SensorData{
        .timestamp_ms = 12345,
        .values = .{ 1.0, -2.5, 9.81 },
    };
    try std.testing.expectEqual(@as(u64, 12345), data.timestamp_ms);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), data.values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.81), data.values[2], 0.001);
}

test "MobileError error set" {
    const errors = [_]MobileError{
        error.FeatureDisabled,
        error.PlatformNotSupported,
        error.SensorUnavailable,
        error.NotificationFailed,
        error.OutOfMemory,
        error.PermissionDenied,
    };
    try std.testing.expectEqual(@as(usize, 6), errors.len);
}

test "readSensor with different sensor names" {
    // Should return default data regardless of sensor name
    const accel = try readSensor("accelerometer");
    const gyro = try readSensor("gyroscope");
    try std.testing.expectEqual(accel.timestamp_ms, gyro.timestamp_ms);
    try std.testing.expectEqual(accel.values[0], gyro.values[0]);
}

test "multiple init deinit cycles" {
    try init(std.testing.allocator, .{});
    deinit();
    try init(std.testing.allocator, .{ .platform = .ios });
    deinit();
}

// ============================================================================
// New tests: Sensor simulation
// ============================================================================

test "SensorType enum coverage" {
    const types = [_]SensorType{
        .accelerometer,
        .gyroscope,
        .magnetometer,
        .gps,
        .barometer,
        .proximity,
        .light,
    };
    try std.testing.expectEqual(@as(usize, 7), types.len);
    // All distinct
    for (types, 0..) |t, i| {
        for (types[i + 1 ..]) |u| {
            try std.testing.expect(t != u);
        }
    }
}

test "Context.readSensor returns correct values per type" {
    const ctx = try Context.init(std.testing.allocator, .{ .enable_sensors = true });
    defer ctx.deinit();

    // Accelerometer: gravity on z-axis
    const accel = try ctx.readSensor(.accelerometer);
    try std.testing.expect(accel.timestamp_ms > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), accel.values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), accel.values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.81), accel.values[2], 0.001);

    // Gyroscope: stationary
    const gyro = try ctx.readSensor(.gyroscope);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gyro.values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gyro.values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gyro.values[2], 0.001);

    // GPS: San Francisco
    const gps = try ctx.readSensor(.gps);
    try std.testing.expectApproxEqAbs(@as(f32, 37.7749), gps.values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -122.4194), gps.values[1], 0.01);

    // Barometer: standard pressure
    const baro = try ctx.readSensor(.barometer);
    try std.testing.expectApproxEqAbs(@as(f32, 1013.25), baro.values[0], 0.01);

    // Proximity: near
    const prox = try ctx.readSensor(.proximity);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), prox.values[0], 0.001);

    // Light: indoor
    const light = try ctx.readSensor(.light);
    try std.testing.expectApproxEqAbs(@as(f32, 500.0), light.values[0], 0.1);

    // Magnetometer: Earth's field
    const mag = try ctx.readSensor(.magnetometer);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), mag.values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mag.values[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), mag.values[2], 0.01);
}

// ============================================================================
// New tests: Notification tracking
// ============================================================================

test "Notification tracking - send, count, clear" {
    const ctx = try Context.init(std.testing.allocator, .{ .enable_notifications = true });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());

    try ctx.sendNotification("Alert", "Something happened", .high);
    try std.testing.expectEqual(@as(usize, 1), ctx.getNotificationCount());

    try ctx.sendNotification("Info", "All good", .low);
    try std.testing.expectEqual(@as(usize, 2), ctx.getNotificationCount());

    // Verify content
    const entry = ctx.notification_log.items[0];
    try std.testing.expectEqualStrings("Alert", entry.title());
    try std.testing.expectEqualStrings("Something happened", entry.body());
    try std.testing.expectEqual(Notification.Priority.high, entry.priority);
    try std.testing.expect(entry.sent_at > 0);

    // Clear
    ctx.clearNotifications();
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());
}

test "Notification priority variants" {
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const priorities = [_]Notification.Priority{ .low, .normal, .high, .critical };
    for (priorities) |p| {
        try ctx.sendNotification("T", "B", p);
    }
    try std.testing.expectEqual(@as(usize, 4), ctx.getNotificationCount());

    // Verify each priority stored correctly
    for (ctx.notification_log.items, 0..) |entry, i| {
        try std.testing.expectEqual(priorities[i], entry.priority);
    }
}

// ============================================================================
// New tests: Permission system
// ============================================================================

test "Permission lifecycle - not_requested, granted, denied" {
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Initially not requested
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.camera));

    // Request grants it
    const status = ctx.requestPermission(.camera);
    try std.testing.expectEqual(PermissionStatus.granted, status);
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.camera));

    // Revoke sets to denied
    ctx.revokePermission(.camera);
    try std.testing.expectEqual(PermissionStatus.denied, ctx.checkPermission(.camera));
}

test "Multiple permissions are independent" {
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // Grant camera
    _ = ctx.requestPermission(.camera);
    // Grant location
    _ = ctx.requestPermission(.location);

    // Revoke only camera
    ctx.revokePermission(.camera);

    try std.testing.expectEqual(PermissionStatus.denied, ctx.checkPermission(.camera));
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.location));
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.microphone));
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.bluetooth));
}

test "Permission enum coverage" {
    const perms = [_]Permission{
        .camera,
        .microphone,
        .location,
        .notifications,
        .storage,
        .contacts,
        .bluetooth,
    };
    try std.testing.expectEqual(@as(usize, 7), perms.len);

    // Request all, verify all granted
    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();
    for (perms) |p| {
        _ = ctx.requestPermission(p);
    }
    for (perms) |p| {
        try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(p));
    }
}

// ============================================================================
// New tests: Device info
// ============================================================================

test "DeviceInfo returns valid data for each platform" {
    // iOS
    const ios_ctx = try Context.init(std.testing.allocator, .{ .platform = .ios });
    defer ios_ctx.deinit();
    const ios_info = ios_ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.ios, ios_info.platform);
    try std.testing.expectEqualStrings("iPhone 15 Pro", ios_info.device_model);
    try std.testing.expect(ios_info.screen_width > 0);
    try std.testing.expect(ios_info.screen_height > 0);
    try std.testing.expect(ios_info.battery_level >= 0.0 and ios_info.battery_level <= 1.0);

    // Android
    const android_ctx = try Context.init(std.testing.allocator, .{ .platform = .android });
    defer android_ctx.deinit();
    const android_info = android_ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.android, android_info.platform);
    try std.testing.expectEqualStrings("Pixel 8 Pro", android_info.device_model);
    try std.testing.expect(android_info.is_charging);

    // Auto (simulator)
    const auto_ctx = try Context.init(std.testing.allocator, .{ .platform = .auto });
    defer auto_ctx.deinit();
    const auto_info = auto_ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.auto, auto_info.platform);
    try std.testing.expectEqualStrings("Simulator", auto_info.device_model);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), auto_info.battery_level, 0.001);
}

test "DeviceInfo os_version is non-empty" {
    const ctx = try Context.init(std.testing.allocator, .{ .platform = .ios });
    defer ctx.deinit();
    const info = ctx.getDeviceInfo();
    try std.testing.expect(info.os_version.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
