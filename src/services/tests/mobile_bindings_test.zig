//! Mobile Bindings Validation Tests
//!
//! Exercises the mobile Context API that underpins the C bindings
//! (`abi_mobile_*` exports in `bindings/c/src/abi_c.zig`).
//!
//! Each test mirrors a C-binding call path:
//!   1. Lifecycle:      init → use → destroy (no leaks)
//!   2. Sensor read:    readSensor returns valid data for every SensorType
//!   3. Device info:    getDeviceInfo returns populated DeviceInfo
//!   4. Permissions:    check (not_requested) → request (granted) → re-check (granted)
//!   5. Notifications:  send → getCount (1) → clear → getCount (0)

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const Context = abi.mobile.Context;
const SensorType = abi.mobile.SensorType;
const Permission = abi.mobile.Permission;
const PermissionStatus = abi.mobile.PermissionStatus;
const Notification = abi.mobile.Notification;
const MobilePlatform = abi.mobile.MobilePlatform;

// ============================================================================
// 1. Lifecycle: init → use → destroy (leak detection via DebugAllocator)
// ============================================================================

test "mobile bindings: lifecycle — init, use, destroy with no leaks" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("memory leak detected in mobile lifecycle test");
    }
    const allocator = gpa.allocator();

    const ctx = try Context.init(allocator, .{});
    // Exercise a few operations before teardown.
    _ = try ctx.readSensor(.accelerometer);
    _ = ctx.getDeviceInfo();
    _ = ctx.checkPermission(.camera);
    ctx.deinit();
}

test "mobile bindings: lifecycle — repeated init/destroy cycles" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("memory leak detected in repeated lifecycle test");
    }
    const allocator = gpa.allocator();

    for (0..5) |_| {
        const ctx = try Context.init(allocator, .{});
        try ctx.sendNotification("ping", "pong", .normal);
        ctx.deinit();
    }
}

// ============================================================================
// 2. Sensor read: valid data for each SensorType
// ============================================================================

test "mobile bindings: sensor — accelerometer returns gravity on z-axis" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{ .enable_sensors = true });
    defer ctx.deinit();

    const data = try ctx.readSensor(.accelerometer);
    try std.testing.expect(data.timestamp_ms > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.values[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 9.81), data.values[2], 0.01);
}

test "mobile bindings: sensor — all types return non-zero timestamp" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const sensor_types = [_]SensorType{
        .accelerometer,
        .gyroscope,
        .magnetometer,
        .gps,
        .barometer,
        .proximity,
        .light,
    };

    for (sensor_types) |st| {
        const data = try ctx.readSensor(st);
        try std.testing.expect(data.timestamp_ms > 0);
        // All sensor data should have exactly 3 values
        try std.testing.expectEqual(@as(usize, 3), data.values.len);
    }
}

test "mobile bindings: sensor — gps returns San Francisco coords" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const gps = try ctx.readSensor(.gps);
    try std.testing.expectApproxEqAbs(@as(f32, 37.7749), gps.values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -122.4194), gps.values[1], 0.01);
}

test "mobile bindings: sensor — barometer returns standard pressure" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const baro = try ctx.readSensor(.barometer);
    try std.testing.expectApproxEqAbs(@as(f32, 1013.25), baro.values[0], 0.01);
}

// ============================================================================
// 3. Device info: populated DeviceInfo per platform
// ============================================================================

test "mobile bindings: device info — ios platform" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{ .platform = .ios });
    defer ctx.deinit();

    const info = ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.ios, info.platform);
    try std.testing.expectEqualStrings("iPhone 15 Pro", info.device_model);
    try std.testing.expectEqualStrings("17.4.1", info.os_version);
    try std.testing.expect(info.screen_width > 0);
    try std.testing.expect(info.screen_height > 0);
    try std.testing.expect(info.battery_level >= 0.0 and info.battery_level <= 1.0);
}

test "mobile bindings: device info — android platform" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{ .platform = .android });
    defer ctx.deinit();

    const info = ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.android, info.platform);
    try std.testing.expectEqualStrings("Pixel 8 Pro", info.device_model);
    try std.testing.expectEqualStrings("14.0", info.os_version);
    try std.testing.expect(info.is_charging);
}

test "mobile bindings: device info — auto (simulator) platform" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{ .platform = .auto });
    defer ctx.deinit();

    const info = ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.auto, info.platform);
    try std.testing.expectEqualStrings("Simulator", info.device_model);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), info.battery_level, 0.001);
    try std.testing.expect(info.is_charging);
}

// ============================================================================
// 4. Permissions: check → request → re-check flow
// ============================================================================

test "mobile bindings: permissions — initial state is not_requested" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    const all_perms = [_]Permission{
        .camera,
        .microphone,
        .location,
        .notifications,
        .storage,
        .contacts,
        .bluetooth,
    };
    for (all_perms) |perm| {
        try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(perm));
    }
}

test "mobile bindings: permissions — request grants, re-check confirms" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    // check → not_requested
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.camera));

    // request → granted
    const result = ctx.requestPermission(.camera);
    try std.testing.expectEqual(PermissionStatus.granted, result);

    // re-check → granted
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.camera));
}

test "mobile bindings: permissions — revoke after grant" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    _ = ctx.requestPermission(.location);
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.location));

    ctx.revokePermission(.location);
    try std.testing.expectEqual(PermissionStatus.denied, ctx.checkPermission(.location));
}

test "mobile bindings: permissions — independent across types" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    _ = ctx.requestPermission(.camera);
    _ = ctx.requestPermission(.microphone);

    // Revoke only camera
    ctx.revokePermission(.camera);

    try std.testing.expectEqual(PermissionStatus.denied, ctx.checkPermission(.camera));
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.microphone));
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.bluetooth));
}

// ============================================================================
// 5. Notifications: send → count → clear → count
// ============================================================================

test "mobile bindings: notifications — send, count, clear cycle" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{ .enable_notifications = true });
    defer ctx.deinit();

    // Initially empty
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());

    // Send one
    try ctx.sendNotification("Alert", "Something happened", .high);
    try std.testing.expectEqual(@as(usize, 1), ctx.getNotificationCount());

    // Clear
    ctx.clearNotifications();
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());
}

test "mobile bindings: notifications — multiple sends accumulate" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    try ctx.sendNotification("A", "first", .low);
    try ctx.sendNotification("B", "second", .normal);
    try ctx.sendNotification("C", "third", .critical);
    try std.testing.expectEqual(@as(usize, 3), ctx.getNotificationCount());

    // Verify content of first entry
    const entry = ctx.notification_log.items[0];
    try std.testing.expectEqualStrings("A", entry.title());
    try std.testing.expectEqualStrings("first", entry.body());
    try std.testing.expectEqual(Notification.Priority.low, entry.priority);
    try std.testing.expect(entry.sent_at > 0);
}

test "mobile bindings: notifications — clear is idempotent" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    const ctx = try Context.init(std.testing.allocator, .{});
    defer ctx.deinit();

    ctx.clearNotifications();
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());

    try ctx.sendNotification("X", "body", .normal);
    ctx.clearNotifications();
    ctx.clearNotifications();
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());
}

// ============================================================================
// End-to-end: full C-binding call sequence in a single context
// ============================================================================

test "mobile bindings: e2e — full lifecycle matching C export call order" {
    if (!build_options.feat_mobile) return error.SkipZigTest;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("memory leak in e2e mobile test");
    }
    const allocator = gpa.allocator();

    // abi_mobile_init
    const ctx = try Context.init(allocator, .{ .platform = .ios });

    // abi_mobile_read_sensor (accelerometer = 0)
    const sensor = try ctx.readSensor(.accelerometer);
    try std.testing.expect(sensor.timestamp_ms > 0);

    // abi_mobile_get_device_info
    const info = ctx.getDeviceInfo();
    try std.testing.expectEqual(MobilePlatform.ios, info.platform);

    // abi_mobile_check_permission (camera = 0) → not_requested
    try std.testing.expectEqual(PermissionStatus.not_requested, ctx.checkPermission(.camera));

    // abi_mobile_request_permission (camera = 0) → granted
    const perm_result = ctx.requestPermission(.camera);
    try std.testing.expectEqual(PermissionStatus.granted, perm_result);

    // abi_mobile_check_permission again → granted
    try std.testing.expectEqual(PermissionStatus.granted, ctx.checkPermission(.camera));

    // abi_mobile_send_notification
    try ctx.sendNotification("Test", "Hello from bindings", .normal);

    // abi_mobile_get_notification_count → 1
    try std.testing.expectEqual(@as(usize, 1), ctx.getNotificationCount());

    // abi_mobile_clear_notifications
    ctx.clearNotifications();

    // abi_mobile_get_notification_count → 0
    try std.testing.expectEqual(@as(usize, 0), ctx.getNotificationCount());

    // abi_mobile_destroy
    ctx.deinit();
}
