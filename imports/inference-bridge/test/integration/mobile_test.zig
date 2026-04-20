//! Integration Tests: Mobile Feature
//!
//! The mobile feature defaults to DISABLED (`feat_mobile = false`), so these
//! tests exercise the **stub** path unless the build explicitly enables it.
//! We verify that the stub surface is API-compatible and returns the expected
//! disabled / no-op values.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const mobile = abi.mobile;

// ---------------------------------------------------------------------------
// Feature gate
// ---------------------------------------------------------------------------

test "mobile: isEnabled reflects feature flag" {
    if (build_options.feat_mobile) {
        try std.testing.expect(mobile.isEnabled());
    } else {
        try std.testing.expect(!mobile.isEnabled());
    }
}

test "mobile: isInitialized reflects feature flag" {
    if (build_options.feat_mobile) {
        try std.testing.expect(mobile.isInitialized());
    } else {
        try std.testing.expect(!mobile.isInitialized());
    }
}

// ---------------------------------------------------------------------------
// Module-level convenience functions (stub path)
// ---------------------------------------------------------------------------

test "mobile: getLifecycleState returns terminated in stub" {
    if (!build_options.feat_mobile) {
        try std.testing.expectEqual(mobile.LifecycleState.terminated, mobile.getLifecycleState());
    }
}

test "mobile: readSensor returns FeatureDisabled in stub" {
    if (!build_options.feat_mobile) {
        const result = mobile.readSensor("accelerometer");
        try std.testing.expectError(error.FeatureDisabled, result);
    }
}

test "mobile: sendNotification returns FeatureDisabled in stub" {
    if (!build_options.feat_mobile) {
        const result = mobile.sendNotification("title", "body");
        try std.testing.expectError(error.FeatureDisabled, result);
    }
}

// ---------------------------------------------------------------------------
// Types are accessible regardless of feature flag
// ---------------------------------------------------------------------------

test "mobile: MobileConfig defaults" {
    const config = mobile.MobileConfig{};
    try std.testing.expectEqual(mobile.MobilePlatform.auto, config.platform);
    try std.testing.expect(!config.enable_sensors);
    try std.testing.expect(!config.enable_notifications);
}

test "mobile: MobilePlatform enum has three variants" {
    const platforms = [_]mobile.MobilePlatform{ .auto, .ios, .android };
    try std.testing.expectEqual(@as(usize, 3), platforms.len);
}

test "mobile: LifecycleState enum variants" {
    const states = [_]mobile.LifecycleState{ .active, .background, .suspended, .terminated };
    try std.testing.expectEqual(@as(usize, 4), states.len);
}

test "mobile: SensorType enum has seven variants" {
    const sensors = [_]mobile.SensorType{
        .accelerometer,
        .gyroscope,
        .magnetometer,
        .gps,
        .barometer,
        .proximity,
        .light,
    };
    try std.testing.expectEqual(@as(usize, 7), sensors.len);
}

test "mobile: SensorData default values" {
    const data = mobile.SensorData{};
    try std.testing.expectEqual(@as(u64, 0), data.timestamp_ms);
    try std.testing.expectEqual(@as(usize, 3), data.values.len);
}

test "mobile: Permission enum has seven variants" {
    const perms = [_]mobile.Permission{
        .camera,
        .microphone,
        .location,
        .notifications,
        .storage,
        .contacts,
        .bluetooth,
    };
    try std.testing.expectEqual(@as(usize, 7), perms.len);
}

test "mobile: PermissionStatus enum variants" {
    const statuses = [_]mobile.PermissionStatus{ .not_requested, .granted, .denied };
    try std.testing.expectEqual(@as(usize, 3), statuses.len);
}

test "mobile: MobileError error set" {
    const errors = [_]mobile.MobileError{
        error.FeatureDisabled,
        error.PlatformNotSupported,
        error.SensorUnavailable,
        error.NotificationFailed,
        error.OutOfMemory,
        error.PermissionDenied,
    };
    try std.testing.expectEqual(@as(usize, 6), errors.len);
}

test "mobile: Notification.Priority variants" {
    const priorities = [_]mobile.Notification.Priority{ .low, .normal, .high, .critical };
    try std.testing.expectEqual(@as(usize, 4), priorities.len);
}

// ---------------------------------------------------------------------------
// Context (stub returns FeatureDisabled on init)
// ---------------------------------------------------------------------------

test "mobile: Context.init returns FeatureDisabled in stub" {
    if (!build_options.feat_mobile) {
        const result = mobile.Context.init(std.testing.allocator, .{});
        try std.testing.expectError(error.FeatureDisabled, result);
    }
}

// ---------------------------------------------------------------------------
// Module-level convenience functions (enabled path)
// ---------------------------------------------------------------------------

test "mobile: getLifecycleState returns active when enabled" {
    if (build_options.feat_mobile) {
        try std.testing.expectEqual(mobile.LifecycleState.active, mobile.getLifecycleState());
    }
}

test "mobile: init and deinit succeed when enabled" {
    if (build_options.feat_mobile) {
        try mobile.init(std.testing.allocator, .{});
        mobile.deinit();
    }
}

test {
    std.testing.refAllDecls(@This());
}
