//! Device information for the mobile feature.
//!
//! Returns simulated device metadata based on the configured platform.

const types = @import("types.zig");

const MobilePlatform = types.MobilePlatform;
const DeviceInfo = types.DeviceInfo;

/// Return simulated device information based on the configured platform.
pub fn getDeviceInfo(platform: MobilePlatform) DeviceInfo {
    return switch (platform) {
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

const std = @import("std");

test "getDeviceInfo returns platform-specific data" {
    const ios = getDeviceInfo(.ios);
    try std.testing.expectEqual(MobilePlatform.ios, ios.platform);
    try std.testing.expectEqualStrings("iPhone 15 Pro", ios.device_model);

    const android = getDeviceInfo(.android);
    try std.testing.expectEqual(MobilePlatform.android, android.platform);

    const auto = getDeviceInfo(.auto);
    try std.testing.expectEqualStrings("Simulator", auto.device_model);
}
