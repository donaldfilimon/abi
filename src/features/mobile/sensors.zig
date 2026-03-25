//! Sensor simulation for the mobile feature.
//!
//! Provides simulated sensor readings (accelerometer, gyroscope, GPS, etc.)
//! for development and testing on non-mobile platforms.

const std = @import("std");
const types = @import("types.zig");

const SensorType = types.SensorType;
const SensorData = types.SensorData;
const MobileError = types.MobileError;

/// Read a simulated sensor value based on the sensor type.
pub fn readSensor(sensor_type: SensorType) MobileError!SensorData {
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

/// Legacy module-level readSensor (string-based, returns default data).
pub fn readSensorLegacy(_: []const u8) MobileError!SensorData {
    return .{};
}
