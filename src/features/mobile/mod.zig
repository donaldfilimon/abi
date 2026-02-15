//! Mobile Module
//!
//! Platform lifecycle, sensors, notifications, and mobile-optimized storage.

const std = @import("std");
const mobile_config = @import("../../core/config/mobile.zig");

pub const MobileConfig = mobile_config.MobileConfig;
pub const MobilePlatform = mobile_config.MobileConfig.Platform;

pub const MobileError = error{
    FeatureDisabled,
    PlatformNotSupported,
    SensorUnavailable,
    NotificationFailed,
    OutOfMemory,
};

pub const LifecycleState = enum {
    active,
    background,
    suspended,
    terminated,
};

pub const SensorData = struct {
    timestamp_ms: u64 = 0,
    values: [3]f32 = .{ 0, 0, 0 },
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MobileConfig,
    state: LifecycleState = .active,

    pub fn init(allocator: std.mem.Allocator, config: MobileConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

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
pub fn readSensor(_: []const u8) MobileError!SensorData {
    return .{};
}
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
