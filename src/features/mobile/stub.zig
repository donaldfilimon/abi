//! Mobile Stub Module — no-op when mobile is disabled (default).

const std = @import("std");
const mobile_config = @import("../../core/config/mobile.zig");

pub const MobileConfig = mobile_config.MobileConfig;
pub const MobilePlatform = mobile_config.MobileConfig.Platform;
pub const MobileError = error{ FeatureDisabled, PlatformNotSupported, SensorUnavailable, NotificationFailed, OutOfMemory };
pub const LifecycleState = enum { active, background, suspended, terminated };
pub const SensorData = struct { timestamp_ms: u64 = 0, values: [3]f32 = .{ 0, 0, 0 } };

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: MobileConfig,
    state: LifecycleState = .active,

    pub fn init(_: std.mem.Allocator, _: MobileConfig) !*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
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
