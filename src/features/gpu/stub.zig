//! Stub for GPU feature when disabled
const std = @import("std");

pub const BackendAvailability = struct {
    available: bool,
    reason: []const u8,
    device_count: usize,
    level: enum { none },
};

pub fn moduleEnabled() bool {
    return false;
}

pub fn getAvailableBackends() []const @import("../gpu/mod.zig").Backend {
    return &.{};
}

pub fn backendAvailability(backend: @import("../gpu/mod.zig").Backend) BackendAvailability {
    _ = backend;
    return BackendAvailability{
        .available = false,
        .reason = "GPU disabled",
        .device_count = 0,
        .level = .none,
    };
}

pub fn availableBackends(allocator: std.mem.Allocator) ![]@import("../gpu/mod.zig").Backend {
    _ = allocator;
    return &.{};
}

pub fn listDevices(allocator: std.mem.Allocator) ![]@import("../gpu/mod.zig").DeviceInfo {
    _ = allocator;
    return &.{};
}

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return error.GpuDisabled;
}

pub fn deinit() void {}
