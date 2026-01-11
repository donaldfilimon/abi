//! Stub for GPU feature when disabled
const std = @import("std");
const compute_gpu = @import("../../compute/gpu/mod.zig");

pub const Backend = compute_gpu.Backend;
pub const DeviceInfo = compute_gpu.DeviceInfo;
pub const BackendAvailability = compute_gpu.BackendAvailability;
pub const GpuError = compute_gpu.GpuError;

pub fn moduleEnabled() bool {
    return false;
}

pub fn backendAvailability(backend: Backend) BackendAvailability {
    _ = backend;
    return BackendAvailability{
        .available = false,
        .reason = "GPU disabled",
        .device_count = 0,
        .level = .none,
    };
}

pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    _ = allocator;
    return &.{};
}

pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    _ = allocator;
    return &.{};
}

pub fn init(allocator: std.mem.Allocator) GpuError!void {
    _ = allocator;
    return error.GpuDisabled;
}

pub fn deinit() void {}
