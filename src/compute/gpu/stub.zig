//! Stub for GPU feature when disabled
const std = @import("std");
const compute_gpu = @import("mod.zig");

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
        .enabled = false,
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

pub const GpuConfig = struct {
    preferred_backend: ?Backend = null,
    enable_profiling: bool = false,
    memory_mode: MemoryMode = .automatic,
    multi_gpu: bool = false,
    load_balance_strategy: LoadBalanceStrategy = .round_robin,
};

pub const MemoryMode = enum {
    automatic,
    explicit,
    unified,
};

pub const LoadBalanceStrategy = enum {
    round_robin,
    memory_aware,
    compute_aware,
    manual,
};

pub const Gpu = struct {
    allocator: std.mem.Allocator,
    config: GpuConfig,

    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) GpuError!Gpu {
        _ = allocator;
        _ = config;
        return error.GpuDisabled;
    }

    pub fn deinit(self: *Gpu) void {
        _ = self;
    }

    pub fn isAvailable(_: *const Gpu) bool {
        return false;
    }

    pub fn getActiveDevice(_: *const Gpu) ?*const DeviceInfo {
        return null;
    }

    pub fn getBackend(_: *const Gpu) ?Backend {
        return null;
    }
};
