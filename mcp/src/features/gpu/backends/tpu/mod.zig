//! TPU Backend for Google Tensor Processing Units
//!
//! Provides GPU compute via Google's TPU hardware.
//! Falls back gracefully on non-TPU platforms or when TPU is unavailable.
//!
//! TPUs (Tensor Processing Units) are Google's custom ASICs designed
//! for machine learning workloads. This backend provides a stub
//! implementation that returns BackendNotSupported when TPU is not available.

const std = @import("std");
const builtin = @import("builtin");

pub const loader = @import("tpu_loader.zig");
pub const operators = @import("tpu_operators.zig");

pub fn isAvailable() bool {
    if (builtin.os.tag != .linux and builtin.os.tag != .macos) return false;
    return loader.canLoadTPU();
}

pub const TpuBackendError = error{
    BackendNotSupported,
    TpuNotFound,
    TpuInitializationFailed,
};

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: anytype,
) TpuBackendError!*anyopaque {
    _ = allocator;
    _ = source;
    return TpuBackendError.BackendNotSupported;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: anytype,
    args: []const ?*const anyopaque,
) TpuBackendError!void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;
    return TpuBackendError.BackendNotSupported;
}

pub fn allocateDeviceMemory(size: usize) TpuBackendError!*anyopaque {
    _ = size;
    return TpuBackendError.BackendNotSupported;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    _ = ptr;
}

test {
    std.testing.refAllDecls(@This());
}
