//! DirectML Backend for Windows GPU Acceleration
//!
//! Provides GPU compute via Microsoft's DirectML API on Windows.
//! Falls back gracefully on non-Windows platforms.
//!
//! DirectML is a high-performance, hardware-accelerated DirectX 12 library
//! for machine learning operators. It provides GPU acceleration for common
//! ML workloads across all DirectX 12-capable hardware.

const std = @import("std");
const builtin = @import("builtin");

pub const loader = @import("loader.zig");
pub const operators = @import("operators.zig");

pub fn isAvailable() bool {
    if (builtin.os.tag != .windows) return false;
    return loader.canLoadDirectML();
}

pub const DirectMlBackendError = error{
    BackendNotSupported,
};

/// Compile a kernel source into an opaque handle.
/// DirectML does not support raw kernel compilation — returns BackendNotSupported.
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: anytype,
) DirectMlBackendError!*anyopaque {
    _ = allocator;
    _ = source;
    return DirectMlBackendError.BackendNotSupported;
}

/// Launch a previously compiled kernel.
/// DirectML does not support raw kernel dispatch — returns BackendNotSupported.
pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: anytype,
    args: []const ?*const anyopaque,
) DirectMlBackendError!void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;
    return DirectMlBackendError.BackendNotSupported;
}

/// Allocate device memory.
/// DirectML manages memory through D3D12 resources — manual allocation not supported.
pub fn allocateDeviceMemory(size: usize) DirectMlBackendError!*anyopaque {
    _ = size;
    return DirectMlBackendError.BackendNotSupported;
}

/// Free device memory.
/// No-op since allocateDeviceMemory always fails on this backend.
pub fn freeDeviceMemory(ptr: *anyopaque) void {
    _ = ptr;
}

test "DirectML availability" {
    const available = isAvailable();
    if (builtin.os.tag != .windows) {
        try std.testing.expect(!available);
    }
}

test "DirectML submodule imports" {
    // Verify submodules are importable (compile-time check)
    _ = loader.DirectMlDevice;
    _ = operators.DmlMatMul;
    _ = operators.DmlConvolution;
    _ = operators.DmlActivation;
}

test {
    std.testing.refAllDecls(@This());
}
