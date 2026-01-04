//! WebGL2 backend implementation.
//! WebGL2 does not expose compute shaders, so kernel compilation is unsupported.
const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");

var webgl2_initialized = false;

pub fn init() !void {
    if (webgl2_initialized) return;
    if (!shared.isWebTarget()) {
        return error.WebGl2NotSupported;
    }
    webgl2_initialized = true;
}

pub fn deinit() void {
    webgl2_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    _ = allocator;
    _ = source;
    return types.KernelError.UnsupportedBackend;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;
    return types.KernelError.UnsupportedBackend;
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    _ = allocator;
    _ = kernel_handle;
}
