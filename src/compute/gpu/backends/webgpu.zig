//! WebGPU backend implementation
//!
//! Provides WebGPU-specific kernel compilation and execution.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

var webgpu_initialized = false;

pub fn init() !void {
    if (webgpu_initialized) return;

    if (builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64) {
        webgpu_initialized = true;
        return;
    }

    if (!tryLoadWebGpu()) {
        return error.WebGpuNotAvailable;
    }

    webgpu_initialized = true;
}

pub fn deinit() void {
    webgpu_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    return fallback.compileKernel(allocator, source);
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    return fallback.launchKernel(allocator, kernel_handle, config, args);
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    fallback.destroyKernel(allocator, kernel_handle);
}

pub fn createCommandEncoder() !*anyopaque {
    return fallback.createOpaqueHandle(WgpuCommandEncoder, .{ .encoder = null });
}

pub fn destroyCommandEncoder(encoder: *anyopaque) void {
    fallback.destroyOpaqueHandle(WgpuCommandEncoder, encoder);
}

fn tryLoadWebGpu() bool {
    const lib_names = [_][]const u8{
        "wgpu_native.dll",
        "libwgpu_native.so",
        "libwgpu_native.dylib",
        "dawn_native.dll",
        "libdawn_native.so",
    };
    return shared.tryLoadAny(lib_names[0..]);
}

const WgpuComputePipeline = struct {
    pipeline: ?*anyopaque,
};

const WgpuCommandEncoder = struct {
    encoder: ?*anyopaque,
};
