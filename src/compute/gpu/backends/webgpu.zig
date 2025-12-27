//! WebGPU backend implementation
//!
//! Provides WebGPU-specific kernel compilation and execution.

const std = @import("std");
const kernels = @import("../kernels.zig");
const builtin = @import("builtin");

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
    source: kernels.KernelSource,
) !*anyopaque {
    _ = source;

    const kernel_handle = try allocator.create(WgpuComputePipeline);
    kernel_handle.* = .{ .pipeline = null };

    return kernel_handle;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: kernels.KernelConfig,
    args: []const ?*const anyopaque,
) !void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;

    return error.WebGpuLaunchFailed;
}

pub fn destroyKernel(kernel_handle: *anyopaque) void {
    const wgpu_kernel: *WgpuComputePipeline = @ptrCast(@alignCast(kernel_handle));
    std.heap.page_allocator.destroy(wgpu_kernel);
}

pub fn createCommandEncoder() !*anyopaque {
    const encoder = try std.heap.page_allocator.create(WgpuCommandEncoder);
    encoder.* = .{ .encoder = null };
    return encoder;
}

pub fn destroyCommandEncoder(encoder: *anyopaque) void {
    const wgpu_encoder: *WgpuCommandEncoder = @ptrCast(@alignCast(encoder));
    std.heap.page_allocator.destroy(wgpu_encoder);
}

fn tryLoadWebGpu() bool {
    const lib_names = [_][]const u8{
        "wgpu_native.dll",
        "libwgpu_native.so",
        "libwgpu_native.dylib",
        "dawn_native.dll",
        "libdawn_native.so",
    };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |_| {
            return true;
        } else |_| {}
    }
    return false;
}

const WgpuComputePipeline = struct {
    pipeline: ?*anyopaque,
};

const WgpuCommandEncoder = struct {
    encoder: ?*anyopaque,
};
