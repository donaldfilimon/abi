//! Metal backend implementation
//!
//! Provides Metal-specific kernel compilation and execution.

const std = @import("std");
const kernels = @import("../kernels.zig");
const builtin = @import("builtin");

var metal_initialized = false;

pub fn init() !void {
    if (metal_initialized) return;

    if (builtin.target.os.tag != .macos) {
        return error.MetalNotSupported;
    }

    if (!tryLoadMetal()) {
        return error.MetalNotAvailable;
    }

    metal_initialized = true;
}

pub fn deinit() void {
    metal_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: kernels.KernelSource,
) !*anyopaque {
    _ = source;

    const kernel_handle = try allocator.create(MtlComputePipeline);
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

    return error.MetalLaunchFailed;
}

pub fn destroyKernel(kernel_handle: *anyopaque) void {
    const mtl_kernel: *MtlComputePipeline = @ptrCast(@alignCast(kernel_handle));
    std.heap.page_allocator.destroy(mtl_kernel);
}

pub fn createCommandBuffer() !*anyopaque {
    const buffer = try std.heap.page_allocator.create(MtlCommandBuffer);
    buffer.* = .{ .buffer = null };
    return buffer;
}

pub fn destroyCommandBuffer(buffer: *anyopaque) void {
    const mtl_buffer: *MtlCommandBuffer = @ptrCast(@alignCast(buffer));
    std.heap.page_allocator.destroy(mtl_buffer);
}

fn tryLoadMetal() bool {
    const lib_names = [_][]const u8{"/System/Library/Frameworks/Metal.framework/Metal"};
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |_| {
            return true;
        } else |_| {}
    }
    return false;
}

const MtlComputePipeline = struct {
    pipeline: ?*anyopaque,
};

const MtlCommandBuffer = struct {
    buffer: ?*anyopaque,
};
