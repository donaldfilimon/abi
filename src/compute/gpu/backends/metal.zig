//! Metal backend implementation
//!
//! Provides Metal-specific kernel compilation and execution.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

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

pub fn createCommandBuffer() !*anyopaque {
    return fallback.createOpaqueHandle(MtlCommandBuffer, .{ .buffer = null });
}

pub fn destroyCommandBuffer(buffer: *anyopaque) void {
    fallback.destroyOpaqueHandle(MtlCommandBuffer, buffer);
}

fn tryLoadMetal() bool {
    const lib_names = [_][]const u8{"/System/Library/Frameworks/Metal.framework/Metal"};
    return shared.tryLoadAny(lib_names[0..]);
}

const MtlComputePipeline = struct {
    pipeline: ?*anyopaque,
};

const MtlCommandBuffer = struct {
    buffer: ?*anyopaque,
};
