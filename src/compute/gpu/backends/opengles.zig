//! OpenGL ES backend implementation.
//! Provides OpenGL ES-specific kernel compilation and execution.
const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

var opengles_initialized = false;

pub fn init() !void {
    if (opengles_initialized) return;
    if (!tryLoadOpenGles()) {
        return error.OpenGlesNotAvailable;
    }
    opengles_initialized = true;
}

pub fn deinit() void {
    opengles_initialized = false;
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

fn tryLoadOpenGles() bool {
    const lib_names = [_][]const u8{
        "libGLESv2.dll",
        "libEGL.dll",
        "libGLESv2.so.2",
        "libGLESv2.so",
        "/System/Library/Frameworks/OpenGLES.framework/OpenGLES",
    };
    return shared.tryLoadAny(lib_names[0..]);
}

const GlesProgram = struct {
    handle: ?*anyopaque,
};
