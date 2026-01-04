//! OpenGL backend implementation.
//! Provides OpenGL-specific kernel compilation and execution.
const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

var opengl_initialized = false;

pub fn init() !void {
    if (opengl_initialized) return;
    if (!tryLoadOpenGl()) {
        return error.OpenGlNotAvailable;
    }
    opengl_initialized = true;
}

pub fn deinit() void {
    opengl_initialized = false;
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

fn tryLoadOpenGl() bool {
    const lib_names = [_][]const u8{
        "opengl32.dll",
        "libGL.so.1",
        "libGL.so",
        "/System/Library/Frameworks/OpenGL.framework/OpenGL",
    };
    return shared.tryLoadAny(lib_names[0..]);
}

const GlProgram = struct {
    handle: ?*anyopaque,
};
