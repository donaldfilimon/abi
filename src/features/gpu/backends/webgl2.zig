//! WebGL2 backend implementation.
//!
//! WebGL2 does not expose compute shaders, so this backend only provides
//! stub implementations. For actual GPU compute in web environments, use WebGPU.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");

pub const WebGl2Error = error{
    NotSupported,
    PlatformNotSupported,
    ComputeShadersUnavailable,
};

var webgl2_initialized = false;
var init_mutex = sync.Mutex{};

pub fn init() WebGl2Error!void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (webgl2_initialized) return;

    if (!shared.isWebTarget()) {
        return WebGl2Error.PlatformNotSupported;
    }

    webgl2_initialized = true;
    std.log.debug("WebGL2 backend initialized (compute shaders not supported)", .{});
}

pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (!webgl2_initialized) return;

    webgl2_initialized = false;
    std.log.debug("WebGL2 backend deinitialized", .{});
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) (types.KernelError || WebGl2Error)!*anyopaque {
    _ = allocator;
    _ = source;
    std.log.warn("WebGL2 does not support compute shaders - use WebGPU for compute workloads", .{});
    return WebGl2Error.ComputeShadersUnavailable;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) (types.KernelError || WebGl2Error)!void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;
    return WebGl2Error.ComputeShadersUnavailable;
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    _ = allocator;
    _ = kernel_handle;
}

pub fn allocateDeviceMemory(size: usize) WebGl2Error!*anyopaque {
    _ = size;
    return WebGl2Error.ComputeShadersUnavailable;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    _ = ptr;
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) WebGl2Error!void {
    _ = dst;
    _ = src;
    _ = size;
    return WebGl2Error.ComputeShadersUnavailable;
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) WebGl2Error!void {
    _ = dst;
    _ = src;
    _ = size;
    return WebGl2Error.ComputeShadersUnavailable;
}

// ============================================================================
// Tests
// ============================================================================

test "WebGl2Error enum covers all cases" {
    const errors = [_]WebGl2Error{
        error.NotSupported,
        error.PlatformNotSupported,
        error.ComputeShadersUnavailable,
    };
    try std.testing.expectEqual(@as(usize, 3), errors.len);
}

test "compileKernel returns ComputeShadersUnavailable" {
    const result = compileKernel(std.testing.allocator, .{ .wgsl = "" });
    try std.testing.expectError(error.ComputeShadersUnavailable, result);
}

test "allocateDeviceMemory returns ComputeShadersUnavailable" {
    const result = allocateDeviceMemory(1024);
    try std.testing.expectError(error.ComputeShadersUnavailable, result);
}
