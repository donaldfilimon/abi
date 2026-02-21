const std = @import("std");
const common = @import("common.zig");
const types = @import("../../kernel_types.zig");
const opengl = @import("../opengl.zig");
const opengles = @import("../opengles.zig");

pub fn compileKernel(
    api: common.Api,
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) anyerror!*anyopaque {
    return switch (api) {
        .opengl => opengl.compileKernel(allocator, source),
        .opengles => opengles.compileKernel(allocator, source),
    };
}

pub fn launchKernel(
    api: common.Api,
    allocator: std.mem.Allocator,
    kernel: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) anyerror!void {
    return switch (api) {
        .opengl => opengl.launchKernel(allocator, kernel, config, args),
        .opengles => opengles.launchKernel(allocator, kernel, config, args),
    };
}

pub fn destroyKernel(api: common.Api, allocator: std.mem.Allocator, kernel: *anyopaque) void {
    switch (api) {
        .opengl => opengl.destroyKernel(allocator, kernel),
        .opengles => opengles.destroyKernel(allocator, kernel),
    }
}

pub fn allocateDeviceMemory(api: common.Api, size: usize) anyerror!*anyopaque {
    return switch (api) {
        .opengl => opengl.allocateDeviceMemory(size),
        .opengles => opengles.allocateDeviceMemory(size),
    };
}

pub fn freeDeviceMemory(api: common.Api, ptr: *anyopaque) void {
    switch (api) {
        .opengl => opengl.freeDeviceMemory(ptr),
        .opengles => opengles.freeDeviceMemory(ptr),
    }
}

pub fn memcpyHostToDevice(api: common.Api, dst: *anyopaque, src: [*]const u8, len: usize) anyerror!void {
    return switch (api) {
        .opengl => opengl.memcpyHostToDevice(dst, @constCast(src), len),
        .opengles => opengles.memcpyHostToDevice(dst, @constCast(src), len),
    };
}

pub fn memcpyDeviceToHost(api: common.Api, dst: [*]u8, src: *anyopaque, len: usize) anyerror!void {
    return switch (api) {
        .opengl => opengl.memcpyDeviceToHost(dst, src, len),
        .opengles => opengles.memcpyDeviceToHost(dst, src, len),
    };
}

pub fn synchronize(api: common.Api) void {
    switch (api) {
        .opengl => opengl.synchronize(),
        .opengles => opengles.synchronize(),
    }
}
