const std = @import("std");
const builtin = @import("builtin");

const config = @import("config.zig");
const buffers = @import("buffers.zig");
const types = @import("types.zig");

pub const InitArgs = types.InitArgs;
pub const BackendResources = types.BackendResources;

pub const webgpu = @import("backends/webgpu.zig");
pub const vulkan = @import("backends/vulkan.zig");
pub const metal = @import("backends/metal.zig");
pub const dx12 = @import("backends/dx12.zig");
pub const opengl = @import("backends/opengl.zig");
pub const opencl = @import("backends/opencl.zig");
pub const cuda = @import("backends/cuda.zig");
pub const cpu = @import("backends/cpu.zig");

pub fn initialize(args: InitArgs, backend: config.Backend) !BackendResources {
    return switch (backend) {
        .webgpu => try webgpu.initialize(args),
        .vulkan => try vulkan.initialize(args),
        .metal => try metal.initialize(args),
        .dx12 => try dx12.initialize(args),
        .opengl => try opengl.initialize(args),
        .opencl => try opencl.initialize(args),
        .cuda => try cuda.initialize(args),
        .cpu_fallback => try cpu.initialize(args),
        .auto => unreachable, // handled by renderer
    };
}

pub fn detectVulkanSupport() bool {
    return vulkan.isSupported();
}

pub fn detectMetalSupport() bool {
    return metal.isSupported();
}

pub fn detectCUDASupport() bool {
    return cuda.isSupported();
}

pub fn detectWebGPUSupport() bool {
    return config.has_webgpu_support;
}

pub fn detectDX12Support() bool {
    return builtin.os.tag == .windows;
}

pub fn detectOpenCLSupport() bool {
    return opencl.isSupported();
}

pub fn detectOpenGLSupport() bool {
    return opengl.isSupported();
}

test "cpu backend initialization" {
    const testing = std.testing;
    const args = InitArgs{ .allocator = testing.allocator, .config = .{} };
    var resources = try cpu.initialize(args);
    defer {
        if (resources.hardware_context) |*ctx| ctx.deinit();
        if (resources.gpu_context) |*ctx| ctx.deinit();
    }

    try testing.expect(resources.backend == .cpu_fallback);
    switch (resources.buffer_manager.queue) {
        .mock => try testing.expect(true),
        .hardware => try testing.expect(false),
    }
}

test "backend initialization gracefully handles missing hardware" {
    const testing = std.testing;
    const args = InitArgs{ .allocator = testing.allocator, .config = .{} };
    const targets = [_]config.Backend{ .webgpu, .vulkan, .metal, .dx12, .opengl, .opencl, .cuda };

    for (targets) |backend| {
        var result = initialize(args, backend) catch |err| {
            try testing.expect(err == config.GpuError.DeviceNotFound or err == config.GpuError.UnsupportedBackend);
            continue;
        };

        defer {
            if (result.hardware_context) |*ctx| ctx.deinit();
            if (result.gpu_context) |*ctx| ctx.deinit();
        }

        try testing.expect(result.backend == backend);
    }
}
