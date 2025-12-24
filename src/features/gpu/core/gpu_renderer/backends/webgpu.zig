const std = @import("std");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    _ = args;
    if (!config.has_webgpu_support) {
        return config.GpuError.DeviceNotFound;
    }

    const hardware = try buffers.HardwareContext.init(.webgpu);
    const buffer_manager = buffers.BufferManager.fromHardwareContext(hardware);

    std.log.info("WebGPU backend ready", .{});

    return .{
        .backend = .webgpu,
        .hardware_context = hardware,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    return config.has_webgpu_support;
}
