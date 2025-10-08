const std = @import("std");
const builtin = @import("builtin");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    if (!isSupported()) {
        return config.GpuError.DeviceNotFound;
    }

    const ctx = try buffers.GPUContext.initDX12(args.allocator);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .mock = ctx.device },
        .queue = .{ .mock = ctx.queue },
    };

    std.log.info("DirectX 12 backend ready", .{});

    return .{
        .backend = .dx12,
        .gpu_context = ctx,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    return builtin.os.tag == .windows;
}
