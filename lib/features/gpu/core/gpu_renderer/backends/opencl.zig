const std = @import("std");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    const ctx = try buffers.GPUContext.initOpenCL(args.allocator);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .mock = ctx.device },
        .queue = .{ .mock = ctx.queue },
    };

    std.log.info("OpenCL backend ready", .{});

    return .{
        .backend = .opencl,
        .gpu_context = ctx,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    _ = config;
    return true;
}
