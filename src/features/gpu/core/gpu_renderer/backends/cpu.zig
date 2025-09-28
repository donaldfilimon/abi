const std = @import("std");

const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    var ctx = try buffers.GPUContext.init(args.allocator);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .mock = ctx.device },
        .queue = .{ .mock = ctx.queue },
    };

    std.log.info("CPU fallback backend ready", .{});

    return .{
        .backend = .cpu_fallback,
        .gpu_context = ctx,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    return true;
}
