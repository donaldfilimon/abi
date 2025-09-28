const std = @import("std");

const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    var ctx = try buffers.GPUContext.initOpenGL(args.allocator);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .mock = ctx.device },
        .queue = .{ .mock = ctx.queue },
    };

    std.log.info("OpenGL backend ready", .{});

    return .{
        .backend = .opengl,
        .gpu_context = ctx,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    return true;
}
