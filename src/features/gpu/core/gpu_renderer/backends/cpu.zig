const std = @import("std");

const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    const ctx = try buffers.GPUContext.init(args.allocator);
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

test "cpu backend returns fallback resources" {
    const testing = std.testing;
    var resources = try initialize(.{ .allocator = testing.allocator, .config = .{} });
    defer {
        if (resources.gpu_context) |*ctx| ctx.deinit();
        if (resources.hardware_context) |*ctx| ctx.deinit();
    }

    try testing.expect(resources.backend == .cpu_fallback);
    try testing.expect(resources.gpu_context != null);
    try testing.expect(resources.hardware_context == null);
    try testing.expect(switch (resources.buffer_manager.device) {
        .mock => true,
        else => false,
    });
}
