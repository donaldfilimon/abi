const std = @import("std");
const builtin = @import("builtin");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    _ = args;
    if (!isSupported()) {
        return config.GpuError.DeviceNotFound;
    }

    var hardware = try buffers.HardwareContext.init(.metal);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .hardware = hardware.device },
        .queue = .{ .hardware = hardware.queue },
    };

    std.log.info("Metal backend ready", .{});

    return .{
        .backend = .metal,
        .hardware_context = hardware,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    return switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos => true,
        else => false,
    };
}
