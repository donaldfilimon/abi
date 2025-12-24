const std = @import("std");
const builtin = @import("builtin");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

const driver_probe = @import("driver_probe.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    _ = args;
    if (!isSupported()) {
        return config.GpuError.DeviceNotFound;
    }

    const hardware = try buffers.HardwareContext.init(.vulkan);
    const buffer_manager = buffers.BufferManager.fromHardwareContext(hardware);

    std.log.info("Vulkan backend ready", .{});

    return .{
        .backend = .vulkan,
        .hardware_context = hardware,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    const candidates = switch (builtin.os.tag) {
        .windows => &[_][]const u8{"vulkan-1.dll"},
        .linux => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
        else => &[_][]const u8{},
    };
    return driver_probe.tryOpenDriver(candidates);
}
