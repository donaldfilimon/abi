const std = @import("std");
const builtin = @import("builtin");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

const driver_probe = @import("driver_probe.zig");

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    if (!isSupported()) {
        return config.GpuError.DeviceNotFound;
    }

    const ctx = try buffers.GPUContext.initCUDA(args.allocator);
    const buffer_manager = buffers.BufferManager.fromMockContext(ctx);

    std.log.info("CUDA backend ready", .{});

    return .{
        .backend = .cuda,
        .gpu_context = ctx,
        .buffer_manager = buffer_manager,
    };
}

pub fn isSupported() bool {
    const candidates = switch (builtin.os.tag) {
        .windows => &[_][]const u8{"nvcuda.dll"},
        .linux => &[_][]const u8{ "libcuda.so", "libcuda.so.1" },
        else => &[_][]const u8{},
    };
    return driver_probe.tryOpenDriver(candidates);
}
