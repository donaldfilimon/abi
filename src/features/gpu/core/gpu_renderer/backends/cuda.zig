const std = @import("std");
const builtin = @import("builtin");

const config = @import("../config.zig");
const buffers = @import("../buffers.zig");
const types = @import("../types.zig");

const DynLib = std.DynLib;

pub fn initialize(args: types.InitArgs) !types.BackendResources {
    if (!isSupported()) {
        return config.GpuError.DeviceNotFound;
    }

    const ctx = try buffers.GPUContext.initCUDA(args.allocator);
    const buffer_manager = buffers.BufferManager{
        .device = .{ .mock = ctx.device },
        .queue = .{ .mock = ctx.queue },
    };

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
    return tryOpenDriver(candidates);
}

fn tryOpenDriver(names: []const []const u8) bool {
    for (names) |name| {
        var lib = DynLib.openZ(name) catch continue;
        defer lib.close();
        return true;
    }
    return false;
}
