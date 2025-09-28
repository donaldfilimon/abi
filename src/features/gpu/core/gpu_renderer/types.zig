const std = @import("std");

const config = @import("config.zig");
const buffers = @import("buffers.zig");

pub const InitArgs = struct {
    allocator: std.mem.Allocator,
    config: config.GPUConfig,
};

pub const BackendResources = struct {
    backend: config.Backend,
    buffer_manager: buffers.BufferManager,
    gpu_context: ?buffers.GPUContext = null,
    hardware_context: ?buffers.HardwareContext = null,
};
