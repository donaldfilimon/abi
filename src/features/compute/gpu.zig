const std = @import("std");

pub const Backend = enum {
    cuda,
    vulkan,
};

pub const GpuContext = struct {
    allocator: std.mem.Allocator,
    backend: Backend,

    pub fn deinit(self: *GpuContext) void {
        _ = self;
    }
};

pub const GpuBuffer = struct {
    size: usize,
};

pub fn isAvailable(_: Backend) bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator, backend: Backend) !GpuContext {
    _ = allocator;
    _ = backend;
    return error.BackendUnavailable;
}

pub fn allocBuffer(_: *GpuContext, _: usize) !GpuBuffer {
    return error.BackendUnavailable;
}
