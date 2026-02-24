const std = @import("std");
const backend_mod = @import("backend.zig");

pub const BackendFactory = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const BackendInstance = struct {};

pub const BackendFeature = enum { compute, graphics, transfer, sparse };

pub fn createBackend(_: std.mem.Allocator, _: backend_mod.Backend) !BackendInstance {
    return error.GpuDisabled;
}

pub fn createBestBackend(_: std.mem.Allocator) !BackendInstance {
    return error.GpuDisabled;
}

pub fn destroyBackend(_: *BackendInstance) void {}

test {
    std.testing.refAllDecls(@This());
}
