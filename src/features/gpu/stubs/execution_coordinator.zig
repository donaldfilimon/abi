const std = @import("std");

pub const ExecutionCoordinator = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const ExecutionMethod = enum { gpu, simd, scalar };

test {
    std.testing.refAllDecls(@This());
}
