const std = @import("std");

pub const KernelDispatcher = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const DispatchError = error{
    GpuDisabled,
    KernelNotFound,
    InvalidArgs,
    LaunchFailed,
};

pub const CompiledKernelHandle = struct {
    id: u64 = 0,
    pub fn isValid(_: CompiledKernelHandle) bool {
        return false;
    }
};

pub const KernelArgs = struct {
    pub fn init() @This() {
        return .{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
