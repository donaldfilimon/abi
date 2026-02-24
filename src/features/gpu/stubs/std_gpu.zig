const std = @import("std");

// std.gpu types — stubs for Zig 0.16 native GPU address spaces and shader built-ins

pub fn GlobalPtr(comptime T: type) type {
    return *T;
}

pub fn SharedPtr(comptime T: type) type {
    return *T;
}

pub fn StoragePtr(comptime T: type) type {
    return *T;
}

pub fn UniformPtr(comptime T: type) type {
    return *const T;
}

pub fn ConstantPtr(comptime T: type) type {
    return *const T;
}

pub fn globalInvocationId() [3]u32 {
    return .{ 0, 0, 0 };
}
pub fn workgroupId() [3]u32 {
    return .{ 0, 0, 0 };
}
pub fn localInvocationId() [3]u32 {
    return .{ 0, 0, 0 };
}
pub fn workgroupBarrier() void {}
pub fn setLocalSize(_: [3]u32) void {}

test {
    std.testing.refAllDecls(@This());
}
