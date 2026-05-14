//! TPU Operators
//!
//! Basic TPU tensor operations.
//! These are stub implementations that return BackendNotSupported.

const std = @import("std");

pub const TpuMatMul = struct {
    handle: *anyopaque,
};

pub const TpuConvolution = struct {
    handle: *anyopaque,
};

pub const TpuActivation = struct {
    handle: *anyopaque,
};

pub fn createMatMul(allocator: std.mem.Allocator) !TpuMatMul {
    _ = allocator;
    return error{BackendNotSupported};
}

pub fn createConvolution(allocator: std.mem.Allocator) !TpuConvolution {
    _ = allocator;
    return error{BackendNotSupported};
}

pub fn createActivation(allocator: std.mem.Allocator, activation_type: []const u8) !TpuActivation {
    _ = allocator;
    _ = activation_type;
    return error{BackendNotSupported};
}
