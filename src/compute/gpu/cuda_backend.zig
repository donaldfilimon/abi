//! CUDA GPU backend
//!
//! CUDA-specific GPU implementation.

const std = @import("std");

pub const CUDAContext = struct {
    device_id: u32,
    context: *anyopaque,
};

pub const CUDAStream = struct {
    stream: *anyopaque,
};

pub fn init() !void {
    _ = std;
}

pub fn deinit() void {}

pub fn isAvailable() bool {
    return false;
}
