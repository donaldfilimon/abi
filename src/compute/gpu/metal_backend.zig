//! Metal GPU backend
//!
//! Metal-specific GPU implementation for Apple platforms.

const std = @import("std");

pub const MetalContext = struct {
    device: *anyopaque,
    command_queue: *anyopaque,
};

pub const MetalCommandBuffer = struct {
    buffer: *anyopaque,
};

pub fn init() !void {
    _ = std;
}

pub fn deinit() void {}

pub fn isAvailable() bool {
    return false;
}
