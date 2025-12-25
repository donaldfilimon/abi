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

var initialized = false;

pub fn init() !void {
    if (initialized) return;
    initialized = true;
}

pub fn deinit() void {
    if (!initialized) return;
    initialized = false;
}

pub fn isAvailable() bool {
    return initialized;
}
