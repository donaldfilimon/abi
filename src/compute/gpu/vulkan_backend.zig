//! Vulkan GPU backend
//!
//! Vulkan-specific GPU implementation.

const std = @import("std");

pub const VulkanContext = struct {
    instance: *anyopaque,
    device: *anyopaque,
};

pub const VulkanQueue = struct {
    queue: *anyopaque,
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
