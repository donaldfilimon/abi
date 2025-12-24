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

pub fn init() !void {
    _ = std;
}

pub fn deinit() void {}

pub fn isAvailable() bool {
    return false;
}
