//! Vulkan backend implementation
//!
//! Provides Vulkan-specific kernel compilation and execution.

const std = @import("std");
const kernels = @import("../kernels.zig");

var vulkan_initialized = false;

pub fn init() !void {
    if (vulkan_initialized) return;

    if (!tryLoadVulkan()) {
        return error.VulkanNotAvailable;
    }

    vulkan_initialized = true;
}

pub fn deinit() void {
    vulkan_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: kernels.KernelSource,
) !*anyopaque {
    _ = source;

    const kernel_handle = try allocator.create(VkShaderModule);
    kernel_handle.* = .{ .handle = null };

    return kernel_handle;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: kernels.KernelConfig,
    args: []const ?*const anyopaque,
) !void {
    _ = allocator;
    _ = kernel_handle;
    _ = config;
    _ = args;

    return error.VulkanLaunchFailed;
}

pub fn destroyKernel(kernel_handle: *anyopaque) void {
    const vk_kernel: *VkShaderModule = @ptrCast(@alignCast(kernel_handle));
    std.heap.page_allocator.destroy(vk_kernel);
}

pub fn createCommandBuffer() !*anyopaque {
    const buffer = try std.heap.page_allocator.create(VkCommandBuffer);
    buffer.* = .{ .handle = null };
    return buffer;
}

pub fn destroyCommandBuffer(buffer: *anyopaque) void {
    const vk_buffer: *VkCommandBuffer = @ptrCast(@alignCast(buffer));
    std.heap.page_allocator.destroy(vk_buffer);
}

fn tryLoadVulkan() bool {
    const lib_names = [_][]const u8{ "vulkan-1.dll", "libvulkan.so.1", "libvulkan.dylib" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |_| {
            return true;
        } else |_| {}
    }
    return false;
}

const VkShaderModule = struct {
    handle: ?*anyopaque,
};

const VkCommandBuffer = struct {
    handle: ?*anyopaque,
};
