//! Vulkan backend implementation with native GPU execution.
//!
//! Provides Vulkan-specific kernel compilation, execution, and memory management
//! using the Vulkan API for cross-platform compute acceleration.
//!
//! This module is split into submodules for maintainability:
//! - vulkan_types.zig: Type definitions and Vulkan API structures
//! - vulkan_init.zig: Initialization, context management, function loading
//! - vulkan_pipelines.zig: Kernel compilation and execution
//! - vulkan_buffers.zig: Memory allocation and transfers

const std = @import("std");

// Import submodules
pub const vulkan_types = @import("vulkan_types.zig");
pub const vulkan_init = @import("vulkan_init.zig");
pub const vulkan_pipelines = @import("vulkan_pipelines.zig");
pub const vulkan_buffers = @import("vulkan_buffers.zig");

// Re-export types
pub const VulkanError = vulkan_types.VulkanError;
pub const VulkanContext = vulkan_types.VulkanContext;
pub const VulkanKernel = vulkan_types.VulkanKernel;
pub const VulkanBuffer = vulkan_types.VulkanBuffer;

// Re-export initialization functions
pub const init = vulkan_init.init;
pub const deinit = vulkan_init.deinit;

// Re-export pipeline functions
pub const compileKernel = vulkan_pipelines.compileKernel;
pub const launchKernel = vulkan_pipelines.launchKernel;
pub const destroyKernel = vulkan_pipelines.destroyKernel;

// Re-export buffer functions
pub const allocateDeviceMemory = vulkan_buffers.allocateDeviceMemory;
pub const freeDeviceMemory = vulkan_buffers.freeDeviceMemory;
pub const memcpyHostToDevice = vulkan_buffers.memcpyHostToDevice;
pub const memcpyDeviceToHost = vulkan_buffers.memcpyDeviceToHost;

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../device.zig").Device;
const DeviceType = @import("../device.zig").DeviceType;
const Backend = @import("../backend.zig").Backend;

/// Enumerate all Vulkan devices available on the system
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!isAvailable()) {
        return &[_]Device{};
    }

    var devices = std.ArrayList(Device).init(allocator);
    errdefer devices.deinit();

    // Query Vulkan for available devices
    // In a full implementation, we'd use vkEnumeratePhysicalDevices
    // For now, return the initialized device if available
    if (vulkan_init.isInitialized()) {
        try devices.append(.{
            .id = 0,
            .backend = .vulkan,
            .name = "Vulkan Device",
            .device_type = .discrete, // Assume discrete for Vulkan
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{
                .supports_fp16 = true,
                .supports_fp64 = false, // Conservative
                .supports_int8 = true,
                .supports_async_transfers = true,
                .unified_memory = false,
            },
            .compute_units = null,
            .clock_mhz = null,
        });
    }

    return devices.toOwnedSlice();
}

/// Check if Vulkan is available on this system
pub fn isAvailable() bool {
    return vulkan_init.isVulkanAvailable();
}
