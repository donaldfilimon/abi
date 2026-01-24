//! Vulkan VTable Backend Implementation
//!
//! Provides a complete VTable implementation for Vulkan, enabling real GPU
//! kernel execution through the polymorphic backend interface.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const vulkan = @import("vulkan.zig");

// Re-export VulkanBackend for factory compatibility
pub const VulkanBackend = vulkan.VulkanBackend;

/// Creates a Vulkan backend instance wrapped in the VTable interface.
///
/// Returns BackendError.NotAvailable if Vulkan is disabled at compile time
/// or the Vulkan driver cannot be loaded.
/// Returns BackendError.InitFailed if Vulkan initialization fails.
pub fn createVulkanVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    // Check if Vulkan is enabled at compile time
    if (comptime !build_options.gpu_vulkan) {
        return interface.BackendError.NotAvailable;
    }

    // Use existing implementation from vulkan.zig
    const backend_impl = vulkan.vulkan_vtable;
    return backend_impl.createVulkanVTable(allocator);
}
