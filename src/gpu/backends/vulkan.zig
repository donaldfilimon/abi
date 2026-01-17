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
