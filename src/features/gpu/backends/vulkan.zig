//! Vulkan backend implementation facade.
//!
//! Re-exports the decomposed Vulkan implementation from sub-modules.
//!
//! ## Sub-modules
//! - `vulkan/backend_impl.zig` — Global library loading and context initialization
//! - `vulkan/vtable.zig` — VTable implementation (VulkanBackend)
//! - `vulkan/resources.zig` — ShaderCache, CommandPool

const std = @import("std");

// Re-export type definitions for build discovery
pub const vulkan_types = @import("vulkan_types.zig");

// Type aliases
pub const VulkanError = vulkan_types.VulkanError;
pub const VkResult = vulkan_types.VkResult;
pub const VkInstance = vulkan_types.VkInstance;
pub const VkPhysicalDevice = vulkan_types.VkPhysicalDevice;
pub const VkDevice = vulkan_types.VkDevice;
pub const VkQueue = vulkan_types.VkQueue;
pub const VkCommandPool = vulkan_types.VkCommandPool;
pub const VkCommandBuffer = vulkan_types.VkCommandBuffer;
pub const VkBuffer = vulkan_types.VkBuffer;
pub const VkDeviceMemory = vulkan_types.VkDeviceMemory;
pub const VkShaderModule = vulkan_types.VkShaderModule;
pub const VkPipelineLayout = vulkan_types.VkPipelineLayout;
pub const VkPipeline = vulkan_types.VkPipeline;
pub const VkDescriptorSetLayout = vulkan_types.VkDescriptorSetLayout;
pub const VkDescriptorPool = vulkan_types.VkDescriptorPool;
pub const VkDescriptorSet = vulkan_types.VkDescriptorSet;
pub const VkFence = vulkan_types.VkFence;
pub const VkPipelineCache = vulkan_types.VkPipelineCache;

pub const VkDeviceSize = vulkan_types.VkDeviceSize;
pub const VkMemoryPropertyFlags = vulkan_types.VkMemoryPropertyFlags;
pub const VkBufferUsageFlags = vulkan_types.VkBufferUsageFlags;
pub const VkShaderStageFlags = vulkan_types.VkShaderStageFlags;
pub const VkPipelineStageFlags = vulkan_types.VkPipelineStageFlags;

pub const VkPipelineBindPoint = vulkan_types.VkPipelineBindPoint;

pub const VkApplicationInfo = vulkan_types.VkApplicationInfo;
pub const VkInstanceCreateInfo = vulkan_types.VkInstanceCreateInfo;
pub const VkDeviceQueueCreateInfo = vulkan_types.VkDeviceQueueCreateInfo;
pub const VkDeviceCreateInfo = vulkan_types.VkDeviceCreateInfo;
pub const VkBufferCreateInfo = vulkan_types.VkBufferCreateInfo;
pub const VkMemoryAllocateInfo = vulkan_types.VkMemoryAllocateInfo;
pub const VkShaderModuleCreateInfo = vulkan_types.VkShaderModuleCreateInfo;
pub const VkDescriptorSetLayoutBinding = vulkan_types.VkDescriptorSetLayoutBinding;
pub const VkDescriptorSetLayoutCreateInfo = vulkan_types.VkDescriptorSetLayoutCreateInfo;
pub const VkPipelineLayoutCreateInfo = vulkan_types.VkPipelineLayoutCreateInfo;
pub const VkPipelineShaderStageCreateInfo = vulkan_types.VkPipelineShaderStageCreateInfo;
pub const VkComputePipelineCreateInfo = vulkan_types.VkComputePipelineCreateInfo;
pub const VkDescriptorPoolSize = vulkan_types.VkDescriptorPoolSize;
pub const VkDescriptorPoolCreateInfo = vulkan_types.VkDescriptorPoolCreateInfo;
pub const VkDescriptorSetAllocateInfo = vulkan_types.VkDescriptorSetAllocateInfo;
pub const VkDescriptorBufferInfo = vulkan_types.VkDescriptorBufferInfo;
pub const VkWriteDescriptorSet = vulkan_types.VkWriteDescriptorSet;
pub const VkCommandBufferAllocateInfo = vulkan_types.VkCommandBufferAllocateInfo;
pub const VkCommandBufferBeginInfo = vulkan_types.VkCommandBufferBeginInfo;
pub const VkBufferMemoryBarrier = vulkan_types.VkBufferMemoryBarrier;
pub const VkSubmitInfo = vulkan_types.VkSubmitInfo;
pub const VkFenceCreateInfo = vulkan_types.VkFenceCreateInfo;
pub const VkCommandPoolCreateInfo = vulkan_types.VkCommandPoolCreateInfo;
pub const VkPipelineCacheCreateInfo = vulkan_types.VkPipelineCacheCreateInfo;
pub const VkQueueFamilyProperties = vulkan_types.VkQueueFamilyProperties;
pub const VkExtent3D = vulkan_types.VkExtent3D;
pub const VkPushConstantRange = vulkan_types.VkPushConstantRange;
pub const VkLayerProperties = vulkan_types.VkLayerProperties;
pub const VkPhysicalDeviceMemoryProperties = vulkan_types.VkPhysicalDeviceMemoryProperties;
pub const VkMemoryType = vulkan_types.VkMemoryType;
pub const VkMemoryHeap = vulkan_types.VkMemoryHeap;
pub const VkMemoryRequirements = vulkan_types.VkMemoryRequirements;
pub const VkPhysicalDeviceType = vulkan_types.VkPhysicalDeviceType;
pub const VkPhysicalDeviceProperties = vulkan_types.VkPhysicalDeviceProperties;
pub const VkPhysicalDeviceLimits = vulkan_types.VkPhysicalDeviceLimits;
pub const VkPhysicalDeviceSparseProperties = vulkan_types.VkPhysicalDeviceSparseProperties;

pub const VkCreateInstanceFn = vulkan_types.VkCreateInstanceFn;
pub const VkDestroyInstanceFn = vulkan_types.VkDestroyInstanceFn;
pub const VkEnumeratePhysicalDevicesFn = vulkan_types.VkEnumeratePhysicalDevicesFn;
pub const VkGetPhysicalDevicePropertiesFn = vulkan_types.VkGetPhysicalDevicePropertiesFn;
pub const VkGetPhysicalDeviceQueueFamilyPropertiesFn = vulkan_types.VkGetPhysicalDeviceQueueFamilyPropertiesFn;
pub const VkGetPhysicalDeviceMemoryPropertiesFn = vulkan_types.VkGetPhysicalDeviceMemoryPropertiesFn;
pub const VkCreateDeviceFn = vulkan_types.VkCreateDeviceFn;
pub const VkDestroyDeviceFn = vulkan_types.VkDestroyDeviceFn;
pub const VkGetDeviceQueueFn = vulkan_types.VkGetDeviceQueueFn;
pub const VkCreateBufferFn = vulkan_types.VkCreateBufferFn;
pub const VkDestroyBufferFn = vulkan_types.VkDestroyBufferFn;
pub const VkGetBufferMemoryRequirementsFn = vulkan_types.VkGetBufferMemoryRequirementsFn;
pub const VkAllocateMemoryFn = vulkan_types.VkAllocateMemoryFn;
pub const VkFreeMemoryFn = vulkan_types.VkFreeMemoryFn;
pub const VkBindBufferMemoryFn = vulkan_types.VkBindBufferMemoryFn;
pub const VkMapMemoryFn = vulkan_types.VkMapMemoryFn;
pub const VkUnmapMemoryFn = vulkan_types.VkUnmapMemoryFn;
pub const VkCreateShaderModuleFn = vulkan_types.VkCreateShaderModuleFn;
pub const VkDestroyShaderModuleFn = vulkan_types.VkDestroyShaderModuleFn;
pub const VkCreateDescriptorSetLayoutFn = vulkan_types.VkCreateDescriptorSetLayoutFn;
pub const VkDestroyDescriptorSetLayoutFn = vulkan_types.VkDestroyDescriptorSetLayoutFn;
pub const VkCreatePipelineLayoutFn = vulkan_types.VkCreatePipelineLayoutFn;
pub const VkDestroyPipelineLayoutFn = vulkan_types.VkDestroyPipelineLayoutFn;
pub const VkCreateComputePipelinesFn = vulkan_types.VkCreateComputePipelinesFn;
pub const VkDestroyPipelineFn = vulkan_types.VkDestroyPipelineFn;
pub const VkCreateCommandPoolFn = vulkan_types.VkCreateCommandPoolFn;
pub const VkDestroyCommandPoolFn = vulkan_types.VkDestroyCommandPoolFn;
pub const VkAllocateCommandBuffersFn = vulkan_types.VkAllocateCommandBuffersFn;
pub const VkFreeCommandBuffersFn = vulkan_types.VkFreeCommandBuffersFn;
pub const VkBeginCommandBufferFn = vulkan_types.VkBeginCommandBufferFn;
pub const VkEndCommandBufferFn = vulkan_types.VkEndCommandBufferFn;
pub const VkCmdBindPipelineFn = vulkan_types.VkCmdBindPipelineFn;
pub const VkCmdBindDescriptorSetsFn = vulkan_types.VkCmdBindDescriptorSetsFn;
pub const VkCmdDispatchFn = vulkan_types.VkCmdDispatchFn;
pub const VkCreateDescriptorPoolFn = vulkan_types.VkCreateDescriptorPoolFn;
pub const VkDestroyDescriptorPoolFn = vulkan_types.VkDestroyDescriptorPoolFn;
pub const VkAllocateDescriptorSetsFn = vulkan_types.VkAllocateDescriptorSetsFn;
pub const VkFreeDescriptorSetsFn = vulkan_types.VkFreeDescriptorSetsFn;
pub const VkUpdateDescriptorSetsFn = vulkan_types.VkUpdateDescriptorSetsFn;
pub const VkCreateFenceFn = vulkan_types.VkCreateFenceFn;
pub const VkDestroyFenceFn = vulkan_types.VkDestroyFenceFn;
pub const VkResetFencesFn = vulkan_types.VkResetFencesFn;
pub const VkWaitForFencesFn = vulkan_types.VkWaitForFencesFn;
pub const VkQueueSubmitFn = vulkan_types.VkQueueSubmitFn;
pub const VkQueueWaitIdleFn = vulkan_types.VkQueueWaitIdleFn;
pub const VkCmdPushConstantsFn = vulkan_types.VkCmdPushConstantsFn;

pub const VK_QUEUE_GRAPHICS_BIT = vulkan_types.VK_QUEUE_GRAPHICS_BIT;
pub const VK_QUEUE_COMPUTE_BIT = vulkan_types.VK_QUEUE_COMPUTE_BIT;
pub const VK_QUEUE_TRANSFER_BIT = vulkan_types.VK_QUEUE_TRANSFER_BIT;
pub const VK_QUEUE_SPARSE_BINDING_BIT = vulkan_types.VK_QUEUE_SPARSE_BINDING_BIT;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = vulkan_types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = vulkan_types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = vulkan_types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT = vulkan_types.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT = vulkan_types.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT = vulkan_types.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = vulkan_types.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = vulkan_types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
pub const VK_SHADER_STAGE_COMPUTE_BIT = vulkan_types.VK_SHADER_STAGE_COMPUTE_BIT;

pub const VulkanContext = vulkan_types.VulkanContext;
pub const VulkanKernel = vulkan_types.VulkanKernel;
pub const VulkanBuffer = vulkan_types.VulkanBuffer;
pub const KernelSourceFormat = vulkan_types.KernelSourceFormat;
pub const KernelSource = vulkan_types.KernelSource;
pub const KernelConfig = vulkan_types.KernelConfig;

// ============================================================================
// Facade re-exports
// ============================================================================

pub const backend_impl = @import("vulkan/backend_impl.zig");
pub const isVulkanAvailable = backend_impl.isVulkanAvailable;
pub const getDetectedApiVersion = backend_impl.getDetectedApiVersion;
pub const enumerateDevices = backend_impl.enumerateDevices;
pub const initVulkanGlobal = backend_impl.initVulkanGlobal;
pub const deinit = backend_impl.deinit;
pub const findMemoryType = backend_impl.findMemoryType;

// We re-export these from vtable.zig which expects them
pub const VulkanBackend = @import("vulkan/vtable.zig").VulkanBackend;
pub const createVulkanVTable = @import("vulkan/vtable.zig").createVulkanVTable;

// We re-export these from resources.zig
pub const ShaderCache = @import("vulkan/resources.zig").ShaderCache;
pub const CommandPool = @import("vulkan/resources.zig").CommandPool;

// ============================================================================
// Variable proxies (since we cannot export 'pub var' from an inner module)
// ============================================================================
// Note: Some modules directly access `vulkan.vulkan_initialized` or `vulkan.vkCreateBuffer`.
// We need to provide accessor functions or require them to import backend_impl directly.
// For now, we will simply rely on the fact that ABI uses createVulkanVTable()

test {
    _ = @import("vulkan/backend_impl.zig");
    _ = @import("vulkan/vtable.zig");
    _ = @import("vulkan_test.zig");
}