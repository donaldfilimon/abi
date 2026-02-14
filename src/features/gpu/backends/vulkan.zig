//! Vulkan backend implementation.
//!
//! Provides Vulkan-specific kernel compilation, execution, and memory management
//! using the Vulkan API for cross-platform compute acceleration.
//!
//! Type definitions live in `vulkan_types.zig`, tests in `vulkan_test.zig`.
//! This file contains runtime code: library loading, initialization, and the
//! VTable backend implementation.

const std = @import("std");

// Re-export extracted type definitions for build discovery
pub const vulkan_types = @import("vulkan_types.zig");

// ============================================================================
// Errors
// ============================================================================

pub const VulkanError = vulkan_types.VulkanError;

// ============================================================================
// Re-exported Vulkan Types (from vulkan_types.zig)
// ============================================================================

// Core result type
pub const VkResult = vulkan_types.VkResult;

// Handle types
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

// Basic types
pub const VkDeviceSize = vulkan_types.VkDeviceSize;
pub const VkMemoryPropertyFlags = vulkan_types.VkMemoryPropertyFlags;
pub const VkBufferUsageFlags = vulkan_types.VkBufferUsageFlags;
pub const VkShaderStageFlags = vulkan_types.VkShaderStageFlags;
pub const VkPipelineStageFlags = vulkan_types.VkPipelineStageFlags;

// Enums
pub const VkPipelineBindPoint = vulkan_types.VkPipelineBindPoint;

// Create info structures
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

// Function pointer types
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
pub const VkCmdPipelineBarrierFn = vulkan_types.VkCmdPipelineBarrierFn;
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
pub const VkCreatePipelineCacheFn = vulkan_types.VkCreatePipelineCacheFn;
pub const VkDestroyPipelineCacheFn = vulkan_types.VkDestroyPipelineCacheFn;
pub const VkGetPipelineCacheDataFn = vulkan_types.VkGetPipelineCacheDataFn;
pub const VkMergePipelineCachesFn = vulkan_types.VkMergePipelineCachesFn;
pub const VkResetCommandBufferFn = vulkan_types.VkResetCommandBufferFn;
pub const VkResetCommandPoolFn = vulkan_types.VkResetCommandPoolFn;
pub const VkEnumerateInstanceLayerPropertiesFn = vulkan_types.VkEnumerateInstanceLayerPropertiesFn;
pub const VkCmdPushConstantsFn = vulkan_types.VkCmdPushConstantsFn;

// Constants
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

// Context and resource structures
pub const VulkanContext = vulkan_types.VulkanContext;
pub const VulkanKernel = vulkan_types.VulkanKernel;
pub const VulkanBuffer = vulkan_types.VulkanBuffer;
pub const KernelSourceFormat = vulkan_types.KernelSourceFormat;
pub const KernelSource = vulkan_types.KernelSource;
pub const KernelConfig = vulkan_types.KernelConfig;

// ============================================================================
// Library Loading and Initialization
// ============================================================================

pub var vulkan_lib: ?std.DynLib = null;
pub var vulkan_initialized: bool = false;
pub var vulkan_context: ?VulkanContext = null;

// Global function pointers
pub var vkCreateInstance: ?VkCreateInstanceFn = null;
pub var vkDestroyInstance: ?VkDestroyInstanceFn = null;
pub var vkEnumeratePhysicalDevices: ?VkEnumeratePhysicalDevicesFn = null;
pub var vkGetPhysicalDeviceProperties: ?VkGetPhysicalDevicePropertiesFn = null;
pub var vkGetPhysicalDeviceQueueFamilyProperties: ?VkGetPhysicalDeviceQueueFamilyPropertiesFn = null;
pub var vkGetPhysicalDeviceMemoryProperties: ?VkGetPhysicalDeviceMemoryPropertiesFn = null;
pub var vkCreateDevice: ?VkCreateDeviceFn = null;
pub var vkDestroyDevice: ?VkDestroyDeviceFn = null;
pub var vkGetDeviceQueue: ?VkGetDeviceQueueFn = null;
pub var vkEnumerateInstanceLayerProperties: ?VkEnumerateInstanceLayerPropertiesFn = null;

// Instance function pointers
pub var vkCreateBuffer: ?VkCreateBufferFn = null;
pub var vkDestroyBuffer: ?VkDestroyBufferFn = null;
pub var vkGetBufferMemoryRequirements: ?VkGetBufferMemoryRequirementsFn = null;
pub var vkAllocateMemory: ?VkAllocateMemoryFn = null;
pub var vkFreeMemory: ?VkFreeMemoryFn = null;
pub var vkBindBufferMemory: ?VkBindBufferMemoryFn = null;
pub var vkMapMemory: ?VkMapMemoryFn = null;
pub var vkUnmapMemory: ?VkUnmapMemoryFn = null;
pub var vkCreateShaderModule: ?VkCreateShaderModuleFn = null;
pub var vkDestroyShaderModule: ?VkDestroyShaderModuleFn = null;
pub var vkCreateDescriptorSetLayout: ?VkCreateDescriptorSetLayoutFn = null;
pub var vkDestroyDescriptorSetLayout: ?VkDestroyDescriptorSetLayoutFn = null;
pub var vkCreatePipelineLayout: ?VkCreatePipelineLayoutFn = null;
pub var vkDestroyPipelineLayout: ?VkDestroyPipelineLayoutFn = null;
pub var vkCreateComputePipelines: ?VkCreateComputePipelinesFn = null;
pub var vkDestroyPipeline: ?VkDestroyPipelineFn = null;
pub var vkCreateCommandPool: ?VkCreateCommandPoolFn = null;
pub var vkDestroyCommandPool: ?VkDestroyCommandPoolFn = null;
pub var vkAllocateCommandBuffers: ?VkAllocateCommandBuffersFn = null;
pub var vkFreeCommandBuffers: ?VkFreeCommandBuffersFn = null;
pub var vkBeginCommandBuffer: ?VkBeginCommandBufferFn = null;
pub var vkEndCommandBuffer: ?VkEndCommandBufferFn = null;
pub var vkCmdBindPipeline: ?VkCmdBindPipelineFn = null;
pub var vkCmdBindDescriptorSets: ?VkCmdBindDescriptorSetsFn = null;
pub var vkCmdDispatch: ?VkCmdDispatchFn = null;
pub var vkCreateDescriptorPool: ?VkCreateDescriptorPoolFn = null;
pub var vkDestroyDescriptorPool: ?VkDestroyDescriptorPoolFn = null;
pub var vkAllocateDescriptorSets: ?VkAllocateDescriptorSetsFn = null;
pub var vkFreeDescriptorSets: ?VkFreeDescriptorSetsFn = null;
pub var vkUpdateDescriptorSets: ?VkUpdateDescriptorSetsFn = null;
pub var vkCreateFence: ?VkCreateFenceFn = null;
pub var vkDestroyFence: ?VkDestroyFenceFn = null;
pub var vkResetFences: ?VkResetFencesFn = null;
pub var vkWaitForFences: ?VkWaitForFencesFn = null;
pub var vkQueueSubmit: ?VkQueueSubmitFn = null;
pub var vkQueueWaitIdle: ?VkQueueWaitIdleFn = null;
pub var vkCmdPushConstants: ?VkCmdPushConstantsFn = null;

fn tryLoadVulkanLibrary() bool {
    const lib_names = [_][]const u8{
        "vulkan-1.dll",
        "libvulkan.so.1",
        "libvulkan.so",
        "libvulkan.1.dylib",
        "libvulkan.dylib",
    };

    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            vulkan_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadGlobalFunctions() bool {
    const lib = vulkan_lib orelse return false;

    vkCreateInstance = lib.lookup(VkCreateInstanceFn, "vkCreateInstance") orelse return false;
    vkEnumerateInstanceLayerProperties = lib.lookup(VkEnumerateInstanceLayerPropertiesFn, "vkEnumerateInstanceLayerProperties");

    // These need an instance to be fully reliable, but we load them here for enumeration
    vkEnumeratePhysicalDevices = lib.lookup(VkEnumeratePhysicalDevicesFn, "vkEnumeratePhysicalDevices");
    vkGetPhysicalDeviceProperties = lib.lookup(VkGetPhysicalDevicePropertiesFn, "vkGetPhysicalDeviceProperties");
    vkGetPhysicalDeviceQueueFamilyProperties = lib.lookup(VkGetPhysicalDeviceQueueFamilyPropertiesFn, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkGetPhysicalDeviceMemoryProperties = lib.lookup(VkGetPhysicalDeviceMemoryPropertiesFn, "vkGetPhysicalDeviceMemoryProperties");
    vkCreateDevice = lib.lookup(VkCreateDeviceFn, "vkCreateDevice");

    return true;
}

fn loadInstanceFunctions(instance: VkInstance) bool {
    _ = instance;
    const lib = vulkan_lib orelse return false;
    // For simplicity using dlsym, but proper way is vkGetInstanceProcAddr
    // We assume dynamic linking resolution works for these symbols

    vkDestroyInstance = lib.lookup(VkDestroyInstanceFn, "vkDestroyInstance");
    vkDestroyDevice = lib.lookup(VkDestroyDeviceFn, "vkDestroyDevice");
    vkGetDeviceQueue = lib.lookup(VkGetDeviceQueueFn, "vkGetDeviceQueue");

    vkCreateBuffer = lib.lookup(VkCreateBufferFn, "vkCreateBuffer");
    vkDestroyBuffer = lib.lookup(VkDestroyBufferFn, "vkDestroyBuffer");
    vkGetBufferMemoryRequirements = lib.lookup(VkGetBufferMemoryRequirementsFn, "vkGetBufferMemoryRequirements");
    vkAllocateMemory = lib.lookup(VkAllocateMemoryFn, "vkAllocateMemory");
    vkFreeMemory = lib.lookup(VkFreeMemoryFn, "vkFreeMemory");
    vkBindBufferMemory = lib.lookup(VkBindBufferMemoryFn, "vkBindBufferMemory");
    vkMapMemory = lib.lookup(VkMapMemoryFn, "vkMapMemory");
    vkUnmapMemory = lib.lookup(VkUnmapMemoryFn, "vkUnmapMemory");

    vkCreateShaderModule = lib.lookup(VkCreateShaderModuleFn, "vkCreateShaderModule");
    vkDestroyShaderModule = lib.lookup(VkDestroyShaderModuleFn, "vkDestroyShaderModule");
    vkCreateDescriptorSetLayout = lib.lookup(VkCreateDescriptorSetLayoutFn, "vkCreateDescriptorSetLayout");
    vkDestroyDescriptorSetLayout = lib.lookup(VkDestroyDescriptorSetLayoutFn, "vkDestroyDescriptorSetLayout");
    vkCreatePipelineLayout = lib.lookup(VkCreatePipelineLayoutFn, "vkCreatePipelineLayout");
    vkDestroyPipelineLayout = lib.lookup(VkDestroyPipelineLayoutFn, "vkDestroyPipelineLayout");
    vkCreateComputePipelines = lib.lookup(VkCreateComputePipelinesFn, "vkCreateComputePipelines");
    vkDestroyPipeline = lib.lookup(VkDestroyPipelineFn, "vkDestroyPipeline");

    vkCreateCommandPool = lib.lookup(VkCreateCommandPoolFn, "vkCreateCommandPool");
    vkDestroyCommandPool = lib.lookup(VkDestroyCommandPoolFn, "vkDestroyCommandPool");
    vkAllocateCommandBuffers = lib.lookup(VkAllocateCommandBuffersFn, "vkAllocateCommandBuffers");
    vkFreeCommandBuffers = lib.lookup(VkFreeCommandBuffersFn, "vkFreeCommandBuffers");
    vkBeginCommandBuffer = lib.lookup(VkBeginCommandBufferFn, "vkBeginCommandBuffer");
    vkEndCommandBuffer = lib.lookup(VkEndCommandBufferFn, "vkEndCommandBuffer");

    vkCmdBindPipeline = lib.lookup(VkCmdBindPipelineFn, "vkCmdBindPipeline");
    vkCmdBindDescriptorSets = lib.lookup(VkCmdBindDescriptorSetsFn, "vkCmdBindDescriptorSets");
    vkCmdDispatch = lib.lookup(VkCmdDispatchFn, "vkCmdDispatch");
    vkCmdPushConstants = lib.lookup(VkCmdPushConstantsFn, "vkCmdPushConstants");

    vkCreateDescriptorPool = lib.lookup(VkCreateDescriptorPoolFn, "vkCreateDescriptorPool");
    vkDestroyDescriptorPool = lib.lookup(VkDestroyDescriptorPoolFn, "vkDestroyDescriptorPool");
    vkAllocateDescriptorSets = lib.lookup(VkAllocateDescriptorSetsFn, "vkAllocateDescriptorSets");
    vkFreeDescriptorSets = lib.lookup(VkFreeDescriptorSetsFn, "vkFreeDescriptorSets");
    vkUpdateDescriptorSets = lib.lookup(VkUpdateDescriptorSetsFn, "vkUpdateDescriptorSets");

    vkCreateFence = lib.lookup(VkCreateFenceFn, "vkCreateFence");
    vkDestroyFence = lib.lookup(VkDestroyFenceFn, "vkDestroyFence");
    vkResetFences = lib.lookup(VkResetFencesFn, "vkResetFences");
    vkWaitForFences = lib.lookup(VkWaitForFencesFn, "vkWaitForFences");
    vkQueueSubmit = lib.lookup(VkQueueSubmitFn, "vkQueueSubmit");
    vkQueueWaitIdle = lib.lookup(VkQueueWaitIdleFn, "vkQueueWaitIdle");

    return true;
}

pub fn initVulkanGlobal(allocator: std.mem.Allocator) VulkanError!void {
    if (vulkan_initialized) return;

    if (!tryLoadVulkanLibrary()) {
        return VulkanError.InitializationFailed;
    }

    if (!loadGlobalFunctions()) {
        return VulkanError.InitializationFailed;
    }

    // Create Instance
    const app_info = VkApplicationInfo{
        .pApplicationName = "ABI Compute",
        .apiVersion = 0x00400000, // Vulkan 1.0
    };

    const create_info = VkInstanceCreateInfo{
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
    };

    var instance: VkInstance = undefined;
    const res = vkCreateInstance.?(&create_info, null, &instance);
    if (res != .success) return VulkanError.InstanceCreationFailed;

    if (!loadInstanceFunctions(instance)) {
        return VulkanError.InitializationFailed;
    }

    // Select Physical Device
    var device_count: u32 = 0;
    _ = vkEnumeratePhysicalDevices.?(instance, &device_count, null);
    if (device_count == 0) return VulkanError.PhysicalDeviceNotFound;

    const p_devices = try allocator.alloc(VkPhysicalDevice, device_count);
    defer allocator.free(p_devices);
    _ = vkEnumeratePhysicalDevices.?(instance, &device_count, p_devices.ptr);

    const physical_device = p_devices[0]; // Prefer first device

    // Find a queue family with compute support
    var queue_family_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_family_count, null);
    if (queue_family_count == 0) return VulkanError.QueueFamilyNotFound;

    const queue_props_buf = try allocator.alloc(VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_props_buf);
    vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_family_count, queue_props_buf.ptr);

    var queue_family_index: u32 = 0;
    var found_compute = false;
    for (queue_props_buf, 0..) |props, i| {
        if (props.queueFlags & VK_QUEUE_COMPUTE_BIT != 0 and props.queueCount > 0) {
            queue_family_index = @intCast(i);
            found_compute = true;
            break;
        }
    }
    if (!found_compute) return VulkanError.QueueFamilyNotFound;

    // Create Logical Device
    const queue_priority: f32 = 1.0;
    const queue_create_info = VkDeviceQueueCreateInfo{
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const device_create_info = VkDeviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
    };

    var device: VkDevice = undefined;
    if (vkCreateDevice.?(physical_device, &device_create_info, null, &device) != .success) {
        return VulkanError.DeviceCreationFailed;
    }

    // Get Queue
    var queue: VkQueue = undefined;
    vkGetDeviceQueue.?(device, queue_family_index, 0, &queue);

    // Create Command Pool
    const pool_info = VkCommandPoolCreateInfo{
        .queueFamilyIndex = queue_family_index,
        .flags = 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    var command_pool: VkCommandPool = undefined;
    if (vkCreateCommandPool.?(device, &pool_info, null, &command_pool) != .success) {
        return VulkanError.InitializationFailed;
    }

    // Get Memory Properties
    var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
    vkGetPhysicalDeviceMemoryProperties.?(physical_device, @ptrCast(&mem_props));

    vulkan_context = VulkanContext{
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .compute_queue = queue,
        .compute_queue_family_index = queue_family_index,
        .command_pool = command_pool,
        .allocator = allocator,
        .memory_properties = mem_props,
    };

    vulkan_initialized = true;
}

pub fn deinit() void {
    if (!vulkan_initialized) return;
    const ctx = vulkan_context.?;

    vkDestroyCommandPool.?(ctx.device, ctx.command_pool, null);
    vkDestroyDevice.?(ctx.device, null);
    vkDestroyInstance.?(ctx.instance, null);

    vulkan_context = null;
    vulkan_initialized = false;
    if (vulkan_lib) |lib| lib.close();
    vulkan_lib = null;
}

// ============================================================================
// Helpers
// ============================================================================

fn findMemoryType(type_filter: u32, properties: VkMemoryPropertyFlags) VulkanError!u32 {
    const mem_props = vulkan_context.?.memory_properties;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        if ((type_filter & (@as(u32, 1) << @intCast(i))) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return VulkanError.MemoryTypeNotFound;
}

// ============================================================================
// VTable Implementation
// ============================================================================

const interface = @import("../interface.zig");

pub const VulkanBackend = struct {
    allocator: std.mem.Allocator,

    // Track resources
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    const Allocation = struct {
        ptr: *anyopaque,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        size: usize,
    };

    const CompiledKernel = struct {
        handle: *anyopaque, // Points to VulkanKernel
        name: []const u8,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Initialize global Vulkan state if needed
        if (!vulkan_initialized) {
            initVulkanGlobal(allocator) catch |err| {
                return switch (err) {
                    VulkanError.InitializationFailed => interface.BackendError.InitFailed,
                    else => interface.BackendError.NotAvailable,
                };
            };
        }

        const self = allocator.create(Self) catch return interface.BackendError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
            .allocations = .empty,
            .kernels = .empty,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        const ctx = vulkan_context.?;

        // Free allocations
        for (self.allocations.items) |alloc| {
            vkUnmapMemory.?(ctx.device, alloc.memory);
            vkDestroyBuffer.?(ctx.device, alloc.buffer, null);
            vkFreeMemory.?(ctx.device, alloc.memory, null);
        }
        self.allocations.deinit(self.allocator);

        // Destroy kernels
        for (self.kernels.items) |k| {
            const kernel: *VulkanKernel = @ptrCast(@alignCast(k.handle));
            vkDestroyPipeline.?(ctx.device, kernel.pipeline, null);
            vkDestroyPipelineLayout.?(ctx.device, kernel.pipeline_layout, null);
            vkDestroyDescriptorSetLayout.?(ctx.device, kernel.descriptor_set_layout, null);
            vkDestroyDescriptorPool.?(ctx.device, kernel.descriptor_pool, null);
            vkDestroyShaderModule.?(ctx.device, kernel.shader_module, null);
            self.allocator.destroy(kernel);
            self.allocator.free(k.name);
        }
        self.kernels.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(_: *Self) u32 {
        if (!vulkan_initialized) return 0;
        return 1;
    }

    pub fn getDeviceCaps(_: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;
        const ctx = vulkan_context orelse return interface.BackendError.NotAvailable;

        var caps = interface.DeviceCaps{
            .max_threads_per_block = 1024,
            .max_shared_memory = 32768,
            .warp_size = 32,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .unified_memory = false,
        };

        var props: VkPhysicalDeviceProperties = undefined;
        vkGetPhysicalDeviceProperties.?(ctx.physical_device, @ptrCast(&props));

        const name_len = std.mem.indexOfScalar(u8, &props.deviceName, 0) orelse 256;
        @memcpy(caps.name[0..name_len], props.deviceName[0..name_len]);
        caps.name_len = name_len;

        caps.max_threads_per_block = props.limits.maxComputeWorkGroupInvocations;
        caps.max_shared_memory = props.limits.maxComputeSharedMemorySize;

        // Total device memory from memory heaps
        var total_mem: u64 = 0;
        for (0..ctx.memory_properties.memoryHeapCount) |h| {
            total_mem += ctx.memory_properties.memoryHeaps[h].size;
        }
        caps.total_memory = total_mem;

        return caps;
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags;
        const ctx = vulkan_context.?;

        const buffer_info = VkBufferCreateInfo{
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = 0, // Exclusive
        };

        var buffer: VkBuffer = undefined;
        if (vkCreateBuffer.?(ctx.device, &buffer_info, null, &buffer) != .success) {
            return interface.MemoryError.OutOfMemory;
        }

        var mem_reqs: VkMemoryRequirements = undefined;
        vkGetBufferMemoryRequirements.?(ctx.device, buffer, @ptrCast(&mem_reqs));

        const mem_type_index = findMemoryType(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) catch {
            vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        };

        const alloc_info = VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type_index,
        };

        var memory: VkDeviceMemory = undefined;
        if (vkAllocateMemory.?(ctx.device, &alloc_info, null, &memory) != .success) {
            vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        if (vkBindBufferMemory.?(ctx.device, buffer, memory, 0) != .success) {
            vkFreeMemory.?(ctx.device, memory, null);
            vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        var ptr: ?*anyopaque = null;
        if (vkMapMemory.?(ctx.device, memory, 0, size, 0, &ptr) != .success) {
            vkFreeMemory.?(ctx.device, memory, null);
            vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        self.allocations.append(self.allocator, .{
            .ptr = ptr.?,
            .buffer = buffer,
            .memory = memory,
            .size = size,
        }) catch {
            return interface.MemoryError.OutOfMemory;
        };

        return ptr.?;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        const ctx = vulkan_context.?;
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                vkUnmapMemory.?(ctx.device, alloc.memory);
                vkDestroyBuffer.?(ctx.device, alloc.buffer, null);
                vkFreeMemory.?(ctx.device, alloc.memory, null);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(_: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        // Mapped memory, just copy
        const dst_ptr: [*]u8 = @ptrCast(dst);
        @memcpy(dst_ptr[0..src.len], src);
    }

    pub fn copyFromDevice(_: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        // Mapped memory, just copy
        const src_ptr: [*]const u8 = @ptrCast(src);
        @memcpy(dst, src_ptr[0..dst.len]);
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        // In Vulkan, we'd use a transfer command buffer on the queue.
        // For mapped memory, it's just a memcpy anyway.
        return Self.copyToDevice(self, dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return Self.copyFromDevice(self, dst, src);
    }

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, kernel_name: []const u8) interface.KernelError!*anyopaque {
        const ctx = vulkan_context.?;

        // 1. Create Shader Module (assumes SPIR-V source)
        // Check 4-byte alignment
        if (source.len % 4 != 0) return interface.KernelError.CompileFailed;

        const create_info = VkShaderModuleCreateInfo{
            .codeSize = source.len,
            .pCode = @ptrCast(@alignCast(source.ptr)),
        };

        var shader_module: VkShaderModule = undefined;
        if (vkCreateShaderModule.?(ctx.device, &create_info, null, &shader_module) != .success) {
            return interface.KernelError.CompileFailed;
        }

        // 2. Create Descriptor Set Layout
        // Simple bindless-like layout: binding 0..N are storage buffers
        // We'll support up to 8 storage buffers for now
        var bindings: [8]VkDescriptorSetLayoutBinding = undefined;
        for (0..8) |i| {
            bindings[i] = VkDescriptorSetLayoutBinding{
                .binding = @intCast(i),
                .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }

        const descriptor_layout_info = VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 8,
            .pBindings = &bindings,
        };

        var descriptor_set_layout: VkDescriptorSetLayout = undefined;
        if (vkCreateDescriptorSetLayout.?(ctx.device, &descriptor_layout_info, null, &descriptor_set_layout) != .success) {
            return interface.KernelError.CompileFailed;
        }

        // 3. Create Pipeline Layout
        const pipeline_layout_info = VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = @ptrCast(&descriptor_set_layout),
        };

        var pipeline_layout: VkPipelineLayout = undefined;
        if (vkCreatePipelineLayout.?(ctx.device, &pipeline_layout_info, null, &pipeline_layout) != .success) {
            vkDestroyDescriptorSetLayout.?(ctx.device, descriptor_set_layout, null);
            return interface.KernelError.CompileFailed;
        }

        // 4. Create Compute Pipeline
        const shader_stage_info = VkPipelineShaderStageCreateInfo{
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
        };

        const pipeline_info = VkComputePipelineCreateInfo{
            .stage = shader_stage_info,
            .layout = pipeline_layout,
        };

        var pipeline: VkPipeline = undefined;
        if (vkCreateComputePipelines.?(ctx.device, null, 1, @ptrCast(&pipeline_info), null, @ptrCast(&pipeline)) != .success) {
            vkDestroyPipelineLayout.?(ctx.device, pipeline_layout, null);
            vkDestroyDescriptorSetLayout.?(ctx.device, descriptor_set_layout, null);
            return interface.KernelError.CompileFailed;
        }

        // 5. Create Descriptor Pool
        const pool_size = VkDescriptorPoolSize{
            .type = 7, // STORAGE_BUFFER
            .descriptorCount = 8,
        };
        const pool_info = VkDescriptorPoolCreateInfo{
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = @ptrCast(&pool_size),
        };
        var descriptor_pool: VkDescriptorPool = undefined;
        if (vkCreateDescriptorPool.?(ctx.device, &pool_info, null, &descriptor_pool) != .success) {
            return interface.KernelError.CompileFailed;
        }

        const vulkan_kernel = allocator.create(VulkanKernel) catch return interface.KernelError.CompileFailed;
        vulkan_kernel.* = .{
            .shader_module = shader_module,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipeline,
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_pool = descriptor_pool,
        };

        // Track
        const name_copy = self.allocator.dupe(u8, kernel_name) catch return interface.KernelError.CompileFailed;
        self.kernels.append(self.allocator, .{ .handle = vulkan_kernel, .name = name_copy }) catch return interface.KernelError.CompileFailed;

        return vulkan_kernel;
    }

    pub fn launchKernel(self: *Self, kernel_handle: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        const ctx = vulkan_context.?;
        const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

        // 1. Allocate Descriptor Set
        const alloc_info = VkDescriptorSetAllocateInfo{
            .descriptorPool = kernel.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = @ptrCast(&kernel.descriptor_set_layout),
        };
        var descriptor_set: VkDescriptorSet = undefined;
        if (vkAllocateDescriptorSets.?(ctx.device, &alloc_info, @ptrCast(&descriptor_set)) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        // 2. Update Descriptor Set
        var writes: [8]VkWriteDescriptorSet = undefined;
        var buffer_infos: [8]VkDescriptorBufferInfo = undefined;

        for (args, 0..) |arg, i| {
            if (i >= 8) break;

            // Find buffer from pointer
            var buffer_handle: VkBuffer = undefined;
            var found = false;
            for (self.allocations.items) |alloc| {
                if (alloc.ptr == arg) {
                    buffer_handle = alloc.buffer;
                    found = true;
                    break;
                }
            }
            if (!found) continue;

            buffer_infos[i] = VkDescriptorBufferInfo{
                .buffer = buffer_handle,
                .offset = 0,
                .range = 0xFFFFFFFFFFFFFFFF, // VK_WHOLE_SIZE
            };

            writes[i] = VkWriteDescriptorSet{
                .dstSet = descriptor_set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = 7, // STORAGE_BUFFER
                .pBufferInfo = &buffer_infos[i],
            };
        }

        vkUpdateDescriptorSets.?(ctx.device, @intCast(@min(args.len, 8)), &writes, 0, null);

        // 3. Record Command Buffer
        const alloc_cmd_info = VkCommandBufferAllocateInfo{
            .commandPool = ctx.command_pool,
            .level = 0, // PRIMARY
            .commandBufferCount = 1,
        };
        var cmd_buffer: VkCommandBuffer = undefined;
        if (vkAllocateCommandBuffers.?(ctx.device, &alloc_cmd_info, @ptrCast(&cmd_buffer)) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        const begin_info = VkCommandBufferBeginInfo{
            .flags = 0x00000004, // ONE_TIME_SUBMIT
        };
        _ = vkBeginCommandBuffer.?(cmd_buffer, &begin_info);

        vkCmdBindPipeline.?(cmd_buffer, .compute, kernel.pipeline);
        vkCmdBindDescriptorSets.?(cmd_buffer, .compute, kernel.pipeline_layout, 0, 1, @ptrCast(&descriptor_set), 0, null);
        vkCmdDispatch.?(cmd_buffer, config.grid_x, config.grid_y, config.grid_z);

        _ = vkEndCommandBuffer.?(cmd_buffer);

        // 4. Submit
        const submit_info = VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&cmd_buffer),
        };
        if (vkQueueSubmit.?(ctx.compute_queue, 1, @ptrCast(&submit_info), null) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        // Wait for idle (simple sync)
        _ = vkQueueWaitIdle.?(ctx.compute_queue);

        // Cleanup command buffer
        vkFreeCommandBuffers.?(ctx.device, ctx.command_pool, 1, @ptrCast(&cmd_buffer));

        // Reset descriptor pool for next use (simplification)
        // In real usage we'd have a better pool strategy
        // vkResetDescriptorPool... but we only have 1 set in pool, so freeing sets works too
        // Actually since we created pool with maxSets=1, we must free the set
        _ = vkFreeDescriptorSets.?(ctx.device, kernel.descriptor_pool, 1, @ptrCast(&descriptor_set));
    }

    pub fn destroyKernel(self: *Self, kernel_handle: *anyopaque) void {
        const ctx = vulkan_context.?;
        const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

        // Remove from tracking if present
        for (self.kernels.items, 0..) |k, i| {
            if (k.handle == kernel_handle) {
                _ = self.kernels.swapRemove(i);
                self.allocator.free(k.name);
                break;
            }
        }

        // Destroy Vulkan resources (shader module can be destroyed after pipeline creation)
        if (vkDestroyPipeline) |destroy_pipeline| {
            destroy_pipeline(ctx.device, kernel.pipeline, null);
        }
        if (vkDestroyPipelineLayout) |destroy_pipeline_layout| {
            destroy_pipeline_layout(ctx.device, kernel.pipeline_layout, null);
        }
        if (vkDestroyDescriptorSetLayout) |destroy_descriptor_set_layout| {
            destroy_descriptor_set_layout(ctx.device, kernel.descriptor_set_layout, null);
        }
        if (vkDestroyDescriptorPool) |destroy_descriptor_pool| {
            destroy_descriptor_pool(ctx.device, kernel.descriptor_pool, null);
        }
        if (vkDestroyShaderModule) |destroy_shader_module| {
            destroy_shader_module(ctx.device, kernel.shader_module, null);
        }

        self.allocator.destroy(kernel);
    }

    pub fn synchronize(_: *Self) interface.BackendError!void {
        const ctx = vulkan_context.?;
        _ = vkQueueWaitIdle.?(ctx.compute_queue);
    }
};

// ============================================================================
// Exports for Factory
// ============================================================================

pub const vulkan_vtable = struct {
    pub fn createVulkanVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
        const impl = VulkanBackend.init(allocator) catch return interface.BackendError.InitFailed;
        return interface.createBackend(VulkanBackend, impl);
    }
};

pub fn isVulkanAvailable() bool {
    // Try to load library if not already loaded
    if (vulkan_lib == null) {
        _ = tryLoadVulkanLibrary();
    }
    return vulkan_lib != null;
}

// ============================================================================
// Shader Cache (stub)
// ============================================================================

/// Shader cache stub for future caching implementation.
/// Currently a placeholder to satisfy imports.
pub const ShaderCache = struct {};

// ============================================================================
// Command Pool (stub)
// ============================================================================

/// Command pool stub for future pooling implementation.
/// Currently a placeholder to satisfy imports.
pub const CommandPool = struct {};

// ============================================================================
// Top-Level VTable Factory Export
// ============================================================================

/// Creates a Vulkan backend instance wrapped in the VTable interface.
///
/// This is the main entry point for creating a Vulkan backend. It wraps
/// the internal vulkan_vtable implementation for external consumers.
///
/// Returns BackendError.NotAvailable if Vulkan driver cannot be loaded.
/// Returns BackendError.InitFailed if Vulkan initialization fails.
pub const createVulkanVTable = vulkan_vtable.createVulkanVTable;

// ============================================================================
// Tests
// ============================================================================

test {
    _ = @import("vulkan_types.zig");
    _ = @import("vulkan_test.zig");
}
