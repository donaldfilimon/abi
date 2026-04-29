const std = @import("std");
const builtin = @import("builtin");
const vulkan_caps = @import("capabilities.zig");

// Re-export extracted type definitions for build discovery
pub const vulkan_types = @import("../vulkan_types.zig");

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
pub var vulkan_initialized = std.atomic.Value(bool).init(false);
pub var vulkan_context: ?VulkanContext = null;
pub var detected_api_version_raw: u32 = vulkan_caps.encodeApiVersion(.{
    .major = 1,
    .minor = 0,
    .patch = 0,
});

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
pub var vkEnumerateInstanceLayerProperties: ?vulkan_types.VkEnumerateInstanceLayerPropertiesFn = null;

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
pub var vkCmdPushConstants: ?VkCmdPushConstantsFn = null;

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
    if (vulkan_lib == null) return false;
    var lib = vulkan_lib.?;

    if (vulkan_caps.queryLoaderApiVersion(&lib)) |api_version| {
        detected_api_version_raw = api_version;
    } else {
        detected_api_version_raw = vulkan_caps.encodeApiVersion(.{
            .major = 1,
            .minor = 0,
            .patch = 0,
        });
    }

    vkCreateInstance = lib.lookup(VkCreateInstanceFn, "vkCreateInstance") orelse return false;
    vkEnumerateInstanceLayerProperties = lib.lookup(vulkan_types.VkEnumerateInstanceLayerPropertiesFn, "vkEnumerateInstanceLayerProperties");
    vkEnumeratePhysicalDevices = lib.lookup(VkEnumeratePhysicalDevicesFn, "vkEnumeratePhysicalDevices");
    vkGetPhysicalDeviceProperties = lib.lookup(VkGetPhysicalDevicePropertiesFn, "vkGetPhysicalDeviceProperties");
    vkGetPhysicalDeviceQueueFamilyProperties = lib.lookup(VkGetPhysicalDeviceQueueFamilyPropertiesFn, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkGetPhysicalDeviceMemoryProperties = lib.lookup(VkGetPhysicalDeviceMemoryPropertiesFn, "vkGetPhysicalDeviceMemoryProperties");
    vkCreateDevice = lib.lookup(VkCreateDeviceFn, "vkCreateDevice");

    return true;
}

fn loadInstanceFunctions(instance: VkInstance) bool {
    _ = instance;
    if (vulkan_lib == null) return false;
    var lib = vulkan_lib.?;

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
    if (vulkan_initialized.load(.acquire)) return;

    if (!tryLoadVulkanLibrary()) {
        return VulkanError.InitializationFailed;
    }

    if (!loadGlobalFunctions()) {
        return VulkanError.InitializationFailed;
    }

    if (!vulkan_caps.meetsTargetMinimum(builtin.target.os.tag, detected_api_version_raw)) {
        const detected = vulkan_caps.decodeApiVersion(detected_api_version_raw);
        const required = vulkan_caps.minimumVersionForTarget(builtin.target.os.tag);
        std.log.warn(
            "Vulkan backend requires >= {}.{} for this target (detected {}.{})",
            .{ required.major, required.minor, detected.major, detected.minor },
        );
        return VulkanError.VersionNotSupported;
    }

    // Create Instance
    const app_info = VkApplicationInfo{
        .pApplicationName = "ABI Compute",
        .apiVersion = detected_api_version_raw,
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

    const p_devices = allocator.alloc(VkPhysicalDevice, device_count) catch return error.MemoryAllocationFailed;
    defer allocator.free(p_devices);
    _ = vkEnumeratePhysicalDevices.?(instance, &device_count, p_devices.ptr);

    const physical_device = p_devices[0];

    // Find a queue family with compute support
    var queue_family_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_family_count, null);
    if (queue_family_count == 0) return VulkanError.QueueFamilyNotFound;

    const queue_props_buf = allocator.alloc(VkQueueFamilyProperties, queue_family_count) catch return error.MemoryAllocationFailed;
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
        .pQueuePriorities = @as([*]const f32, @ptrCast(&queue_priority)),
    };

    const device_create_info = VkDeviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = @as(?[*]const VkDeviceQueueCreateInfo, @ptrCast(&queue_create_info)),
    };

    var device: VkDevice = undefined;
    if (vkCreateDevice.?(physical_device, &device_create_info, null, &device) != .success) {
        return VulkanError.DeviceCreationFailed;
    }

    var queue: VkQueue = undefined;
    vkGetDeviceQueue.?(device, queue_family_index, 0, &queue);

    const pool_info = VkCommandPoolCreateInfo{
        .queueFamilyIndex = queue_family_index,
        .flags = 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    var command_pool: VkCommandPool = undefined;
    if (vkCreateCommandPool.?(device, &pool_info, null, &command_pool) != .success) {
        return VulkanError.InitializationFailed;
    }

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

    vulkan_initialized.store(true, .release);
}

pub fn deinit() void {
    if (!vulkan_initialized.load(.acquire)) return;
    const ctx = vulkan_context.?;

    vkDestroyCommandPool.?(ctx.device, ctx.command_pool, null);
    vkDestroyDevice.?(ctx.device, null);
    vkDestroyInstance.?(ctx.instance, null);

    vulkan_context = null;
    vulkan_initialized.store(false, .release);
    detected_api_version_raw = vulkan_caps.encodeApiVersion(.{
        .major = 1,
        .minor = 0,
        .patch = 0,
    });
    if (vulkan_lib != null) {
        var lib = vulkan_lib.?;
        lib.close();
    }
    vulkan_lib = null;
}

// ============================================================================
// Helpers
// ============================================================================

pub fn findMemoryType(type_filter: u32, properties: VkMemoryPropertyFlags) VulkanError!u32 {
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
// Re-exports from sub-modules
// ============================================================================

const interface = @import("../../interface.zig");

pub fn isVulkanAvailable() bool {
    if (vulkan_lib == null) {
        _ = tryLoadVulkanLibrary();
    }
    if (vulkan_lib == null) return false;

    var lib = vulkan_lib.?;
    if (vulkan_caps.queryLoaderApiVersion(&lib)) |api_version| {
        return vulkan_caps.meetsTargetMinimum(builtin.target.os.tag, api_version);
    }
    const fallback = vulkan_caps.encodeApiVersion(.{ .major = 1, .minor = 0, .patch = 0 });
    return vulkan_caps.meetsTargetMinimum(builtin.target.os.tag, fallback);
}

pub fn getDetectedApiVersion() vulkan_caps.VulkanVersion {
    return vulkan_caps.decodeApiVersion(detected_api_version_raw);
}

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../../device.zig").Device;

pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!tryLoadVulkanLibrary()) {
        return &[_]Device{};
    }

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    if (vulkan_context) |_| {
        const name = try allocator.dupe(u8, "Vulkan Device");
        errdefer allocator.free(name);

        try devices.append(allocator, .{
            .id = 0,
            .backend = .vulkan,
            .name = name,
            .device_type = .discrete,
            .vendor = .unknown,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{
                .supports_fp16 = true,
                .supports_int8 = true,
                .supports_async_transfers = true,
                .unified_memory = false,
            },
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        });
    }

    return devices.toOwnedSlice(allocator);
}

test {
    _ = @import("../vulkan_types.zig");
    _ = @import("capabilities.zig");
    _ = @import("../vulkan_test.zig");
}
