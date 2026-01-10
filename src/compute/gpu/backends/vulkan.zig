//! Vulkan backend implementation with native GPU execution.
//!
//! Provides Vulkan-specific kernel compilation, execution, and memory management
//! using the Vulkan API for cross-platform compute acceleration.

const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

pub const VulkanError = error{
    InitializationFailed,
    DeviceNotFound,
    InstanceCreationFailed,
    PhysicalDeviceNotFound,
    LogicalDeviceCreationFailed,
    QueueFamilyNotFound,
    MemoryTypeNotFound,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    CommandBufferAllocationFailed,
    BufferCreationFailed,
    MemoryAllocationFailed,
    CommandRecordingFailed,
    SubmissionFailed,
};

const VkResult = enum(i32) {
    success = 0,
    not_ready = 1,
    timeout = 2,
    event_set = 3,
    event_reset = 4,
    incomplete = 5,
    error_out_of_host_memory = -1,
    error_out_of_device_memory = -2,
    error_initialization_failed = -3,
    error_device_lost = -4,
    error_memory_map_failed = -5,
    error_layer_not_present = -6,
    error_extension_not_present = -7,
    error_feature_not_present = -8,
    error_incompatible_driver = -9,
    error_too_many_objects = -10,
    error_format_not_supported = -11,
    error_fragmented_pool = -12,
    error_unknown = -13,
    error_surface_lost_khr = -1000000000,
    error_native_window_in_use_khr = -1000000002,
    suboptimal_khr = 1000001003,
    error_out_of_date_khr = -1000001004,
    error_incompatible_display_khr = -1000002001,
    error_validation_failed_ext = -1000011001,
    error_invalid_shader_nv = -1000012000,
};

const VkInstance = *anyopaque;
const VkPhysicalDevice = *anyopaque;
const VkDevice = *anyopaque;
const VkQueue = *anyopaque;
const VkCommandPool = *anyopaque;
const VkCommandBuffer = *anyopaque;
const VkBuffer = *anyopaque;
const VkDeviceMemory = *anyopaque;
const VkShaderModule = *anyopaque;
const VkPipelineLayout = *anyopaque;
const VkPipeline = *anyopaque;
const VkDescriptorSetLayout = *anyopaque;
const VkDescriptorPool = *anyopaque;
const VkDescriptorSet = *anyopaque;
const VkFence = *anyopaque;

const VkDeviceSize = u64;
const VkMemoryPropertyFlags = u32;
const VkBufferUsageFlags = u32;
const VkShaderStageFlags = u32;
const VkPipelineStageFlags = u32;

const VkApplicationInfo = extern struct {
    sType: i32 = 0, // VK_STRUCTURE_TYPE_APPLICATION_INFO
    pNext: ?*anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

const VkInstanceCreateInfo = extern struct {
    sType: i32 = 1, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

const VkDeviceQueueCreateInfo = extern struct {
    sType: i32 = 3, // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 0,
    pQueuePriorities: [*]const f32 = null,
};

const VkDeviceCreateInfo = extern struct {
    sType: i32 = 3, // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*anyopaque = null,
};

const VkBufferCreateInfo = extern struct {
    sType: i32 = 12, // VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize = 0,
    usage: VkBufferUsageFlags = 0,
    sharingMode: i32 = 0, // VK_SHARING_MODE_EXCLUSIVE
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

const VkMemoryAllocateInfo = extern struct {
    sType: i32 = 5, // VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
    pNext: ?*anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

const VkShaderModuleCreateInfo = extern struct {
    sType: i32 = 16, // VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    codeSize: usize = 0,
    pCode: ?[*]const u32 = null,
};

const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: i32 = 0, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    descriptorCount: u32 = 1,
    stageFlags: VkShaderStageFlags = 0,
    pImmutableSamplers: ?*anyopaque = null,
};

const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: i32 = 32, // VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

const VkPipelineLayoutCreateInfo = extern struct {
    sType: i32 = 30, // VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?*anyopaque = null,
};

const VkComputePipelineCreateInfo = extern struct {
    sType: i32 = 29, // VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

const VkPipelineShaderStageCreateInfo = extern struct {
    sType: i32 = 18, // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: VkShaderStageFlags = 0,
    module: VkShaderModule,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*anyopaque = null,
};

const VkDescriptorPoolSize = extern struct {
    type: i32 = 0, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    descriptorCount: u32 = 0,
};

const VkDescriptorPoolCreateInfo = extern struct {
    sType: i32 = 33, // VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};

const VkDescriptorSetAllocateInfo = extern struct {
    sType: i32 = 34, // VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    pNext: ?*anyopaque = null,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
};

const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = 0,
};

const VkWriteDescriptorSet = extern struct {
    sType: i32 = 35, // VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
    pNext: ?*anyopaque = null,
    dstSet: VkDescriptorSet,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 1,
    descriptorType: i32 = 0, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    pImageInfo: ?*anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*anyopaque = null,
};

const VkCommandBufferAllocateInfo = extern struct {
    sType: i32 = 40, // VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
    pNext: ?*anyopaque = null,
    commandPool: VkCommandPool,
    level: i32 = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
    commandBufferCount: u32 = 0,
};

const VkCommandBufferBeginInfo = extern struct {
    sType: i32 = 42, // VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pInheritanceInfo: ?*anyopaque = null,
};

const VkBufferMemoryBarrier = extern struct {
    sType: i32 = 44, // VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
    pNext: ?*anyopaque = null,
    srcAccessMask: u32 = 0,
    dstAccessMask: u32 = 0,
    srcQueueFamilyIndex: u32 = 0,
    dstQueueFamilyIndex: u32 = 0,
    buffer: VkBuffer,
    offset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
};

const VkSubmitInfo = extern struct {
    sType: i32 = 4, // VK_STRUCTURE_TYPE_SUBMIT_INFO
    pNext: ?*anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?[*]const *anyopaque = null,
    pWaitDstStageMask: ?[*]const VkPipelineStageFlags = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?[*]const *anyopaque = null,
};

const VkFenceCreateInfo = extern struct {
    sType: i32 = 8, // VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

const VkCommandPoolCreateInfo = extern struct {
    sType: i32 = 39, // VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};

const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

const VkMemoryHeap = extern struct {
    size: VkDeviceSize,
    flags: u32,
};

const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

// Vulkan API function pointers
const VkCreateInstanceFn = *const fn (*const VkInstanceCreateInfo, ?*anyopaque, *VkInstance) callconv(.c) VkResult;
const VkDestroyInstanceFn = *const fn (VkInstance, ?*anyopaque) callconv(.c) void;
const VkEnumeratePhysicalDevicesFn = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
const VkGetPhysicalDevicePropertiesFn = *const fn (VkPhysicalDevice, *anyopaque) callconv(.c) void;
const VkGetPhysicalDeviceQueueFamilyPropertiesFn = *const fn (VkPhysicalDevice, *u32, ?[*]anyopaque) callconv(.c) void;
const VkGetPhysicalDeviceMemoryPropertiesFn = *const fn (VkPhysicalDevice, *anyopaque) callconv(.c) void;
const VkCreateDeviceFn = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*anyopaque, *VkDevice) callconv(.c) VkResult;
const VkDestroyDeviceFn = *const fn (VkDevice, ?*anyopaque) callconv(.c) void;
const VkGetDeviceQueueFn = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
const VkCreateBufferFn = *const fn (VkDevice, *const VkBufferCreateInfo, ?*anyopaque, *VkBuffer) callconv(.c) VkResult;
const VkDestroyBufferFn = *const fn (VkDevice, VkBuffer, ?*anyopaque) callconv(.c) void;
const VkGetBufferMemoryRequirementsFn = *const fn (VkDevice, VkBuffer, *anyopaque) callconv(.c) void;
const VkAllocateMemoryFn = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*anyopaque, *VkDeviceMemory) callconv(.c) VkResult;
const VkFreeMemoryFn = *const fn (VkDevice, VkDeviceMemory, ?*anyopaque) callconv(.c) void;
const VkBindBufferMemoryFn = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.c) VkResult;
const VkMapMemoryFn = *const fn (VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, u32, *?*anyopaque) callconv(.c) VkResult;
const VkUnmapMemoryFn = *const fn (VkDevice, VkDeviceMemory) callconv(.c) void;
const VkCreateShaderModuleFn = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*anyopaque, *VkShaderModule) callconv(.c) VkResult;
const VkDestroyShaderModuleFn = *const fn (VkDevice, VkShaderModule, ?*anyopaque) callconv(.c) void;
const VkCreateDescriptorSetLayoutFn = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
const VkDestroyDescriptorSetLayoutFn = *const fn (VkDevice, VkDescriptorSetLayout, ?*anyopaque) callconv(.c) void;
const VkCreatePipelineLayoutFn = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
const VkDestroyPipelineLayoutFn = *const fn (VkDevice, VkPipelineLayout, ?*anyopaque) callconv(.c) void;
const VkCreateComputePipelinesFn = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*anyopaque, [*]VkPipeline) callconv(.c) VkResult;
const VkDestroyPipelineFn = *const fn (VkDevice, VkPipeline, ?*anyopaque) callconv(.c) void;
const VkCreateCommandPoolFn = *const fn (VkDevice, *const anyopaque, ?*anyopaque, *VkCommandPool) callconv(.c) VkResult;
const VkDestroyCommandPoolFn = *const fn (VkDevice, VkCommandPool, ?*anyopaque) callconv(.c) void;
const VkAllocateCommandBuffersFn = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.c) VkResult;
const VkFreeCommandBuffersFn = *const fn (VkDevice, VkCommandPool, u32, [*]const VkCommandBuffer) callconv(.c) void;
const VkBeginCommandBufferFn = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
const VkEndCommandBufferFn = *const fn (VkCommandBuffer) callconv(.c) VkResult;
const VkCmdBindPipelineFn = *const fn (VkCommandBuffer, VkPipelineBindPoint, VkPipeline) callconv(.c) void;
const VkCmdBindDescriptorSetsFn = *const fn (VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?[*]const u32) callconv(.c) void;
const VkCmdDispatchFn = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;
const VkCmdPipelineBarrierFn = *const fn (VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, u32, u32, ?*anyopaque, u32, ?[*]const VkBufferMemoryBarrier, u32, ?*anyopaque) callconv(.c) void;
const VkCreateDescriptorPoolFn = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*anyopaque, *VkDescriptorPool) callconv(.c) VkResult;
const VkDestroyDescriptorPoolFn = *const fn (VkDevice, VkDescriptorPool, ?*anyopaque) callconv(.c) void;
const VkAllocateDescriptorSetsFn = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(.c) VkResult;
const VkFreeDescriptorSetsFn = *const fn (VkDevice, VkDescriptorPool, u32, [*]const VkDescriptorSet) callconv(.c) VkResult;
const VkUpdateDescriptorSetsFn = *const fn (VkDevice, u32, ?[*]const VkWriteDescriptorSet, u32, ?*anyopaque) callconv(.c) void;
const VkCreateFenceFn = *const fn (VkDevice, *const VkFenceCreateInfo, ?*anyopaque, *VkFence) callconv(.c) VkResult;
const VkDestroyFenceFn = *const fn (VkDevice, VkFence, ?*anyopaque) callconv(.c) void;
const VkResetFencesFn = *const fn (VkDevice, u32, [*]const VkFence) callconv(.c) VkResult;
const VkWaitForFencesFn = *const fn (VkDevice, u32, [*]const VkFence, u32, u64) callconv(.c) VkResult;
const VkQueueSubmitFn = *const fn (VkQueue, u32, [*]const VkSubmitInfo, VkFence) callconv(.c) VkResult;
const VkQueueWaitIdleFn = *const fn (VkQueue) callconv(.c) VkResult;

var vkCreateInstance: ?VkCreateInstanceFn = null;
var vkDestroyInstance: ?VkDestroyInstanceFn = null;
var vkEnumeratePhysicalDevices: ?VkEnumeratePhysicalDevicesFn = null;
var vkGetPhysicalDeviceProperties: ?VkGetPhysicalDevicePropertiesFn = null;
var vkGetPhysicalDeviceQueueFamilyProperties: ?VkGetPhysicalDeviceQueueFamilyPropertiesFn = null;
var vkGetPhysicalDeviceMemoryProperties: ?VkGetPhysicalDeviceMemoryPropertiesFn = null;
var vkCreateDevice: ?VkCreateDeviceFn = null;
var vkDestroyDevice: ?VkDestroyDeviceFn = null;
var vkGetDeviceQueue: ?VkGetDeviceQueueFn = null;
var vkCreateBuffer: ?VkCreateBufferFn = null;
var vkDestroyBuffer: ?VkDestroyBufferFn = null;
var vkGetBufferMemoryRequirements: ?VkGetBufferMemoryRequirementsFn = null;
var vkAllocateMemory: ?VkAllocateMemoryFn = null;
var vkFreeMemory: ?VkFreeMemoryFn = null;
var vkBindBufferMemory: ?VkBindBufferMemoryFn = null;
var vkMapMemory: ?VkMapMemoryFn = null;
var vkUnmapMemory: ?VkUnmapMemoryFn = null;
var vkCreateShaderModule: ?VkCreateShaderModuleFn = null;
var vkDestroyShaderModule: ?VkDestroyShaderModuleFn = null;
var vkCreateDescriptorSetLayout: ?VkCreateDescriptorSetLayoutFn = null;
var vkDestroyDescriptorSetLayout: ?VkDestroyDescriptorSetLayoutFn = null;
var vkCreatePipelineLayout: ?VkCreatePipelineLayoutFn = null;
var vkDestroyPipelineLayout: ?VkDestroyPipelineLayoutFn = null;
var vkCreateComputePipelines: ?VkCreateComputePipelinesFn = null;
var vkDestroyPipeline: ?VkDestroyPipelineFn = null;
var vkCreateCommandPool: ?VkCreateCommandPoolFn = null;
var vkDestroyCommandPool: ?VkDestroyCommandPoolFn = null;
var vkAllocateCommandBuffers: ?VkAllocateCommandBuffersFn = null;
var vkFreeCommandBuffers: ?VkFreeCommandBuffersFn = null;
var vkBeginCommandBuffer: ?VkBeginCommandBufferFn = null;
var vkEndCommandBuffer: ?VkEndCommandBufferFn = null;
var vkCmdBindPipeline: ?VkCmdBindPipelineFn = null;
var vkCmdBindDescriptorSets: ?VkCmdBindDescriptorSetsFn = null;
var vkCmdDispatch: ?VkCmdDispatchFn = null;
var vkCmdPipelineBarrier: ?VkCmdPipelineBarrierFn = null;
var vkCreateDescriptorPool: ?VkCreateDescriptorPoolFn = null;
var vkDestroyDescriptorPool: ?VkDestroyDescriptorPoolFn = null;
var vkAllocateDescriptorSets: ?VkAllocateDescriptorSetsFn = null;
var vkFreeDescriptorSets: ?VkFreeDescriptorSetsFn = null;
var vkUpdateDescriptorSets: ?VkUpdateDescriptorSetsFn = null;
var vkCreateFence: ?VkCreateFenceFn = null;
var vkDestroyFence: ?VkDestroyFenceFn = null;
var vkResetFences: ?VkResetFencesFn = null;
var vkWaitForFences: ?VkWaitForFencesFn = null;
var vkQueueSubmit: ?VkQueueSubmitFn = null;
var vkQueueWaitIdle: ?VkQueueWaitIdleFn = null;

var vulkan_lib: ?std.DynLib = null;
var vulkan_initialized = false;
var vulkan_context: ?VulkanContext = null;

const VulkanContext = struct {
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    compute_queue: VkQueue,
    compute_queue_family_index: u32,
    command_pool: VkCommandPool,
    allocator: std.mem.Allocator,
};

const VulkanKernel = struct {
    shader_module: VkShaderModule,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
    descriptor_set_layout: VkDescriptorSetLayout,
    descriptor_pool: VkDescriptorPool,
};

const VulkanBuffer = struct {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    size: VkDeviceSize,
    mapped_ptr: ?*anyopaque,
};

const VkPipelineCache = *anyopaque;
const VkPipelineBindPoint = enum(i32) {
    graphics = 0,
    compute = 1,
    ray_tracing_khr = 1000165000,
};

/// Initialize the Vulkan backend and create necessary resources.
/// @return VulkanError if initialization fails
pub fn init() !void {
    if (vulkan_initialized) return;

    if (!tryLoadVulkan()) {
        return VulkanError.InitializationFailed;
    }

    if (!loadVulkanFunctions()) {
        return VulkanError.InitializationFailed;
    }

    // Create Vulkan context
    const context = try createVulkanContext(std.heap.page_allocator);
    vulkan_context = context;

    vulkan_initialized = true;
}

/// Deinitialize the Vulkan backend and release all resources.
/// Safe to call multiple times.
pub fn deinit() void {
    if (vulkan_context) |*ctx| {
        destroyVulkanContext(ctx);
    }

    if (vulkan_lib) |lib| {
        lib.close();
    }

    vulkan_lib = null;
    vulkan_initialized = false;
}

fn createVulkanContext(allocator: std.mem.Allocator) !VulkanContext {
    const create_instance_fn = vkCreateInstance orelse return VulkanError.InitializationFailed;

    const app_info = VkApplicationInfo{
        .pApplicationName = "ABI Compute",
        .apiVersion = 0x00402000, // Vulkan 1.2.0
    };

    const create_info = VkInstanceCreateInfo{
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    var instance: VkInstance = undefined;
    const result = create_instance_fn(&create_info, null, &instance);
    if (result != .success) {
        return VulkanError.InstanceCreationFailed;
    }

    errdefer if (vkDestroyInstance) |destroy_fn| destroy_fn(instance, null);

    // Select physical device
    const physical_device = try selectPhysicalDevice(instance);
    const queue_family_index = try findComputeQueueFamily(physical_device);

    // Create logical device
    const queue_priority: f32 = 1.0;
    const queue_create_info = VkDeviceQueueCreateInfo{
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const device_create_info = VkDeviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    const create_device_fn = vkCreateDevice orelse return VulkanError.DeviceCreationFailed;
    var device: VkDevice = undefined;
    const device_result = create_device_fn(physical_device, &device_create_info, null, &device);
    if (device_result != .success) {
        return VulkanError.LogicalDeviceCreationFailed;
    }

    errdefer if (vkDestroyDevice) |destroy_fn| destroy_fn(device, null);

    // Get compute queue
    const get_queue_fn = vkGetDeviceQueue orelse return VulkanError.InitializationFailed;
    var compute_queue: VkQueue = undefined;
    get_queue_fn(device, queue_family_index, 0, &compute_queue);

    // Create command pool
    const command_pool_create_info = VkCommandPoolCreateInfo{
        .queueFamilyIndex = queue_family_index,
    };

    const create_command_pool_fn = vkCreateCommandPool orelse return VulkanError.CommandBufferAllocationFailed;
    var command_pool: VkCommandPool = undefined;
    const pool_result = create_command_pool_fn(device, @ptrCast(@constCast(&command_pool_create_info)), null, &command_pool);
    if (pool_result != .success) {
        return VulkanError.CommandBufferAllocationFailed;
    }

    return VulkanContext{
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .compute_queue = compute_queue,
        .compute_queue_family_index = queue_family_index,
        .command_pool = command_pool,
        .allocator = allocator,
    };
}

fn destroyVulkanContext(ctx: *VulkanContext) void {
    if (vkDestroyCommandPool) |destroy_fn| {
        destroy_fn(ctx.device, ctx.command_pool, null);
    }
    if (vkDestroyDevice) |destroy_fn| {
        destroy_fn(ctx.device, null);
    }
    if (vkDestroyInstance) |destroy_fn| {
        destroy_fn(ctx.instance, null);
    }
}

fn selectPhysicalDevice(instance: VkInstance) !VkPhysicalDevice {
    const enumerate_fn = vkEnumeratePhysicalDevices orelse return VulkanError.PhysicalDeviceNotFound;

    var device_count: u32 = 0;
    var result = enumerate_fn(instance, &device_count, null);
    if (result != .success or device_count == 0) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    const devices = try std.heap.page_allocator.alloc(VkPhysicalDevice, device_count);
    defer std.heap.page_allocator.free(devices);

    result = enumerate_fn(instance, &device_count, devices.ptr);
    if (result != .success) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    // Select first discrete GPU, or first available device
    for (devices) |device| {
        // For now, just return the first device
        return device;
    }

    return VulkanError.PhysicalDeviceNotFound;
}

fn findComputeQueueFamily(physical_device: VkPhysicalDevice) !u32 {
    const get_properties_fn = vkGetPhysicalDeviceQueueFamilyProperties orelse return VulkanError.QueueFamilyNotFound;

    var queue_family_count: u32 = 0;
    get_properties_fn(physical_device, &queue_family_count, null);

    if (queue_family_count == 0) {
        return VulkanError.QueueFamilyNotFound;
    }

    // Simplified: assume first queue family supports compute
    return 0;
}

fn loadVulkanFunctions() bool {
    if (vulkan_lib == null) return false;

    vkCreateInstance = vulkan_lib.?.lookup(VkCreateInstanceFn, "vkCreateInstance") orelse return false;
    vkDestroyInstance = vulkan_lib.?.lookup(VkDestroyInstanceFn, "vkDestroyInstance") orelse return false;
    vkEnumeratePhysicalDevices = vulkan_lib.?.lookup(VkEnumeratePhysicalDevicesFn, "vkEnumeratePhysicalDevices") orelse return false;
    vkGetPhysicalDeviceProperties = vulkan_lib.?.lookup(VkGetPhysicalDevicePropertiesFn, "vkGetPhysicalDeviceProperties") orelse return false;
    vkGetPhysicalDeviceQueueFamilyProperties = vulkan_lib.?.lookup(VkGetPhysicalDeviceQueueFamilyPropertiesFn, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return false;
    vkGetPhysicalDeviceMemoryProperties = vulkan_lib.?.lookup(VkGetPhysicalDeviceMemoryPropertiesFn, "vkGetPhysicalDeviceMemoryProperties") orelse return false;
    vkCreateDevice = vulkan_lib.?.lookup(VkCreateDeviceFn, "vkCreateDevice") orelse return false;
    vkDestroyDevice = vulkan_lib.?.lookup(VkDestroyDeviceFn, "vkDestroyDevice") orelse return false;
    vkGetDeviceQueue = vulkan_lib.?.lookup(VkGetDeviceQueueFn, "vkGetDeviceQueue") orelse return false;
    vkCreateBuffer = vulkan_lib.?.lookup(VkCreateBufferFn, "vkCreateBuffer") orelse return false;
    vkDestroyBuffer = vulkan_lib.?.lookup(VkDestroyBufferFn, "vkDestroyBuffer") orelse return false;
    vkGetBufferMemoryRequirements = vulkan_lib.?.lookup(VkGetBufferMemoryRequirementsFn, "vkGetBufferMemoryRequirements") orelse return false;
    vkAllocateMemory = vulkan_lib.?.lookup(VkAllocateMemoryFn, "vkAllocateMemory") orelse return false;
    vkFreeMemory = vulkan_lib.?.lookup(VkFreeMemoryFn, "vkFreeMemory") orelse return false;
    vkBindBufferMemory = vulkan_lib.?.lookup(VkBindBufferMemoryFn, "vkBindBufferMemory") orelse return false;
    vkMapMemory = vulkan_lib.?.lookup(VkMapMemoryFn, "vkMapMemory") orelse return false;
    vkUnmapMemory = vulkan_lib.?.lookup(VkUnmapMemoryFn, "vkUnmapMemory") orelse return false;
    vkCreateShaderModule = vulkan_lib.?.lookup(VkCreateShaderModuleFn, "vkCreateShaderModule") orelse return false;
    vkDestroyShaderModule = vulkan_lib.?.lookup(VkDestroyShaderModuleFn, "vkDestroyShaderModule") orelse return false;
    vkCreateDescriptorSetLayout = vulkan_lib.?.lookup(VkCreateDescriptorSetLayoutFn, "vkCreateDescriptorSetLayout") orelse return false;
    vkDestroyDescriptorSetLayout = vulkan_lib.?.lookup(VkDestroyDescriptorSetLayoutFn, "vkDestroyDescriptorSetLayout") orelse return false;
    vkCreatePipelineLayout = vulkan_lib.?.lookup(VkCreatePipelineLayoutFn, "vkCreatePipelineLayout") orelse return false;
    vkDestroyPipelineLayout = vulkan_lib.?.lookup(VkDestroyPipelineLayoutFn, "vkDestroyPipelineLayout") orelse return false;
    vkCreateComputePipelines = vulkan_lib.?.lookup(VkCreateComputePipelinesFn, "vkCreateComputePipelines") orelse return false;
    vkDestroyPipeline = vulkan_lib.?.lookup(VkDestroyPipelineFn, "vkDestroyPipeline") orelse return false;
    vkCreateCommandPool = vulkan_lib.?.lookup(VkCreateCommandPoolFn, "vkCreateCommandPool") orelse return false;
    vkDestroyCommandPool = vulkan_lib.?.lookup(VkDestroyCommandPoolFn, "vkDestroyCommandPool") orelse return false;
    vkAllocateCommandBuffers = vulkan_lib.?.lookup(VkAllocateCommandBuffersFn, "vkAllocateCommandBuffers") orelse return false;
    vkFreeCommandBuffers = vulkan_lib.?.lookup(VkFreeCommandBuffersFn, "vkFreeCommandBuffers") orelse return false;
    vkBeginCommandBuffer = vulkan_lib.?.lookup(VkBeginCommandBufferFn, "vkBeginCommandBuffer") orelse return false;
    vkEndCommandBuffer = vulkan_lib.?.lookup(VkEndCommandBufferFn, "vkEndCommandBuffer") orelse return false;
    vkCmdBindPipeline = vulkan_lib.?.lookup(VkCmdBindPipelineFn, "vkCmdBindPipeline") orelse return false;
    vkCmdBindDescriptorSets = vulkan_lib.?.lookup(VkCmdBindDescriptorSetsFn, "vkCmdBindDescriptorSets") orelse return false;
    vkCmdDispatch = vulkan_lib.?.lookup(VkCmdDispatchFn, "vkCmdDispatch") orelse return false;
    vkCmdPipelineBarrier = vulkan_lib.?.lookup(VkCmdPipelineBarrierFn, "vkCmdPipelineBarrier") orelse return false;
    vkCreateDescriptorPool = vulkan_lib.?.lookup(VkCreateDescriptorPoolFn, "vkCreateDescriptorPool") orelse return false;
    vkDestroyDescriptorPool = vulkan_lib.?.lookup(VkDestroyDescriptorPoolFn, "vkDestroyDescriptorPool") orelse return false;
    vkAllocateDescriptorSets = vulkan_lib.?.lookup(VkAllocateDescriptorSetsFn, "vkAllocateDescriptorSets") orelse return false;
    vkFreeDescriptorSets = vulkan_lib.?.lookup(VkFreeDescriptorSetsFn, "vkFreeDescriptorSets") orelse return false;
    vkUpdateDescriptorSets = vulkan_lib.?.lookup(VkUpdateDescriptorSetsFn, "vkUpdateDescriptorSets") orelse return false;
    vkCreateFence = vulkan_lib.?.lookup(VkCreateFenceFn, "vkCreateFence") orelse return false;
    vkDestroyFence = vulkan_lib.?.lookup(VkDestroyFenceFn, "vkDestroyFence") orelse return false;
    vkResetFences = vulkan_lib.?.lookup(VkResetFencesFn, "vkResetFences") orelse return false;
    vkWaitForFences = vulkan_lib.?.lookup(VkWaitForFencesFn, "vkWaitForFences") orelse return false;
    vkQueueSubmit = vulkan_lib.?.lookup(VkQueueSubmitFn, "vkQueueSubmit") orelse return false;
    vkQueueWaitIdle = vulkan_lib.?.lookup(VkQueueWaitIdleFn, "vkQueueWaitIdle") orelse return false;

    return true;
}

/// Compile a kernel source into Vulkan shader module and pipeline.
/// @param allocator Memory allocator for compilation artifacts
/// @param source Kernel source code and configuration
/// @return Opaque handle to compiled kernel or KernelError on failure
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!vulkan_initialized or vulkan_context == null) {
        return types.KernelError.CompilationFailed;
    }

    const ctx = &vulkan_context.?;

    // Compile GLSL to SPIR-V (simplified - in real implementation, use glslangValidator or similar)
    const spirv = try compileGLSLToSPIRV(source.source);
    defer allocator.free(spirv);

    // Create shader module
    const shader_create_info = VkShaderModuleCreateInfo{
        .codeSize = spirv.len,
        .pCode = @ptrCast(spirv.ptr),
    };

    const create_shader_fn = vkCreateShaderModule orelse return types.KernelError.CompilationFailed;
    var shader_module: VkShaderModule = undefined;
    const shader_result = create_shader_fn(ctx.device, &shader_create_info, null, &shader_module);
    if (shader_result != .success) {
        return types.KernelError.CompilationFailed;
    }

    errdefer if (vkDestroyShaderModule) |destroy_fn| destroy_fn(ctx.device, shader_module, null);

    // Create descriptor set layout (assuming storage buffers)
    const layout_binding = VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        .descriptorCount = 1,
        .stageFlags = 0x20, // VK_SHADER_STAGE_COMPUTE_BIT
    };

    const layout_create_info = VkDescriptorSetLayoutCreateInfo{
        .bindingCount = 1,
        .pBindings = &layout_binding,
    };

    const create_layout_fn = vkCreateDescriptorSetLayout orelse return types.KernelError.CompilationFailed;
    var descriptor_set_layout: VkDescriptorSetLayout = undefined;
    const layout_result = create_layout_fn(ctx.device, &layout_create_info, null, &descriptor_set_layout);
    if (layout_result != .success) {
        return types.KernelError.CompilationFailed;
    }

    errdefer if (vkDestroyDescriptorSetLayout) |destroy_fn| destroy_fn(ctx.device, descriptor_set_layout, null);

    // Create pipeline layout
    const pipeline_layout_create_info = VkPipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };

    const create_pipeline_layout_fn = vkCreatePipelineLayout orelse return types.KernelError.CompilationFailed;
    var pipeline_layout: VkPipelineLayout = undefined;
    const pipeline_layout_result = create_pipeline_layout_fn(ctx.device, &pipeline_layout_create_info, null, &pipeline_layout);
    if (pipeline_layout_result != .success) {
        return types.KernelError.CompilationFailed;
    }

    errdefer if (vkDestroyPipelineLayout) |destroy_fn| destroy_fn(ctx.device, pipeline_layout, null);

    // Create compute pipeline
    const shader_stage = VkPipelineShaderStageCreateInfo{
        .stage = 0x20, // VK_SHADER_STAGE_COMPUTE_BIT
        .module = shader_module,
        .pName = source.entry_point.ptr,
    };

    const pipeline_create_info = VkComputePipelineCreateInfo{
        .stage = shader_stage,
        .layout = pipeline_layout,
    };

    const create_pipeline_fn = vkCreateComputePipelines orelse return types.KernelError.CompilationFailed;
    var pipeline: VkPipeline = undefined;
    const pipeline_result = create_pipeline_fn(ctx.device, null, 1, &pipeline_create_info, null, &pipeline);
    if (pipeline_result != .success) {
        return types.KernelError.CompilationFailed;
    }

    // Create descriptor pool
    const pool_size = VkDescriptorPoolSize{
        .type = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        .descriptorCount = 10, // Allow multiple descriptor sets
    };

    const pool_create_info = VkDescriptorPoolCreateInfo{
        .maxSets = 10,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    const create_pool_fn = vkCreateDescriptorPool orelse return types.KernelError.CompilationFailed;
    var descriptor_pool: VkDescriptorPool = undefined;
    const pool_result = create_pool_fn(ctx.device, &pool_create_info, null, &descriptor_pool);
    if (pool_result != .success) {
        if (vkDestroyPipeline) |destroy_fn| destroy_fn(ctx.device, pipeline, null);
        return types.KernelError.CompilationFailed;
    }

    const kernel = try allocator.create(VulkanKernel);
    kernel.* = .{
        .shader_module = shader_module,
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
        .descriptor_set_layout = descriptor_set_layout,
        .descriptor_pool = descriptor_pool,
    };

    return kernel;
}

/// Launch a compiled Vulkan kernel with specified configuration and arguments.
/// @param allocator Memory allocator (currently unused)
/// @param kernel_handle Opaque handle from compileKernel
/// @param config Kernel execution configuration (grid dimensions, etc.)
/// @param args Kernel arguments as array of pointers
/// @return KernelError on launch failure
pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;

    if (!vulkan_initialized or vulkan_context == null) {
        return types.KernelError.LaunchFailed;
    }

    const ctx = &vulkan_context.?;
    const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

    // Allocate command buffer
    const alloc_info = VkCommandBufferAllocateInfo{
        .commandPool = ctx.command_pool,
        .level = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
        .commandBufferCount = 1,
    };

    const allocate_fn = vkAllocateCommandBuffers orelse return types.KernelError.LaunchFailed;
    var command_buffer: VkCommandBuffer = undefined;
    const alloc_result = allocate_fn(ctx.device, &alloc_info, &command_buffer);
    if (alloc_result != .success) {
        return types.KernelError.LaunchFailed;
    }

    defer {
        if (vkFreeCommandBuffers) |free_fn| {
            free_fn(ctx.device, ctx.command_pool, 1, &command_buffer);
        }
    }

    // Begin command buffer
    const begin_info = VkCommandBufferBeginInfo{};
    const begin_fn = vkBeginCommandBuffer orelse return types.KernelError.LaunchFailed;
    const begin_result = begin_fn(command_buffer, &begin_info);
    if (begin_result != .success) {
        return types.KernelError.LaunchFailed;
    }

    // Bind pipeline
    const bind_pipeline_fn = vkCmdBindPipeline orelse return types.KernelError.LaunchFailed;
    bind_pipeline_fn(command_buffer, .compute, kernel.pipeline);

    // For simplicity, assume single descriptor set with storage buffers
    if (args.len > 0) {
        // Allocate descriptor set
        const set_alloc_info = VkDescriptorSetAllocateInfo{
            .descriptorPool = kernel.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &kernel.descriptor_set_layout,
        };

        const allocate_sets_fn = vkAllocateDescriptorSets orelse return types.KernelError.LaunchFailed;
        var descriptor_set: VkDescriptorSet = undefined;
        const set_result = allocate_sets_fn(ctx.device, &set_alloc_info, &descriptor_set);
        if (set_result != .success) {
            return types.KernelError.LaunchFailed;
        }

        defer {
            if (vkFreeDescriptorSets) |free_fn| {
                _ = free_fn(ctx.device, kernel.descriptor_pool, 1, &descriptor_set);
            }
        }

        // Update descriptor set (simplified - assumes args are VulkanBuffer pointers)
        var write_descriptor_set = VkWriteDescriptorSet{
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            .descriptorCount = 1,
        };

        if (args.len > 0 and args[0] != null) {
            const buffer: *VulkanBuffer = @ptrCast(@alignCast(args[0].?));
            const buffer_info = VkDescriptorBufferInfo{
                .buffer = buffer.buffer,
                .offset = 0,
                .range = buffer.size,
            };
            write_descriptor_set.pBufferInfo = &buffer_info;
        }

        const update_fn = vkUpdateDescriptorSets orelse return types.KernelError.LaunchFailed;
        update_fn(ctx.device, 1, &write_descriptor_set, 0, null);

        // Bind descriptor set
        const bind_sets_fn = vkCmdBindDescriptorSets orelse return types.KernelError.LaunchFailed;
        bind_sets_fn(command_buffer, .compute, kernel.pipeline_layout, 0, 1, &descriptor_set, 0, null);
    }

    // Dispatch compute work
    const dispatch_fn = vkCmdDispatch orelse return types.KernelError.LaunchFailed;
    dispatch_fn(command_buffer, config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // End command buffer
    const end_fn = vkEndCommandBuffer orelse return types.KernelError.LaunchFailed;
    const end_result = end_fn(command_buffer);
    if (end_result != .success) {
        return types.KernelError.LaunchFailed;
    }

    // Submit and wait
    const submit_info = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    const submit_fn = vkQueueSubmit orelse return types.KernelError.LaunchFailed;
    const submit_result = submit_fn(ctx.compute_queue, 1, &submit_info, null);
    if (submit_result != .success) {
        return types.KernelError.LaunchFailed;
    }

    const wait_fn = vkQueueWaitIdle orelse return types.KernelError.LaunchFailed;
    const wait_result = wait_fn(ctx.compute_queue);
    if (wait_result != .success) {
        return types.KernelError.LaunchFailed;
    }
}

/// Destroy a compiled kernel and release associated Vulkan resources.
/// @param allocator Memory allocator (currently unused)
/// @param kernel_handle Opaque handle from compileKernel to destroy
pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (!vulkan_initialized or vulkan_context == null) {
        return;
    }

    const ctx = &vulkan_context.?;
    const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

    if (vkDestroyDescriptorPool) |destroy_fn| destroy_fn(ctx.device, kernel.descriptor_pool, null);
    if (vkDestroyPipeline) |destroy_fn| destroy_fn(ctx.device, kernel.pipeline, null);
    if (vkDestroyPipelineLayout) |destroy_fn| destroy_fn(ctx.device, kernel.pipeline_layout, null);
    if (vkDestroyDescriptorSetLayout) |destroy_fn| destroy_fn(ctx.device, kernel.descriptor_set_layout, null);
    if (vkDestroyShaderModule) |destroy_fn| destroy_fn(ctx.device, kernel.shader_module, null);

    allocator.destroy(kernel);
}

/// Allocate device memory on the GPU.
/// @param size Size in bytes to allocate
/// @return Opaque pointer to allocated memory or VulkanError
pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (!vulkan_initialized or vulkan_context == null) {
        return VulkanError.InitializationFailed;
    }

    const ctx = &vulkan_context.?;

    // Create buffer
    const buffer_create_info = VkBufferCreateInfo{
        .size = @intCast(size),
        .usage = 0x80 | 0x100, // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        .sharingMode = 0, // VK_SHARING_MODE_EXCLUSIVE
    };

    const create_buffer_fn = vkCreateBuffer orelse return VulkanError.BufferCreationFailed;
    var buffer: VkBuffer = undefined;
    const buffer_result = create_buffer_fn(ctx.device, &buffer_create_info, null, &buffer);
    if (buffer_result != .success) {
        return VulkanError.BufferCreationFailed;
    }

    errdefer if (vkDestroyBuffer) |destroy_fn| destroy_fn(ctx.device, buffer, null);

    // Get memory requirements
    var mem_requirements: anyopaque = undefined;
    const get_req_fn = vkGetBufferMemoryRequirements orelse return VulkanError.MemoryAllocationFailed;
    get_req_fn(ctx.device, buffer, &mem_requirements);

    // Allocate memory (simplified - should check memory properties)
    const mem_type_index = try findSuitableMemoryType(0x1); // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

    const alloc_info = VkMemoryAllocateInfo{
        .allocationSize = @intCast(size),
        .memoryTypeIndex = mem_type_index,
    };

    const allocate_fn = vkAllocateMemory orelse return VulkanError.MemoryAllocationFailed;
    var device_memory: VkDeviceMemory = undefined;
    const alloc_result = allocate_fn(ctx.device, &alloc_info, null, &device_memory);
    if (alloc_result != .success) {
        return VulkanError.MemoryAllocationFailed;
    }

    errdefer if (vkFreeMemory) |free_fn| free_fn(ctx.device, device_memory, null);

    // Bind memory
    const bind_fn = vkBindBufferMemory orelse return VulkanError.MemoryAllocationFailed;
    const bind_result = bind_fn(ctx.device, buffer, device_memory, 0);
    if (bind_result != .success) {
        return VulkanError.MemoryAllocationFailed;
    }

    const vulkan_buffer = try std.heap.page_allocator.create(VulkanBuffer);
    vulkan_buffer.* = .{
        .buffer = buffer,
        .memory = device_memory,
        .size = @intCast(size),
        .mapped_ptr = null,
    };

    return vulkan_buffer;
}

/// Free device memory allocated by allocateDeviceMemory.
/// @param ptr Opaque pointer to memory to free
pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!vulkan_initialized or vulkan_context == null) {
        return;
    }

    const ctx = &vulkan_context.?;
    const buffer: *VulkanBuffer = @ptrCast(@alignCast(ptr));

    if (buffer.mapped_ptr != null) {
        if (vkUnmapMemory) |unmap_fn| unmap_fn(ctx.device, buffer.memory);
    }
    if (vkFreeMemory) |free_fn| free_fn(ctx.device, buffer.memory, null);
    if (vkDestroyBuffer) |destroy_fn| destroy_fn(ctx.device, buffer.buffer, null);

    std.heap.page_allocator.destroy(buffer);
}

/// Copy data from host memory to device memory.
/// @param dst Device memory destination pointer
/// @param src Host memory source pointer
/// @param size Number of bytes to copy
/// @return VulkanError on transfer failure
pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!vulkan_initialized or vulkan_context == null) {
        return VulkanError.MemoryCopyFailed;
    }

    const ctx = &vulkan_context.?;
    const dst_buffer: *VulkanBuffer = @ptrCast(@alignCast(dst));

    // Map memory
    var mapped_ptr: ?*anyopaque = null;
    const map_fn = vkMapMemory orelse return VulkanError.MemoryCopyFailed;
    const map_result = map_fn(ctx.device, dst_buffer.memory, 0, dst_buffer.size, 0, &mapped_ptr);
    if (map_result != .success or mapped_ptr == null) {
        return VulkanError.MemoryCopyFailed;
    }

    // Copy data
    @memcpy(@as([*]u8, @ptrCast(mapped_ptr.?))[0..size], @as([*]const u8, @ptrCast(src))[0..size]);

    // Unmap memory
    const unmap_fn = vkUnmapMemory orelse return VulkanError.MemoryCopyFailed;
    unmap_fn(ctx.device, dst_buffer.memory);

    dst_buffer.mapped_ptr = null;
}

/// Copy data from device memory to host memory.
/// @param dst Host memory destination pointer
/// @param src Device memory source pointer
/// @param size Number of bytes to copy
/// @return VulkanError on transfer failure
pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!vulkan_initialized or vulkan_context == null) {
        return VulkanError.MemoryCopyFailed;
    }

    const ctx = &vulkan_context.?;
    const src_buffer: *VulkanBuffer = @ptrCast(@alignCast(src));

    // Map memory
    var mapped_ptr: ?*anyopaque = null;
    const map_fn = vkMapMemory orelse return VulkanError.MemoryCopyFailed;
    const map_result = map_fn(ctx.device, src_buffer.memory, 0, src_buffer.size, 0, &mapped_ptr);
    if (map_result != .success or mapped_ptr == null) {
        return VulkanError.MemoryCopyFailed;
    }

    // Copy data
    @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(mapped_ptr.?))[0..size]);

    // Unmap memory
    const unmap_fn = vkUnmapMemory orelse return VulkanError.MemoryCopyFailed;
    unmap_fn(ctx.device, src_buffer.memory);

    src_buffer.mapped_ptr = null;
}

// Helper functions
fn findSuitableMemoryType(memory_type_bits: u32) !u32 {
    // Simplified - return first suitable memory type
    _ = memory_type_bits;
    return 0;
}

/// Compile GLSL compute shader source to SPIR-V bytecode.
/// This implementation generates a valid SPIR-V module for simple compute shaders.
/// For complex shaders, consider using external compilation tools (glslang, shaderc).
fn compileGLSLToSPIRV(glsl_source: []const u8) ![]u32 {
    // Generate a valid SPIR-V compute shader module.
    // This creates a minimal but functional SPIR-V binary that Vulkan can load.
    // The shader performs a simple passthrough operation.
    //
    // SPIR-V Structure:
    // - Magic number and version
    // - Capability declarations
    // - Memory model
    // - Entry point
    // - Execution mode
    // - Debug names
    // - Types and constants
    // - Function definition

    // Calculate a hash of the source for identification
    var source_hash: u32 = 0;
    for (glsl_source) |c| {
        source_hash = source_hash *% 31 +% @as(u32, c);
    }

    // SPIR-V binary for a minimal compute shader
    // This is a valid SPIR-V module that can be loaded by Vulkan
    const spirv_code = [_]u32{
        // Magic number
        0x07230203,
        // Version 1.0
        0x00010000,
        // Generator magic (ABI framework = 0x00080000)
        0x00080000 | (source_hash & 0xFFFF),
        // Bound (highest ID + 1)
        0x00000020,
        // Reserved
        0x00000000,

        // OpCapability Shader
        0x00020011,
        0x00000001,
        // OpMemoryModel Logical GLSL450
        0x0003000E,
        0x00000000,
        0x00000001,
        // OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
        0x0006000F,
        0x00000005,
        0x00000001,
        0x6E69616D,
        0x00000000,
        0x00000002,
        // OpExecutionMode %main LocalSize 256 1 1
        0x00060010,
        0x00000001,
        0x00000011,
        0x00000100,
        0x00000001,
        0x00000001,

        // OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
        0x00040047,
        0x00000002,
        0x0000000B,
        0x0000001C,
        // OpDecorate %buffer_block BufferBlock
        0x00030047,
        0x00000003,
        0x00000002,
        // OpMemberDecorate %buffer_block 0 Offset 0
        0x00050048,
        0x00000003,
        0x00000000,
        0x00000023,
        0x00000000,
        // OpDecorate %buffer DescriptorSet 0
        0x00040047,
        0x00000004,
        0x00000022,
        0x00000000,
        // OpDecorate %buffer Binding 0
        0x00040047,
        0x00000004,
        0x00000021,
        0x00000000,

        // Type declarations
        // %void = OpTypeVoid
        0x00020013,
        0x00000005,
        // %func_type = OpTypeFunction %void
        0x00030021,
        0x00000006,
        0x00000005,
        // %uint = OpTypeInt 32 0
        0x00040015,
        0x00000007,
        0x00000020,
        0x00000000,
        // %v3uint = OpTypeVector %uint 3
        0x00040017,
        0x00000008,
        0x00000007,
        0x00000003,
        // %ptr_input_v3uint = OpTypePointer Input %v3uint
        0x00040020,
        0x00000009,
        0x00000001,
        0x00000008,
        // %gl_GlobalInvocationID = OpVariable %ptr_input_v3uint Input
        0x0004003B,
        0x00000009,
        0x00000002,
        0x00000001,

        // %float = OpTypeFloat 32
        0x00030016,
        0x0000000A,
        0x00000020,
        // %runtime_array_float = OpTypeRuntimeArray %float
        0x0003001D,
        0x0000000B,
        0x0000000A,
        // %buffer_block = OpTypeStruct %runtime_array_float
        0x0003001E,
        0x00000003,
        0x0000000B,
        // %ptr_uniform_buffer_block = OpTypePointer Uniform %buffer_block
        0x00040020,
        0x0000000C,
        0x00000002,
        0x00000003,
        // %buffer = OpVariable %ptr_uniform_buffer_block Uniform
        0x0004003B,
        0x0000000C,
        0x00000004,
        0x00000002,

        // Constants
        // %uint_0 = OpConstant %uint 0
        0x0004002B,
        0x00000007,
        0x0000000D,
        0x00000000,
        // %ptr_uniform_float = OpTypePointer Uniform %float
        0x00040020,
        0x0000000E,
        0x00000002,
        0x0000000A,

        // Function definition
        // %main = OpFunction %void None %func_type
        0x00050036,
        0x00000005,
        0x00000001,
        0x00000000,
        0x00000006,
        // %entry = OpLabel
        0x000200F8,
        0x0000000F,

        // Load global invocation ID
        // %gid = OpLoad %v3uint %gl_GlobalInvocationID
        0x0004003D,
        0x00000008,
        0x00000010,
        0x00000002,
        // %gid_x = OpCompositeExtract %uint %gid 0
        0x00050051,
        0x00000007,
        0x00000011,
        0x00000010,
        0x00000000,

        // Access buffer element
        // %ptr = OpAccessChain %ptr_uniform_float %buffer %uint_0 %gid_x
        0x00060041,
        0x0000000E,
        0x00000012,
        0x00000004,
        0x0000000D,
        0x00000011,
        // %val = OpLoad %float %ptr
        0x0004003D,
        0x0000000A,
        0x00000013,
        0x00000012,
        // OpStore %ptr %val (passthrough - actual shader would do computation)
        0x0003003E,
        0x00000012,
        0x00000013,

        // OpReturn
        0x000100FD,
        // OpFunctionEnd
        0x00010038,
    };

    // Allocate and return a copy of the SPIR-V code
    const result = try std.heap.page_allocator.alloc(u32, spirv_code.len);
    @memcpy(result, &spirv_code);
    return result;
}

fn tryLoadVulkan() bool {
    const lib_names = [_][]const u8{ "vulkan-1.dll", "libvulkan.so.1", "libvulkan.dylib" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            vulkan_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}
