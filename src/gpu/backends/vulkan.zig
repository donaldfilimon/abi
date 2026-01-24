//! Vulkan backend implementation (consolidated).
//!
//! Provides Vulkan-specific kernel compilation, execution, and memory management
//! using the Vulkan API for cross-platform compute acceleration.
//!
//! This is a consolidated module combining type definitions, initialization,
//! pipeline stubs, and buffer stubs.

const std = @import("std");

// ============================================================================
// Errors
// ============================================================================

pub const VulkanError = error{
    InitializationFailed,
    DeviceNotFound,
    InstanceCreationFailed,
    PhysicalDeviceNotFound,
    LogicalDeviceCreationFailed,
    DeviceCreationFailed,
    QueueFamilyNotFound,
    MemoryTypeNotFound,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    CommandBufferAllocationFailed,
    BufferCreationFailed,
    MemoryAllocationFailed,
    MemoryCopyFailed,
    CommandRecordingFailed,
    SubmissionFailed,
    InvalidHandle,
    SynchronizationFailed,
    DeviceLost,
    ValidationLayerNotAvailable,
    NotFound,
};

// ============================================================================
// Vulkan Types
// ============================================================================

pub const VkResult = enum(i32) {
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

// Vulkan handle types
pub const VkInstance = *anyopaque;
pub const VkPhysicalDevice = *anyopaque;
pub const VkDevice = *anyopaque;
pub const VkQueue = *anyopaque;
pub const VkCommandPool = *anyopaque;
pub const VkCommandBuffer = *anyopaque;
pub const VkBuffer = *anyopaque;
pub const VkDeviceMemory = *anyopaque;
pub const VkShaderModule = *anyopaque;
pub const VkPipelineLayout = *anyopaque;
pub const VkPipeline = *anyopaque;
pub const VkDescriptorSetLayout = *anyopaque;
pub const VkDescriptorPool = *anyopaque;
pub const VkDescriptorSet = *anyopaque;
pub const VkFence = *anyopaque;
pub const VkPipelineCache = *anyopaque;

// Vulkan basic types
pub const VkDeviceSize = u64;
pub const VkMemoryPropertyFlags = u32;
pub const VkBufferUsageFlags = u32;
pub const VkShaderStageFlags = u32;
pub const VkPipelineStageFlags = u32;

pub const VkPipelineBindPoint = enum(i32) {
    graphics = 0,
    compute = 1,
    ray_tracing_khr = 1000165000,
};

// Vulkan create info structures
pub const VkApplicationInfo = extern struct {
    sType: i32 = 0,
    pNext: ?*anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

pub const VkInstanceCreateInfo = extern struct {
    sType: i32 = 1,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: i32 = 3,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 0,
    pQueuePriorities: [*]const f32 = null,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: i32 = 3,
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

pub const VkBufferCreateInfo = extern struct {
    sType: i32 = 12,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize = 0,
    usage: VkBufferUsageFlags = 0,
    sharingMode: i32 = 0,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

pub const VkMemoryAllocateInfo = extern struct {
    sType: i32 = 5,
    pNext: ?*anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: i32 = 16,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    codeSize: usize = 0,
    pCode: ?[*]const u32 = null,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: i32 = 0,
    descriptorCount: u32 = 1,
    stageFlags: VkShaderStageFlags = 0,
    pImmutableSamplers: ?*anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: i32 = 32,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: i32 = 30,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?*anyopaque = null,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: i32 = 18,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: VkShaderStageFlags = 0,
    module: VkShaderModule,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*anyopaque = null,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: i32 = 29,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

pub const VkDescriptorPoolSize = extern struct {
    type: i32 = 0,
    descriptorCount: u32 = 0,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: i32 = 33,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: i32 = 34,
    pNext: ?*anyopaque = null,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = 0,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: i32 = 35,
    pNext: ?*anyopaque = null,
    dstSet: VkDescriptorSet,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 1,
    descriptorType: i32 = 0,
    pImageInfo: ?*anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*anyopaque = null,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: i32 = 40,
    pNext: ?*anyopaque = null,
    commandPool: VkCommandPool,
    level: i32 = 0,
    commandBufferCount: u32 = 0,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: i32 = 42,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pInheritanceInfo: ?*anyopaque = null,
};

pub const VkBufferMemoryBarrier = extern struct {
    sType: i32 = 44,
    pNext: ?*anyopaque = null,
    srcAccessMask: u32 = 0,
    dstAccessMask: u32 = 0,
    srcQueueFamilyIndex: u32 = 0,
    dstQueueFamilyIndex: u32 = 0,
    buffer: VkBuffer,
    offset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
};

pub const VkSubmitInfo = extern struct {
    sType: i32 = 4,
    pNext: ?*anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?[*]const *anyopaque = null,
    pWaitDstStageMask: ?[*]const VkPipelineStageFlags = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?[*]const *anyopaque = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: i32 = 8,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: i32 = 39,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
};

pub const VkPipelineCacheCreateInfo = extern struct {
    sType: i32 = 17,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    initialDataSize: usize = 0,
    pInitialData: ?*const anyopaque = null,
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: VkExtent3D,
};

pub const VkExtent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: VkShaderStageFlags = 0,
    offset: u32 = 0,
    size: u32 = 0,
};

pub const VkLayerProperties = extern struct {
    layerName: [256]u8,
    specVersion: u32,
    implementationVersion: u32,
    description: [256]u8,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};

pub const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize,
    flags: u32,
};

pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

pub const VkPhysicalDeviceType = enum(i32) {
    other = 0,
    integrated_gpu = 1,
    discrete_gpu = 2,
    virtual_gpu = 3,
    cpu = 4,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: VkPhysicalDeviceType,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: VkPhysicalDeviceLimits,
    sparseProperties: VkPhysicalDeviceSparseProperties,
};

pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: VkDeviceSize,
    sparseAddressSpaceSize: VkDeviceSize,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputBindings: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: VkDeviceSize,
    minUniformBufferOffsetAlignment: VkDeviceSize,
    minStorageBufferOffsetAlignment: VkDeviceSize,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: u32,
    framebufferDepthSampleCounts: u32,
    framebufferStencilSampleCounts: u32,
    framebufferNoAttachmentsSampleCounts: u32,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: u32,
    sampledImageIntegerSampleCounts: u32,
    sampledImageDepthSampleCounts: u32,
    sampledImageStencilSampleCounts: u32,
    storageImageSampleCounts: u32,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: u32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: u32,
    standardSampleLocations: u32,
    optimalBufferCopyOffsetAlignment: VkDeviceSize,
    optimalBufferCopyRowPitchAlignment: VkDeviceSize,
    nonCoherentAtomSize: VkDeviceSize,
};

pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: u32,
    residencyStandard2DMultisampleBlockShape: u32,
    residencyStandard3DBlockShape: u32,
    residencyAlignedMipSize: u32,
    residencyNonResidentStrict: u32,
};

// Vulkan function pointer types
pub const VkCreateInstanceFn = *const fn (*const VkInstanceCreateInfo, ?*anyopaque, *VkInstance) callconv(.c) VkResult;
pub const VkDestroyInstanceFn = *const fn (VkInstance, ?*anyopaque) callconv(.c) void;
pub const VkEnumeratePhysicalDevicesFn = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
pub const VkGetPhysicalDevicePropertiesFn = *const fn (VkPhysicalDevice, *anyopaque) callconv(.c) void;
pub const VkGetPhysicalDeviceQueueFamilyPropertiesFn = *const fn (VkPhysicalDevice, *u32, ?[*]anyopaque) callconv(.c) void;
pub const VkGetPhysicalDeviceMemoryPropertiesFn = *const fn (VkPhysicalDevice, *anyopaque) callconv(.c) void;
pub const VkCreateDeviceFn = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*anyopaque, *VkDevice) callconv(.c) VkResult;
pub const VkDestroyDeviceFn = *const fn (VkDevice, ?*anyopaque) callconv(.c) void;
pub const VkGetDeviceQueueFn = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
pub const VkCreateBufferFn = *const fn (VkDevice, *const VkBufferCreateInfo, ?*anyopaque, *VkBuffer) callconv(.c) VkResult;
pub const VkDestroyBufferFn = *const fn (VkDevice, VkBuffer, ?*anyopaque) callconv(.c) void;
pub const VkGetBufferMemoryRequirementsFn = *const fn (VkDevice, VkBuffer, *anyopaque) callconv(.c) void;
pub const VkAllocateMemoryFn = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*anyopaque, *VkDeviceMemory) callconv(.c) VkResult;
pub const VkFreeMemoryFn = *const fn (VkDevice, VkDeviceMemory, ?*anyopaque) callconv(.c) void;
pub const VkBindBufferMemoryFn = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.c) VkResult;
pub const VkMapMemoryFn = *const fn (VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, u32, *?*anyopaque) callconv(.c) VkResult;
pub const VkUnmapMemoryFn = *const fn (VkDevice, VkDeviceMemory) callconv(.c) void;
pub const VkCreateShaderModuleFn = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*anyopaque, *VkShaderModule) callconv(.c) VkResult;
pub const VkDestroyShaderModuleFn = *const fn (VkDevice, VkShaderModule, ?*anyopaque) callconv(.c) void;
pub const VkCreateDescriptorSetLayoutFn = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
pub const VkDestroyDescriptorSetLayoutFn = *const fn (VkDevice, VkDescriptorSetLayout, ?*anyopaque) callconv(.c) void;
pub const VkCreatePipelineLayoutFn = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
pub const VkDestroyPipelineLayoutFn = *const fn (VkDevice, VkPipelineLayout, ?*anyopaque) callconv(.c) void;
pub const VkCreateComputePipelinesFn = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*anyopaque, [*]VkPipeline) callconv(.c) VkResult;
pub const VkDestroyPipelineFn = *const fn (VkDevice, VkPipeline, ?*anyopaque) callconv(.c) void;
pub const VkCreateCommandPoolFn = *const fn (VkDevice, *const anyopaque, ?*anyopaque, *VkCommandPool) callconv(.c) VkResult;
pub const VkDestroyCommandPoolFn = *const fn (VkDevice, VkCommandPool, ?*anyopaque) callconv(.c) void;
pub const VkAllocateCommandBuffersFn = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.c) VkResult;
pub const VkFreeCommandBuffersFn = *const fn (VkDevice, VkCommandPool, u32, [*]const VkCommandBuffer) callconv(.c) void;
pub const VkBeginCommandBufferFn = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub const VkEndCommandBufferFn = *const fn (VkCommandBuffer) callconv(.c) VkResult;
pub const VkCmdBindPipelineFn = *const fn (VkCommandBuffer, VkPipelineBindPoint, VkPipeline) callconv(.c) void;
pub const VkCmdBindDescriptorSetsFn = *const fn (VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?[*]const u32) callconv(.c) void;
pub const VkCmdDispatchFn = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;
pub const VkCmdPipelineBarrierFn = *const fn (VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, u32, u32, ?*anyopaque, u32, ?[*]const VkBufferMemoryBarrier, u32, ?*anyopaque) callconv(.c) void;
pub const VkCreateDescriptorPoolFn = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*anyopaque, *VkDescriptorPool) callconv(.c) VkResult;
pub const VkDestroyDescriptorPoolFn = *const fn (VkDevice, VkDescriptorPool, ?*anyopaque) callconv(.c) void;
pub const VkAllocateDescriptorSetsFn = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(.c) VkResult;
pub const VkFreeDescriptorSetsFn = *const fn (VkDevice, VkDescriptorPool, u32, [*]const VkDescriptorSet) callconv(.c) VkResult;
pub const VkUpdateDescriptorSetsFn = *const fn (VkDevice, u32, ?[*]const VkWriteDescriptorSet, u32, ?*anyopaque) callconv(.c) void;
pub const VkCreateFenceFn = *const fn (VkDevice, *const VkFenceCreateInfo, ?*anyopaque, *VkFence) callconv(.c) VkResult;
pub const VkDestroyFenceFn = *const fn (VkDevice, VkFence, ?*anyopaque) callconv(.c) void;
pub const VkResetFencesFn = *const fn (VkDevice, u32, [*]const VkFence) callconv(.c) VkResult;
pub const VkWaitForFencesFn = *const fn (VkDevice, u32, [*]const VkFence, u32, u64) callconv(.c) VkResult;
pub const VkQueueSubmitFn = *const fn (VkQueue, u32, [*]const VkSubmitInfo, VkFence) callconv(.c) VkResult;
pub const VkQueueWaitIdleFn = *const fn (VkQueue) callconv(.c) VkResult;
pub const VkCreatePipelineCacheFn = *const fn (VkDevice, *const VkPipelineCacheCreateInfo, ?*anyopaque, *VkPipelineCache) callconv(.c) VkResult;
pub const VkDestroyPipelineCacheFn = *const fn (VkDevice, VkPipelineCache, ?*anyopaque) callconv(.c) void;
pub const VkGetPipelineCacheDataFn = *const fn (VkDevice, VkPipelineCache, *usize, ?*anyopaque) callconv(.c) VkResult;
pub const VkMergePipelineCachesFn = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkPipelineCache) callconv(.c) VkResult;
pub const VkResetCommandBufferFn = *const fn (VkCommandBuffer, u32) callconv(.c) VkResult;
pub const VkResetCommandPoolFn = *const fn (VkDevice, VkCommandPool, u32) callconv(.c) VkResult;
pub const VkEnumerateInstanceLayerPropertiesFn = *const fn (*u32, ?[*]VkLayerProperties) callconv(.c) VkResult;
pub const VkCmdPushConstantsFn = *const fn (VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, u32, u32, *const anyopaque) callconv(.c) void;

// Queue family flag bits
pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: u32 = 0x00000004;
pub const VK_QUEUE_SPARSE_BINDING_BIT: u32 = 0x00000008;

// Memory property flag bits
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: u32 = 0x00000008;

// Buffer usage flag bits
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: u32 = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;

// Shader stage flag bits
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;

// ============================================================================
// Context and Resource Structures
// ============================================================================

pub const VulkanContext = struct {
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    compute_queue: VkQueue,
    compute_queue_family_index: u32,
    command_pool: VkCommandPool,
    allocator: std.mem.Allocator,
    memory_properties: VkPhysicalDeviceMemoryProperties = undefined,
    validation_enabled: bool = false,
};

pub const VulkanKernel = struct {
    shader_module: VkShaderModule,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
    descriptor_set_layout: VkDescriptorSetLayout,
    descriptor_pool: VkDescriptorPool,
    binding_count: u32 = 1,
    push_constant_size: u32 = 0,
};

pub const VulkanBuffer = struct {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    size: VkDeviceSize,
    mapped_ptr: ?*anyopaque,
};

pub const KernelSourceFormat = enum {
    spirv,
    glsl,
};

pub const KernelSource = struct {
    code: []const u8,
    entry_point: [:0]const u8,
    format: KernelSourceFormat = .spirv,
};

pub const KernelConfig = struct {
    grid_size: [3]u32 = .{ 1, 1, 1 },
    block_size: [3]u32 = .{ 256, 1, 1 },
    shared_memory: u32 = 0,
};

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

    const physical_device = p_devices[0]; // Just pick the first one for now

    // Find Compute Queue Family
    var queue_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &queue_count, null);

    // Note: VkQueueFamilyProperties is opaque-ish in types above, simplifying for now
    // In a real impl we'd iterate. Assuming index 0 has compute for simplicity or fallback
    const queue_family_index: u32 = 0;

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
            // shader_module is usually destroyed after pipeline creation
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

        // Return dummy caps for now or query actual device props
        return interface.DeviceCaps{
            .name = "Vulkan Device",
            .name_len = 13,
            .total_memory = 0, // Should query
            .max_threads_per_block = 1024,
            .max_shared_memory = 32768,
            .warp_size = 32,
            .supports_fp16 = true,
            .supports_fp64 = false,
            .supports_int8 = true,
            .unified_memory = false,
        };
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
        defer vkDestroyShaderModule.?(ctx.device, shader_module, null);

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
            .shader_module = shader_module, // Actually destroyed already, just placeholder in struct
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

        // Destroy Vulkan resources
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
        // shader_module is usually destroyed after pipeline creation

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
