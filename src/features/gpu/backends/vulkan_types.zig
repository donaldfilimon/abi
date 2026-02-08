//! Vulkan type definitions, error types, handles, extern structs, and function pointers.
//!
//! Pure type definitions with no runtime code. Extracted from vulkan.zig
//! for better code organization.

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
