//! Vulkan backend initialization and context management.
//!
//! Handles Vulkan library loading, function resolution, instance/device creation,
//! and overall backend lifecycle management.

const std = @import("std");
const types = @import("vulkan_types.zig");

// Re-export types for convenience
pub const VulkanError = types.VulkanError;
pub const VulkanContext = types.VulkanContext;

// Global state
pub var vulkan_lib: ?std.DynLib = null;
pub var vulkan_initialized = false;
pub var vulkan_context: ?VulkanContext = null;

// Vulkan function pointers
pub var vkCreateInstance: ?types.VkCreateInstanceFn = null;
pub var vkDestroyInstance: ?types.VkDestroyInstanceFn = null;
pub var vkEnumeratePhysicalDevices: ?types.VkEnumeratePhysicalDevicesFn = null;
pub var vkGetPhysicalDeviceProperties: ?types.VkGetPhysicalDevicePropertiesFn = null;
pub var vkGetPhysicalDeviceQueueFamilyProperties: ?types.VkGetPhysicalDeviceQueueFamilyPropertiesFn = null;
pub var vkGetPhysicalDeviceMemoryProperties: ?types.VkGetPhysicalDeviceMemoryPropertiesFn = null;
pub var vkCreateDevice: ?types.VkCreateDeviceFn = null;
pub var vkDestroyDevice: ?types.VkDestroyDeviceFn = null;
pub var vkGetDeviceQueue: ?types.VkGetDeviceQueueFn = null;
pub var vkCreateBuffer: ?types.VkCreateBufferFn = null;
pub var vkDestroyBuffer: ?types.VkDestroyBufferFn = null;
pub var vkGetBufferMemoryRequirements: ?types.VkGetBufferMemoryRequirementsFn = null;
pub var vkAllocateMemory: ?types.VkAllocateMemoryFn = null;
pub var vkFreeMemory: ?types.VkFreeMemoryFn = null;
pub var vkBindBufferMemory: ?types.VkBindBufferMemoryFn = null;
pub var vkMapMemory: ?types.VkMapMemoryFn = null;
pub var vkUnmapMemory: ?types.VkUnmapMemoryFn = null;
pub var vkCreateShaderModule: ?types.VkCreateShaderModuleFn = null;
pub var vkDestroyShaderModule: ?types.VkDestroyShaderModuleFn = null;
pub var vkCreateDescriptorSetLayout: ?types.VkCreateDescriptorSetLayoutFn = null;
pub var vkDestroyDescriptorSetLayout: ?types.VkDestroyDescriptorSetLayoutFn = null;
pub var vkCreatePipelineLayout: ?types.VkCreatePipelineLayoutFn = null;
pub var vkDestroyPipelineLayout: ?types.VkDestroyPipelineLayoutFn = null;
pub var vkCreateComputePipelines: ?types.VkCreateComputePipelinesFn = null;
pub var vkDestroyPipeline: ?types.VkDestroyPipelineFn = null;
pub var vkCreateCommandPool: ?types.VkCreateCommandPoolFn = null;
pub var vkDestroyCommandPool: ?types.VkDestroyCommandPoolFn = null;
pub var vkAllocateCommandBuffers: ?types.VkAllocateCommandBuffersFn = null;
pub var vkFreeCommandBuffers: ?types.VkFreeCommandBuffersFn = null;
pub var vkBeginCommandBuffer: ?types.VkBeginCommandBufferFn = null;
pub var vkEndCommandBuffer: ?types.VkEndCommandBufferFn = null;
pub var vkCmdBindPipeline: ?types.VkCmdBindPipelineFn = null;
pub var vkCmdBindDescriptorSets: ?types.VkCmdBindDescriptorSetsFn = null;
pub var vkCmdDispatch: ?types.VkCmdDispatchFn = null;
pub var vkCmdPipelineBarrier: ?types.VkCmdPipelineBarrierFn = null;
pub var vkCreateDescriptorPool: ?types.VkCreateDescriptorPoolFn = null;
pub var vkDestroyDescriptorPool: ?types.VkDestroyDescriptorPoolFn = null;
pub var vkAllocateDescriptorSets: ?types.VkAllocateDescriptorSetsFn = null;
pub var vkFreeDescriptorSets: ?types.VkFreeDescriptorSetsFn = null;
pub var vkUpdateDescriptorSets: ?types.VkUpdateDescriptorSetsFn = null;
pub var vkCreateFence: ?types.VkCreateFenceFn = null;
pub var vkDestroyFence: ?types.VkDestroyFenceFn = null;
pub var vkResetFences: ?types.VkResetFencesFn = null;
pub var vkWaitForFences: ?types.VkWaitForFencesFn = null;
pub var vkQueueSubmit: ?types.VkQueueSubmitFn = null;
pub var vkQueueWaitIdle: ?types.VkQueueWaitIdleFn = null;

/// Initialize the Vulkan backend and create necessary resources.
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

    const app_info = types.VkApplicationInfo{
        .pApplicationName = "ABI Compute",
        .apiVersion = 0x00402000, // Vulkan 1.2.0
    };

    const create_info = types.VkInstanceCreateInfo{
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    var instance: types.VkInstance = undefined;
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
    const queue_create_info = types.VkDeviceQueueCreateInfo{
        .queueFamilyIndex = queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const device_create_info = types.VkDeviceCreateInfo{
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    const create_device_fn = vkCreateDevice orelse return VulkanError.DeviceCreationFailed;
    var device: types.VkDevice = undefined;
    const device_result = create_device_fn(physical_device, &device_create_info, null, &device);
    if (device_result != .success) {
        return VulkanError.LogicalDeviceCreationFailed;
    }

    errdefer if (vkDestroyDevice) |destroy_fn| destroy_fn(device, null);

    // Get compute queue
    const get_queue_fn = vkGetDeviceQueue orelse return VulkanError.InitializationFailed;
    var compute_queue: types.VkQueue = undefined;
    get_queue_fn(device, queue_family_index, 0, &compute_queue);

    // Create command pool
    const command_pool_create_info = types.VkCommandPoolCreateInfo{
        .queueFamilyIndex = queue_family_index,
    };

    const create_command_pool_fn = vkCreateCommandPool orelse return VulkanError.CommandBufferAllocationFailed;
    var command_pool: types.VkCommandPool = undefined;
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

/// Device scoring for selection priority.
/// Higher scores are preferred. Discrete GPUs score highest.
const DeviceScore = struct {
    device: types.VkPhysicalDevice,
    score: u32,
};

fn selectPhysicalDevice(instance: types.VkInstance) !types.VkPhysicalDevice {
    const enumerate_fn = vkEnumeratePhysicalDevices orelse return VulkanError.PhysicalDeviceNotFound;

    var device_count: u32 = 0;
    var result = enumerate_fn(instance, &device_count, null);
    if (result != .success or device_count == 0) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    // Use a temporary arena allocator for device enumeration
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    const devices = try temp_allocator.alloc(types.VkPhysicalDevice, device_count);

    result = enumerate_fn(instance, &device_count, devices.ptr);
    if (result != .success) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    // Score each device and select the best one
    var best_device: ?types.VkPhysicalDevice = null;
    var best_score: u32 = 0;

    const get_properties_fn = vkGetPhysicalDeviceProperties orelse {
        // If we can't get properties, just return the first device
        return devices[0];
    };

    for (devices) |device| {
        var properties: types.VkPhysicalDeviceProperties = undefined;
        get_properties_fn(device, &properties);

        const score = scorePhysicalDevice(&properties);
        if (best_device == null or score > best_score) {
            best_device = device;
            best_score = score;
        }
    }

    return best_device orelse VulkanError.PhysicalDeviceNotFound;
}

/// Score a physical device based on its type and capabilities.
/// Scoring hierarchy (from CLAUDE.md):
/// - Discrete GPU > Integrated GPU > Virtual GPU > CPU > Other
fn scorePhysicalDevice(properties: *const types.VkPhysicalDeviceProperties) u32 {
    const device_type = properties.deviceType;

    // Base score by device type
    const type_score: u32 = switch (device_type) {
        .discrete_gpu => 1000,
        .integrated_gpu => 500,
        .virtual_gpu => 100,
        .cpu => 50,
        else => 10,
    };

    // Bonus for API version (prefer newer Vulkan versions)
    const api_version = properties.apiVersion;
    const api_bonus: u32 = @min(api_version / 0x00100000, 10); // Cap at version 10

    return type_score + api_bonus;
}

fn findComputeQueueFamily(physical_device: types.VkPhysicalDevice) !u32 {
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

    vkCreateInstance = vulkan_lib.?.lookup(types.VkCreateInstanceFn, "vkCreateInstance") orelse return false;
    vkDestroyInstance = vulkan_lib.?.lookup(types.VkDestroyInstanceFn, "vkDestroyInstance") orelse return false;
    vkEnumeratePhysicalDevices = vulkan_lib.?.lookup(types.VkEnumeratePhysicalDevicesFn, "vkEnumeratePhysicalDevices") orelse return false;
    vkGetPhysicalDeviceProperties = vulkan_lib.?.lookup(types.VkGetPhysicalDevicePropertiesFn, "vkGetPhysicalDeviceProperties") orelse return false;
    vkGetPhysicalDeviceQueueFamilyProperties = vulkan_lib.?.lookup(types.VkGetPhysicalDeviceQueueFamilyPropertiesFn, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return false;
    vkGetPhysicalDeviceMemoryProperties = vulkan_lib.?.lookup(types.VkGetPhysicalDeviceMemoryPropertiesFn, "vkGetPhysicalDeviceMemoryProperties") orelse return false;
    vkCreateDevice = vulkan_lib.?.lookup(types.VkCreateDeviceFn, "vkCreateDevice") orelse return false;
    vkDestroyDevice = vulkan_lib.?.lookup(types.VkDestroyDeviceFn, "vkDestroyDevice") orelse return false;
    vkGetDeviceQueue = vulkan_lib.?.lookup(types.VkGetDeviceQueueFn, "vkGetDeviceQueue") orelse return false;
    vkCreateBuffer = vulkan_lib.?.lookup(types.VkCreateBufferFn, "vkCreateBuffer") orelse return false;
    vkDestroyBuffer = vulkan_lib.?.lookup(types.VkDestroyBufferFn, "vkDestroyBuffer") orelse return false;
    vkGetBufferMemoryRequirements = vulkan_lib.?.lookup(types.VkGetBufferMemoryRequirementsFn, "vkGetBufferMemoryRequirements") orelse return false;
    vkAllocateMemory = vulkan_lib.?.lookup(types.VkAllocateMemoryFn, "vkAllocateMemory") orelse return false;
    vkFreeMemory = vulkan_lib.?.lookup(types.VkFreeMemoryFn, "vkFreeMemory") orelse return false;
    vkBindBufferMemory = vulkan_lib.?.lookup(types.VkBindBufferMemoryFn, "vkBindBufferMemory") orelse return false;
    vkMapMemory = vulkan_lib.?.lookup(types.VkMapMemoryFn, "vkMapMemory") orelse return false;
    vkUnmapMemory = vulkan_lib.?.lookup(types.VkUnmapMemoryFn, "vkUnmapMemory") orelse return false;
    vkCreateShaderModule = vulkan_lib.?.lookup(types.VkCreateShaderModuleFn, "vkCreateShaderModule") orelse return false;
    vkDestroyShaderModule = vulkan_lib.?.lookup(types.VkDestroyShaderModuleFn, "vkDestroyShaderModule") orelse return false;
    vkCreateDescriptorSetLayout = vulkan_lib.?.lookup(types.VkCreateDescriptorSetLayoutFn, "vkCreateDescriptorSetLayout") orelse return false;
    vkDestroyDescriptorSetLayout = vulkan_lib.?.lookup(types.VkDestroyDescriptorSetLayoutFn, "vkDestroyDescriptorSetLayout") orelse return false;
    vkCreatePipelineLayout = vulkan_lib.?.lookup(types.VkCreatePipelineLayoutFn, "vkCreatePipelineLayout") orelse return false;
    vkDestroyPipelineLayout = vulkan_lib.?.lookup(types.VkDestroyPipelineLayoutFn, "vkDestroyPipelineLayout") orelse return false;
    vkCreateComputePipelines = vulkan_lib.?.lookup(types.VkCreateComputePipelinesFn, "vkCreateComputePipelines") orelse return false;
    vkDestroyPipeline = vulkan_lib.?.lookup(types.VkDestroyPipelineFn, "vkDestroyPipeline") orelse return false;
    vkCreateCommandPool = vulkan_lib.?.lookup(types.VkCreateCommandPoolFn, "vkCreateCommandPool") orelse return false;
    vkDestroyCommandPool = vulkan_lib.?.lookup(types.VkDestroyCommandPoolFn, "vkDestroyCommandPool") orelse return false;
    vkAllocateCommandBuffers = vulkan_lib.?.lookup(types.VkAllocateCommandBuffersFn, "vkAllocateCommandBuffers") orelse return false;
    vkFreeCommandBuffers = vulkan_lib.?.lookup(types.VkFreeCommandBuffersFn, "vkFreeCommandBuffers") orelse return false;
    vkBeginCommandBuffer = vulkan_lib.?.lookup(types.VkBeginCommandBufferFn, "vkBeginCommandBuffer") orelse return false;
    vkEndCommandBuffer = vulkan_lib.?.lookup(types.VkEndCommandBufferFn, "vkEndCommandBuffer") orelse return false;
    vkCmdBindPipeline = vulkan_lib.?.lookup(types.VkCmdBindPipelineFn, "vkCmdBindPipeline") orelse return false;
    vkCmdBindDescriptorSets = vulkan_lib.?.lookup(types.VkCmdBindDescriptorSetsFn, "vkCmdBindDescriptorSets") orelse return false;
    vkCmdDispatch = vulkan_lib.?.lookup(types.VkCmdDispatchFn, "vkCmdDispatch") orelse return false;
    vkCmdPipelineBarrier = vulkan_lib.?.lookup(types.VkCmdPipelineBarrierFn, "vkCmdPipelineBarrier") orelse return false;
    vkCreateDescriptorPool = vulkan_lib.?.lookup(types.VkCreateDescriptorPoolFn, "vkCreateDescriptorPool") orelse return false;
    vkDestroyDescriptorPool = vulkan_lib.?.lookup(types.VkDestroyDescriptorPoolFn, "vkDestroyDescriptorPool") orelse return false;
    vkAllocateDescriptorSets = vulkan_lib.?.lookup(types.VkAllocateDescriptorSetsFn, "vkAllocateDescriptorSets") orelse return false;
    vkFreeDescriptorSets = vulkan_lib.?.lookup(types.VkFreeDescriptorSetsFn, "vkFreeDescriptorSets") orelse return false;
    vkUpdateDescriptorSets = vulkan_lib.?.lookup(types.VkUpdateDescriptorSetsFn, "vkUpdateDescriptorSets") orelse return false;
    vkCreateFence = vulkan_lib.?.lookup(types.VkCreateFenceFn, "vkCreateFence") orelse return false;
    vkDestroyFence = vulkan_lib.?.lookup(types.VkDestroyFenceFn, "vkDestroyFence") orelse return false;
    vkResetFences = vulkan_lib.?.lookup(types.VkResetFencesFn, "vkResetFences") orelse return false;
    vkWaitForFences = vulkan_lib.?.lookup(types.VkWaitForFencesFn, "vkWaitForFences") orelse return false;
    vkQueueSubmit = vulkan_lib.?.lookup(types.VkQueueSubmitFn, "vkQueueSubmit") orelse return false;
    vkQueueWaitIdle = vulkan_lib.?.lookup(types.VkQueueWaitIdleFn, "vkQueueWaitIdle") orelse return false;

    return true;
}

pub fn tryLoadVulkan() bool {
    const lib_names = [_][]const u8{ "vulkan-1.dll", "libvulkan.so.1", "libvulkan.dylib" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            vulkan_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}
