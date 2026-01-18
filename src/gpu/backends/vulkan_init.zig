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

// Pipeline cache functions
pub var vkCreatePipelineCache: ?types.VkCreatePipelineCacheFn = null;
pub var vkDestroyPipelineCache: ?types.VkDestroyPipelineCacheFn = null;
pub var vkGetPipelineCacheData: ?types.VkGetPipelineCacheDataFn = null;
pub var vkMergePipelineCaches: ?types.VkMergePipelineCachesFn = null;

// Command buffer reset functions
pub var vkResetCommandBuffer: ?types.VkResetCommandBufferFn = null;
pub var vkResetCommandPool: ?types.VkResetCommandPoolFn = null;

// Validation layer functions
pub var vkEnumerateInstanceLayerProperties: ?types.VkEnumerateInstanceLayerPropertiesFn = null;

// Push constants
pub var vkCmdPushConstants: ?types.VkCmdPushConstantsFn = null;

/// Configuration for Vulkan initialization
pub const VulkanInitConfig = struct {
    /// Enable validation layers for debugging
    enable_validation: bool = false,
    /// Preferred device type (null = auto-select best)
    preferred_device_type: ?types.VkPhysicalDeviceType = null,
    /// Application name for Vulkan instance
    app_name: [:0]const u8 = "ABI Compute",
};

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

    // Cache memory properties for efficient memory type selection
    var memory_properties: types.VkPhysicalDeviceMemoryProperties = undefined;
    if (vkGetPhysicalDeviceMemoryProperties) |get_mem_props_fn| {
        get_mem_props_fn(physical_device, &memory_properties);
    }

    return VulkanContext{
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .compute_queue = compute_queue,
        .compute_queue_family_index = queue_family_index,
        .command_pool = command_pool,
        .allocator = allocator,
        .memory_properties = memory_properties,
        .validation_enabled = false, // Will be set by initWithConfig
    };
}

/// Initialize the Vulkan backend with custom configuration.
pub fn initWithConfig(config: VulkanInitConfig) !void {
    if (vulkan_initialized) return;

    if (!tryLoadVulkan()) {
        return VulkanError.InitializationFailed;
    }

    if (!loadVulkanFunctions()) {
        return VulkanError.InitializationFailed;
    }

    // Create Vulkan context with config
    const context = try createVulkanContextWithConfig(std.heap.page_allocator, config);
    vulkan_context = context;

    vulkan_initialized = true;
}

fn createVulkanContextWithConfig(allocator: std.mem.Allocator, config: VulkanInitConfig) !VulkanContext {
    const create_instance_fn = vkCreateInstance orelse return VulkanError.InitializationFailed;

    // Check for validation layer support if requested
    var validation_available = false;
    var enabled_layer_names: [1][*:0]const u8 = undefined;
    var enabled_layer_count: u32 = 0;

    if (config.enable_validation) {
        if (vkEnumerateInstanceLayerProperties) |enumerate_fn| {
            var layer_count: u32 = 0;
            _ = enumerate_fn(&layer_count, null);

            if (layer_count > 0) {
                var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer arena.deinit();
                const temp_allocator = arena.allocator();

                const layers = temp_allocator.alloc(types.VkLayerProperties, layer_count) catch null;
                if (layers) |layer_props| {
                    _ = enumerate_fn(&layer_count, layer_props.ptr);

                    // Look for validation layer
                    const validation_layer_name = "VK_LAYER_KHRONOS_validation";
                    for (layer_props) |layer| {
                        const name_slice = std.mem.sliceTo(&layer.layerName, 0);
                        if (std.mem.eql(u8, name_slice, validation_layer_name)) {
                            validation_available = true;
                            enabled_layer_names[0] = validation_layer_name;
                            enabled_layer_count = 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    const app_info = types.VkApplicationInfo{
        .pApplicationName = config.app_name.ptr,
        .apiVersion = 0x00402000, // Vulkan 1.2.0
    };

    const create_info = types.VkInstanceCreateInfo{
        .pApplicationInfo = &app_info,
        .enabledLayerCount = enabled_layer_count,
        .ppEnabledLayerNames = if (enabled_layer_count > 0) &enabled_layer_names else null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    var instance: types.VkInstance = undefined;
    const result = create_instance_fn(&create_info, null, &instance);
    if (result != .success) {
        return VulkanError.InstanceCreationFailed;
    }

    errdefer if (vkDestroyInstance) |destroy_fn| destroy_fn(instance, null);

    // Select physical device with optional preference
    const physical_device = try selectPhysicalDeviceWithPreference(instance, config.preferred_device_type);
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

    // Create command pool with reset capability
    const command_pool_create_info = types.VkCommandPoolCreateInfo{
        .flags = 0x2, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        .queueFamilyIndex = queue_family_index,
    };

    const create_command_pool_fn = vkCreateCommandPool orelse return VulkanError.CommandBufferAllocationFailed;
    var command_pool: types.VkCommandPool = undefined;
    const pool_result = create_command_pool_fn(device, @ptrCast(@constCast(&command_pool_create_info)), null, &command_pool);
    if (pool_result != .success) {
        return VulkanError.CommandBufferAllocationFailed;
    }

    // Cache memory properties
    var memory_properties: types.VkPhysicalDeviceMemoryProperties = undefined;
    if (vkGetPhysicalDeviceMemoryProperties) |get_mem_props_fn| {
        get_mem_props_fn(physical_device, &memory_properties);
    }

    return VulkanContext{
        .instance = instance,
        .physical_device = physical_device,
        .device = device,
        .compute_queue = compute_queue,
        .compute_queue_family_index = queue_family_index,
        .command_pool = command_pool,
        .allocator = allocator,
        .memory_properties = memory_properties,
        .validation_enabled = validation_available and config.enable_validation,
    };
}

fn selectPhysicalDeviceWithPreference(instance: types.VkInstance, preferred_type: ?types.VkPhysicalDeviceType) !types.VkPhysicalDevice {
    const enumerate_fn = vkEnumeratePhysicalDevices orelse return VulkanError.PhysicalDeviceNotFound;

    var device_count: u32 = 0;
    var result = enumerate_fn(instance, &device_count, null);
    if (result != .success or device_count == 0) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    const devices = try temp_allocator.alloc(types.VkPhysicalDevice, device_count);

    result = enumerate_fn(instance, &device_count, devices.ptr);
    if (result != .success) {
        return VulkanError.PhysicalDeviceNotFound;
    }

    const get_properties_fn = vkGetPhysicalDeviceProperties orelse {
        return devices[0];
    };

    // If we have a preference, try to find a matching device first
    if (preferred_type) |ptype| {
        for (devices) |device| {
            var properties: types.VkPhysicalDeviceProperties = undefined;
            get_properties_fn(device, &properties);
            if (properties.deviceType == ptype) {
                return device;
            }
        }
    }

    // Fall back to scoring-based selection
    var best_device: ?types.VkPhysicalDevice = null;
    var best_score: u32 = 0;

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

    // Use a temporary arena allocator for queue family properties
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    const queue_families = temp_allocator.alloc(types.VkQueueFamilyProperties, queue_family_count) catch {
        return VulkanError.QueueFamilyNotFound;
    };

    get_properties_fn(physical_device, &queue_family_count, @ptrCast(queue_families.ptr));

    // First, try to find a dedicated compute queue (compute but not graphics)
    // This is optimal for compute workloads as it avoids contention with graphics
    for (queue_families, 0..) |family, i| {
        const has_compute = (family.queueFlags & types.VK_QUEUE_COMPUTE_BIT) != 0;
        const has_graphics = (family.queueFlags & types.VK_QUEUE_GRAPHICS_BIT) != 0;

        if (has_compute and !has_graphics and family.queueCount > 0) {
            return @intCast(i);
        }
    }

    // If no dedicated compute queue, find any queue with compute support
    for (queue_families, 0..) |family, i| {
        const has_compute = (family.queueFlags & types.VK_QUEUE_COMPUTE_BIT) != 0;

        if (has_compute and family.queueCount > 0) {
            return @intCast(i);
        }
    }

    return VulkanError.QueueFamilyNotFound;
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

    // Pipeline cache functions (optional - not critical for basic operation)
    vkCreatePipelineCache = vulkan_lib.?.lookup(types.VkCreatePipelineCacheFn, "vkCreatePipelineCache");
    vkDestroyPipelineCache = vulkan_lib.?.lookup(types.VkDestroyPipelineCacheFn, "vkDestroyPipelineCache");
    vkGetPipelineCacheData = vulkan_lib.?.lookup(types.VkGetPipelineCacheDataFn, "vkGetPipelineCacheData");
    vkMergePipelineCaches = vulkan_lib.?.lookup(types.VkMergePipelineCachesFn, "vkMergePipelineCaches");

    // Command buffer reset functions (optional - not critical for basic operation)
    vkResetCommandBuffer = vulkan_lib.?.lookup(types.VkResetCommandBufferFn, "vkResetCommandBuffer");
    vkResetCommandPool = vulkan_lib.?.lookup(types.VkResetCommandPoolFn, "vkResetCommandPool");

    // Validation layer functions (optional)
    vkEnumerateInstanceLayerProperties = vulkan_lib.?.lookup(types.VkEnumerateInstanceLayerPropertiesFn, "vkEnumerateInstanceLayerProperties");

    // Push constants (optional - not critical for basic operation)
    vkCmdPushConstants = vulkan_lib.?.lookup(types.VkCmdPushConstantsFn, "vkCmdPushConstants");

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

/// Get the number of available Vulkan-capable physical devices.
pub fn getPhysicalDeviceCount() u32 {
    if (!vulkan_initialized) return 0;

    const enumerate_fn = vkEnumeratePhysicalDevices orelse return 0;
    const ctx = vulkan_context orelse return 0;

    var device_count: u32 = 0;
    const result = enumerate_fn(ctx.instance, &device_count, null);
    if (result != .success) {
        return 0;
    }

    return device_count;
}

/// Find a suitable memory type index based on requirements and desired properties.
pub fn findMemoryType(type_filter: u32, properties: u32) !u32 {
    const ctx = vulkan_context orelse return VulkanError.InitializationFailed;

    for (0..ctx.memory_properties.memoryTypeCount) |i| {
        const type_bit = @as(u32, 1) << @intCast(i);
        const has_required_type = (type_filter & type_bit) != 0;
        const has_required_props = (ctx.memory_properties.memoryTypes[i].propertyFlags & properties) == properties;

        if (has_required_type and has_required_props) {
            return @intCast(i);
        }
    }

    return VulkanError.MemoryTypeNotFound;
}

/// Find memory type suitable for device-local storage buffers.
pub fn findDeviceLocalMemoryType(type_filter: u32) !u32 {
    return findMemoryType(type_filter, types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

/// Find memory type suitable for host-visible staging buffers.
pub fn findHostVisibleMemoryType(type_filter: u32) !u32 {
    // Try to find host-visible and host-coherent memory first
    if (findMemoryType(type_filter, types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) |idx| {
        return idx;
    } else |_| {}

    // Fall back to just host-visible
    return findMemoryType(type_filter, types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}

/// Check if Vulkan is available on this system.
pub fn isVulkanAvailable() bool {
    if (vulkan_initialized) return true;

    // Try to load Vulkan library
    if (!tryLoadVulkan()) return false;

    // Check if we can enumerate devices
    if (!loadVulkanFunctions()) {
        if (vulkan_lib) |lib| {
            lib.close();
            vulkan_lib = null;
        }
        return false;
    }

    return true;
}

/// Check if validation layers are enabled
pub fn isValidationEnabled() bool {
    if (vulkan_context) |ctx| {
        return ctx.validation_enabled;
    }
    return false;
}
