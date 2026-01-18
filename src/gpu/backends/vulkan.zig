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

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../device.zig").Device;
const DeviceType = @import("../device.zig").DeviceType;
const Backend = @import("../backend.zig").Backend;

/// Enumerate all Vulkan physical devices available on the system.
///
/// Returns a slice of Device structs for each Vulkan-capable GPU.
/// **Caller owns the returned memory** and must free it with `allocator.free(devices)`.
///
/// This function queries the Vulkan runtime for all physical devices and returns
/// their properties including device type, memory info, and capabilities.
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!isAvailable()) {
        return &[_]Device{};
    }

    // Ensure Vulkan is initialized for function pointers
    if (!vulkan_init.vulkan_initialized) {
        vulkan_init.init() catch return &[_]Device{};
    }

    // Get the instance from context (or create temporary one)
    const ctx = vulkan_init.vulkan_context orelse return &[_]Device{};

    // Get enumerate function
    const enumerate_fn = vulkan_init.vkEnumeratePhysicalDevices orelse return &[_]Device{};
    const get_props_fn = vulkan_init.vkGetPhysicalDeviceProperties orelse return &[_]Device{};
    const get_mem_props_fn = vulkan_init.vkGetPhysicalDeviceMemoryProperties orelse null;

    // Query device count
    var device_count: u32 = 0;
    var result = enumerate_fn(ctx.instance, &device_count, null);
    if (result != .success or device_count == 0) {
        return &[_]Device{};
    }

    // Allocate temporary storage for physical devices
    const physical_devices = try allocator.alloc(vulkan_types.VkPhysicalDevice, device_count);
    defer allocator.free(physical_devices);

    result = enumerate_fn(ctx.instance, &device_count, physical_devices.ptr);
    if (result != .success) {
        return &[_]Device{};
    }

    // Build device list
    var devices = std.ArrayList(Device).init(allocator);
    errdefer devices.deinit();

    for (physical_devices[0..device_count], 0..) |phys_dev, i| {
        var properties: vulkan_types.VkPhysicalDeviceProperties = undefined;
        get_props_fn(phys_dev, @ptrCast(&properties));

        // Get memory properties if available
        var total_memory: ?u64 = null;
        if (get_mem_props_fn) |mem_fn| {
            var mem_props: vulkan_types.VkPhysicalDeviceMemoryProperties = undefined;
            mem_fn(phys_dev, @ptrCast(&mem_props));

            // Sum up device-local heap sizes
            var local_memory: u64 = 0;
            for (mem_props.memoryHeaps[0..mem_props.memoryHeapCount]) |heap| {
                // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001
                if ((heap.flags & 1) != 0) {
                    local_memory += heap.size;
                }
            }
            if (local_memory > 0) {
                total_memory = local_memory;
            }
        }

        // Convert device type
        const device_type: DeviceType = switch (properties.deviceType) {
            .discrete_gpu => .discrete,
            .integrated_gpu => .integrated,
            .virtual_gpu => .virtual,
            .cpu => .cpu,
            else => .other,
        };

        // Extract and duplicate null-terminated device name (properties is on stack)
        // Always allocate to ensure consistent memory ownership for cleanup
        const name_slice = std.mem.sliceTo(&properties.deviceName, 0);
        const name: []const u8 = if (name_slice.len > 0)
            try allocator.dupe(u8, name_slice)
        else
            try allocator.dupe(u8, "Vulkan Device");

        errdefer allocator.free(name);

        try devices.append(.{
            .id = @intCast(i),
            .backend = .vulkan,
            .name = name,
            .device_type = device_type,
            .total_memory = total_memory,
            .available_memory = null, // Not tracked at enumeration time
            .is_emulated = device_type == .virtual,
            .capability = .{
                .supports_fp16 = true, // Most Vulkan devices support fp16
                .supports_fp64 = false, // Conservative; would need feature query
                .supports_int8 = true,
                .supports_async_transfers = true,
                .unified_memory = device_type == .integrated_gpu,
                .max_threads_per_block = properties.limits.maxComputeWorkGroupInvocations,
                .max_shared_memory_bytes = properties.limits.maxComputeSharedMemorySize,
            },
            .compute_units = null, // Would need VK_KHR_maintenance3 or similar
            .clock_mhz = null, // Not exposed by Vulkan directly
        });
    }

    return devices.toOwnedSlice();
}

/// Check if Vulkan is available on this system
pub fn isAvailable() bool {
    return vulkan_init.isVulkanAvailable();
}
