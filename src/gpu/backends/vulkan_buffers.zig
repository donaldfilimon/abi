//! Vulkan memory allocation and buffer management.
//!
//! Handles device memory allocation, host-to-device and device-to-host
//! memory transfers for compute operations.

const std = @import("std");
const types = @import("vulkan_types.zig");
const init = @import("vulkan_init.zig");

pub const VulkanError = types.VulkanError;

/// Memory allocation flags for buffer creation.
pub const MemoryLocation = enum {
    /// GPU-local memory, fastest for GPU operations but not directly accessible from CPU
    device_local,
    /// Host-visible memory, can be mapped and accessed from CPU
    host_visible,
    /// Host-visible and coherent memory, no explicit flush required
    host_coherent,
};

/// Allocate device memory on the GPU with specified memory location.
pub fn allocateDeviceMemoryEx(size: usize, location: MemoryLocation) !*anyopaque {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return VulkanError.InitializationFailed;
    }

    const ctx = &init.vulkan_context.?;

    // Determine buffer usage flags based on memory location
    const usage_flags: u32 = switch (location) {
        .device_local => types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | types.VK_BUFFER_USAGE_TRANSFER_DST_BIT | types.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .host_visible, .host_coherent => types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | types.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | types.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };

    // Create buffer
    const buffer_create_info = types.VkBufferCreateInfo{
        .size = @intCast(size),
        .usage = usage_flags,
        .sharingMode = 0, // VK_SHARING_MODE_EXCLUSIVE
    };

    const create_buffer_fn = init.vkCreateBuffer orelse return VulkanError.BufferCreationFailed;
    var buffer: types.VkBuffer = undefined;
    const buffer_result = create_buffer_fn(ctx.device, &buffer_create_info, null, &buffer);
    if (buffer_result != .success) {
        return VulkanError.BufferCreationFailed;
    }

    errdefer if (init.vkDestroyBuffer) |destroy_fn| destroy_fn(ctx.device, buffer, null);

    // Get memory requirements
    var mem_requirements: types.VkMemoryRequirements = undefined;
    const get_req_fn = init.vkGetBufferMemoryRequirements orelse return VulkanError.MemoryAllocationFailed;
    get_req_fn(ctx.device, buffer, &mem_requirements);

    // Find appropriate memory type based on location preference
    const memory_properties: u32 = switch (location) {
        .device_local => types.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .host_visible => types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        .host_coherent => types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };

    const mem_type_index = try init.findMemoryType(mem_requirements.memoryTypeBits, memory_properties);

    const alloc_info = types.VkMemoryAllocateInfo{
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = mem_type_index,
    };

    const allocate_fn = init.vkAllocateMemory orelse return VulkanError.MemoryAllocationFailed;
    var device_memory: types.VkDeviceMemory = undefined;
    const alloc_result = allocate_fn(ctx.device, &alloc_info, null, &device_memory);
    if (alloc_result != .success) {
        return VulkanError.MemoryAllocationFailed;
    }

    errdefer if (init.vkFreeMemory) |free_fn| free_fn(ctx.device, device_memory, null);

    // Bind memory
    const bind_fn = init.vkBindBufferMemory orelse return VulkanError.MemoryAllocationFailed;
    const bind_result = bind_fn(ctx.device, buffer, device_memory, 0);
    if (bind_result != .success) {
        return VulkanError.MemoryAllocationFailed;
    }

    const vulkan_buffer = try std.heap.page_allocator.create(types.VulkanBuffer);
    vulkan_buffer.* = .{
        .buffer = buffer,
        .memory = device_memory,
        .size = mem_requirements.size,
        .mapped_ptr = null,
    };

    return vulkan_buffer;
}

/// Allocate device memory on the GPU (device-local, default).
pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    // Try device-local first for best GPU performance
    // If that fails, fall back to host-visible memory (for integrated GPUs)
    return allocateDeviceMemoryEx(size, .device_local) catch |err| {
        if (err == VulkanError.MemoryTypeNotFound) {
            return allocateDeviceMemoryEx(size, .host_coherent);
        }
        return err;
    };
}

/// Allocate staging buffer for host-to-device transfers.
pub fn allocateStagingBuffer(size: usize) !*anyopaque {
    return allocateDeviceMemoryEx(size, .host_coherent);
}

/// Free device memory allocated by allocateDeviceMemory.
pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return;
    }

    const ctx = &init.vulkan_context.?;
    const buffer: *types.VulkanBuffer = @ptrCast(@alignCast(ptr));

    if (buffer.mapped_ptr != null) {
        if (init.vkUnmapMemory) |unmap_fn| unmap_fn(ctx.device, buffer.memory);
    }
    if (init.vkFreeMemory) |free_fn| free_fn(ctx.device, buffer.memory, null);
    if (init.vkDestroyBuffer) |destroy_fn| destroy_fn(ctx.device, buffer.buffer, null);

    std.heap.page_allocator.destroy(buffer);
}

/// Copy data from host memory to device memory.
pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return VulkanError.MemoryCopyFailed;
    }

    const ctx = &init.vulkan_context.?;
    const dst_buffer: *types.VulkanBuffer = @ptrCast(@alignCast(dst));

    // Map memory
    var mapped_ptr: ?*anyopaque = null;
    const map_fn = init.vkMapMemory orelse return VulkanError.MemoryCopyFailed;
    const map_result = map_fn(ctx.device, dst_buffer.memory, 0, dst_buffer.size, 0, &mapped_ptr);
    if (map_result != .success or mapped_ptr == null) {
        return VulkanError.MemoryCopyFailed;
    }

    // Copy data
    @memcpy(@as([*]u8, @ptrCast(mapped_ptr.?))[0..size], @as([*]const u8, @ptrCast(src))[0..size]);

    // Unmap memory
    const unmap_fn = init.vkUnmapMemory orelse return VulkanError.MemoryCopyFailed;
    unmap_fn(ctx.device, dst_buffer.memory);

    dst_buffer.mapped_ptr = null;
}

/// Copy data from device memory to host memory.
pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return VulkanError.MemoryCopyFailed;
    }

    const ctx = &init.vulkan_context.?;
    const src_buffer: *types.VulkanBuffer = @ptrCast(@alignCast(src));

    // Map memory
    var mapped_ptr: ?*anyopaque = null;
    const map_fn = init.vkMapMemory orelse return VulkanError.MemoryCopyFailed;
    const map_result = map_fn(ctx.device, src_buffer.memory, 0, src_buffer.size, 0, &mapped_ptr);
    if (map_result != .success or mapped_ptr == null) {
        return VulkanError.MemoryCopyFailed;
    }

    // Copy data
    @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(mapped_ptr.?))[0..size]);

    // Unmap memory
    const unmap_fn = init.vkUnmapMemory orelse return VulkanError.MemoryCopyFailed;
    unmap_fn(ctx.device, src_buffer.memory);

    src_buffer.mapped_ptr = null;
}
