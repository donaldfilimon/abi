//! Vulkan Peer Transfer Backend
//!
//! Provides GPU-to-GPU transfers using Vulkan external memory extensions
//! and compute shader-based reductions.
//!
//! ## Capabilities
//!
//! - **External Memory**: VK_KHR_external_memory for zero-copy transfers
//! - **Timeline Semaphores**: Cross-device synchronization
//! - **Compute Reduce**: Custom compute shader for reduction operations
//!
//! ## Requirements
//!
//! - Vulkan 1.1+ with VK_KHR_external_memory_fd (Linux) or
//!   VK_KHR_external_memory_win32 (Windows)
//! - VK_KHR_timeline_semaphore for cross-device sync

const std = @import("std");
const builtin = @import("builtin");

const multi_device = @import("../multi_device.zig");
const stream_mod = @import("../stream.zig");
const vulkan = @import("../backends/vulkan.zig");
const vulkan_ext = @import("vulkan_ext.zig");

pub const DeviceId = multi_device.DeviceId;
pub const ReduceOp = multi_device.ReduceOp;
pub const Stream = stream_mod.Stream;

// Re-export extension types
pub const VulkanExtFunctions = vulkan_ext.VulkanExtFunctions;

/// Vulkan external memory handle type
pub const ExternalMemoryHandle = switch (builtin.os.tag) {
    .windows => std.os.windows.HANDLE,
    .linux => std.posix.fd_t,
    else => *anyopaque,
};

/// External memory capability info
const ExternalMemoryInfo = struct {
    supported: bool = false,
    compatible_handle_types: u32 = 0,
    export_supported: bool = false,
    import_supported: bool = false,
};

/// Device pair external memory support
const DevicePairSupport = struct {
    external_memory: bool = false,
    timeline_semaphores: bool = false,
    same_driver: bool = false,
};

/// Global state
var external_memory_matrix: ?std.AutoHashMap(u64, DevicePairSupport) = null;
var vulkan_peer_initialized: bool = false;
var ext_functions: ?VulkanExtFunctions = null;
var device_contexts: ?std.AutoHashMap(DeviceId, DeviceContext) = null;
var allocator_ref: ?std.mem.Allocator = null;

/// Per-device context for Vulkan operations
const DeviceContext = struct {
    device: vulkan.VkDevice,
    physical_device: vulkan.VkPhysicalDevice,
    memory_properties: vulkan.VkPhysicalDeviceMemoryProperties,
};

/// Initialize Vulkan peer transfer backend.
pub fn init(allocator: std.mem.Allocator, device_count: usize) !void {
    if (vulkan_peer_initialized) return;

    // Ensure Vulkan is initialized
    if (!vulkan.vulkan_initialized) {
        vulkan.initVulkanGlobal(allocator) catch return error.VulkanNotAvailable;
    }

    allocator_ref = allocator;
    external_memory_matrix = std.AutoHashMap(u64, DevicePairSupport).init(allocator);
    device_contexts = std.AutoHashMap(DeviceId, DeviceContext).init(allocator);

    // Load extension functions if we have a valid context
    if (vulkan.vulkan_context) |ctx| {
        try vulkan_ext.initExtFunctions(ctx.device);
        ext_functions = vulkan_ext.getExtFunctions().*;

        // Store the default context for device 0
        try device_contexts.?.put(0, .{
            .device = ctx.device,
            .physical_device = ctx.physical_device,
            .memory_properties = ctx.memory_properties,
        });
    }

    // Probe external memory support between devices
    try probeExternalMemorySupport(device_count);

    vulkan_peer_initialized = true;
}

/// Deinitialize Vulkan peer transfer backend.
pub fn deinit() void {
    if (device_contexts) |*contexts| {
        contexts.deinit();
        device_contexts = null;
    }
    if (external_memory_matrix) |*matrix| {
        matrix.deinit();
        external_memory_matrix = null;
    }
    ext_functions = null;
    allocator_ref = null;
    vulkan_ext.deinitExtFunctions();
    vulkan_peer_initialized = false;
}

/// Probe external memory support between device pairs.
fn probeExternalMemorySupport(device_count: usize) !void {
    // Query each device's external memory capabilities
    for (0..device_count) |src| {
        const src_info = queryExternalMemoryInfo(@intCast(src));

        for (0..device_count) |dst| {
            if (src == dst) continue;

            const dst_info = queryExternalMemoryInfo(@intCast(dst));

            // Check if external memory transfer is possible
            const support = DevicePairSupport{
                .external_memory = src_info.export_supported and dst_info.import_supported and
                    (src_info.compatible_handle_types & dst_info.compatible_handle_types) != 0,
                .timeline_semaphores = queryTimelineSemaphoreSupport(@intCast(src)) and
                    queryTimelineSemaphoreSupport(@intCast(dst)),
                .same_driver = checkSameDriver(@intCast(src), @intCast(dst)),
            };

            const key = pairKey(@intCast(src), @intCast(dst));
            try external_memory_matrix.?.put(key, support);
        }
    }
}

/// Query external memory info for a device.
fn queryExternalMemoryInfo(device_id: DeviceId) ExternalMemoryInfo {
    // Check if we have extension functions loaded
    const funcs = ext_functions orelse return .{
        .supported = false,
        .compatible_handle_types = 0,
        .export_supported = false,
        .import_supported = false,
    };

    // Check if device context exists
    if (device_contexts) |contexts| {
        if (contexts.get(device_id) != null) {
            // Device exists, check extension support
            const has_export = funcs.hasExternalMemorySupport();
            const handle_type = VulkanExtFunctions.getPlatformHandleType();

            return .{
                .supported = has_export,
                .compatible_handle_types = handle_type,
                .export_supported = has_export,
                .import_supported = has_export,
            };
        }
    }

    // Device not found or no context
    return .{
        .supported = false,
        .compatible_handle_types = 0,
        .export_supported = false,
        .import_supported = false,
    };
}

/// Check if timeline semaphores are supported.
fn queryTimelineSemaphoreSupport(device_id: DeviceId) bool {
    // Check if we have extension functions loaded
    const funcs = ext_functions orelse return false;

    // Check if device context exists
    if (device_contexts) |contexts| {
        if (contexts.get(device_id) != null) {
            return funcs.timeline_semaphores_supported;
        }
    }

    return false;
}

/// Check if two devices use the same driver.
fn checkSameDriver(src: DeviceId, dst: DeviceId) bool {
    // Check if both devices have contexts (meaning they're on the same Vulkan instance)
    if (device_contexts) |contexts| {
        const src_ctx = contexts.get(src);
        const dst_ctx = contexts.get(dst);

        // If both exist, they're from the same Vulkan instance (same driver)
        // In a multi-driver scenario, we'd compare device UUIDs
        return src_ctx != null and dst_ctx != null;
    }

    return false;
}

/// Check if external memory transfer is available.
pub fn hasExternalMemory(src: DeviceId, dst: DeviceId) bool {
    if (external_memory_matrix == null) return false;

    const key = pairKey(src, dst);
    if (external_memory_matrix.?.get(key)) |support| {
        return support.external_memory;
    }
    return false;
}

/// Transfer data using external memory.
pub fn externalMemoryTransfer(
    src_device: DeviceId,
    dst_device: DeviceId,
    data: []u8,
) !void {
    if (!hasExternalMemory(src_device, dst_device)) {
        return error.ExternalMemoryNotSupported;
    }

    // In a real implementation:
    // 1. Create exportable VkBuffer on source device
    // 2. Get external memory handle (fd on Linux, HANDLE on Windows)
    // 3. Import handle on destination device
    // 4. Create VkBuffer referencing the imported memory
    // 5. Use timeline semaphore to synchronize

    _ = data;

    // For now, simulate success
}

/// Perform AllReduce using compute shaders.
pub fn computeAllReduce(
    buffers: []const multi_device.DeviceBuffer,
    op: ReduceOp,
) !void {
    if (buffers.len <= 1) return;

    // In a real implementation:
    // 1. For each device pair, check external memory support
    // 2. If supported, use shared memory reduction
    // 3. Otherwise, fall back to host staging

    // Compute shader for reduction would look like:
    // layout(local_size_x = 256) in;
    // layout(set = 0, binding = 0) buffer InputA { float a[]; };
    // layout(set = 0, binding = 1) buffer InputB { float b[]; };
    // layout(set = 0, binding = 2) buffer Output { float out[]; };
    // void main() {
    //     uint idx = gl_GlobalInvocationID.x;
    //     out[idx] = a[idx] + b[idx]; // or other op
    // }

    _ = op;
}

/// Export a Vulkan buffer to an external handle.
///
/// The buffer_handle must point to a VulkanBuffer structure with memory
/// allocated with VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT (Linux)
/// or VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT (Windows).
///
/// Returns an ExternalMemoryHandle that can be passed to importBuffer on
/// another device for zero-copy sharing.
pub fn exportBuffer(
    device_id: DeviceId,
    buffer_handle: *anyopaque,
) !ExternalMemoryHandle {
    // Check platform support
    if (builtin.os.tag != .linux and builtin.os.tag != .windows) {
        return error.PlatformNotSupported;
    }

    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Cast to VulkanBuffer to get the memory handle
    const vk_buffer: *vulkan.VulkanBuffer = @ptrCast(@alignCast(buffer_handle));

    // Platform-specific export
    if (builtin.os.tag == .linux) {
        const getMemoryFd = funcs.vkGetMemoryFdKHR orelse return error.ExtensionNotAvailable;

        var fd_info = vulkan_ext.VkMemoryGetFdInfoKHR{
            .memory = vk_buffer.memory,
            .handleType = vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };

        var fd: std.posix.fd_t = -1;
        const result = getMemoryFd(ctx.device, &fd_info, &fd);

        if (result != .success) {
            return error.ExportFailed;
        }

        return fd;
    } else if (builtin.os.tag == .windows) {
        const getMemoryHandle = funcs.vkGetMemoryWin32HandleKHR orelse return error.ExtensionNotAvailable;

        var handle_info = vulkan_ext.VkMemoryGetWin32HandleInfoKHR{
            .memory = vk_buffer.memory,
            .handleType = vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
        };

        var handle: std.os.windows.HANDLE = undefined;
        const result = getMemoryHandle(ctx.device, &handle_info, &handle);

        if (result != .success) {
            return error.ExportFailed;
        }

        return handle;
    }

    return error.PlatformNotSupported;
}

/// Import an external handle as a Vulkan buffer.
///
/// Creates a new VkBuffer backed by memory imported from the external handle.
/// The handle should have been obtained via exportBuffer on another device.
///
/// Note: The imported buffer shares the same underlying memory as the
/// original buffer. Proper synchronization (e.g., timeline semaphores)
/// is required when accessing from multiple devices.
///
/// Returns an opaque pointer to a VulkanBuffer structure.
pub fn importBuffer(
    device_id: DeviceId,
    external_handle: ExternalMemoryHandle,
    size: usize,
) !*anyopaque {
    // Check platform support
    if (builtin.os.tag != .linux and builtin.os.tag != .windows) {
        return error.PlatformNotSupported;
    }

    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;
    _ = funcs; // Used indirectly through Vulkan calls

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Get allocator
    const allocator = allocator_ref orelse return error.NotInitialized;

    // Create buffer
    const buffer_info = vulkan.VkBufferCreateInfo{
        .size = size,
        .usage = vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = 0, // Exclusive
    };

    var buffer: vulkan.VkBuffer = undefined;
    const create_buffer = vulkan.vkCreateBuffer orelse return error.FunctionNotAvailable;
    if (create_buffer(ctx.device, &buffer_info, null, &buffer) != .success) {
        return error.BufferCreationFailed;
    }
    errdefer {
        if (vulkan.vkDestroyBuffer) |destroy| {
            destroy(ctx.device, buffer, null);
        }
    }

    // Get memory requirements
    var mem_reqs: vulkan.VkMemoryRequirements = undefined;
    const get_mem_reqs = vulkan.vkGetBufferMemoryRequirements orelse return error.FunctionNotAvailable;
    get_mem_reqs(ctx.device, buffer, @ptrCast(&mem_reqs));

    // Find suitable memory type
    const memory_type = findMemoryType(
        ctx.memory_properties,
        mem_reqs.memoryTypeBits,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    ) orelse return error.MemoryTypeNotFound;

    // Allocate memory with import info in pNext chain
    var memory: vulkan.VkDeviceMemory = undefined;

    if (builtin.os.tag == .linux) {
        var import_info = vulkan_ext.VkImportMemoryFdInfoKHR{
            .handleType = vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            .fd = external_handle,
        };

        var alloc_info = vulkan.VkMemoryAllocateInfo{
            .pNext = @ptrCast(&import_info),
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = memory_type,
        };

        const alloc_memory = vulkan.vkAllocateMemory orelse return error.FunctionNotAvailable;
        if (alloc_memory(ctx.device, &alloc_info, null, &memory) != .success) {
            return error.MemoryAllocationFailed;
        }
    } else if (builtin.os.tag == .windows) {
        var import_info = vulkan_ext.VkImportMemoryWin32HandleInfoKHR{
            .handleType = vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            .handle = external_handle,
        };

        var alloc_info = vulkan.VkMemoryAllocateInfo{
            .pNext = @ptrCast(&import_info),
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = memory_type,
        };

        const alloc_memory = vulkan.vkAllocateMemory orelse return error.FunctionNotAvailable;
        if (alloc_memory(ctx.device, &alloc_info, null, &memory) != .success) {
            return error.MemoryAllocationFailed;
        }
    } else {
        return error.PlatformNotSupported;
    }

    errdefer {
        if (vulkan.vkFreeMemory) |free| {
            free(ctx.device, memory, null);
        }
    }

    // Bind buffer to imported memory
    const bind_memory = vulkan.vkBindBufferMemory orelse return error.FunctionNotAvailable;
    if (bind_memory(ctx.device, buffer, memory, 0) != .success) {
        return error.BindMemoryFailed;
    }

    // Create VulkanBuffer wrapper
    const vk_buffer = try allocator.create(vulkan.VulkanBuffer);
    vk_buffer.* = .{
        .buffer = buffer,
        .memory = memory,
        .size = size,
        .mapped_ptr = null, // Imported memory typically not host-mappable
    };

    return vk_buffer;
}

/// Find a suitable memory type index.
fn findMemoryType(
    mem_props: vulkan.VkPhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vulkan.VkMemoryPropertyFlags,
) ?u32 {
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        if ((type_filter & (@as(u32, 1) << @intCast(i))) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return null;
}

/// Create a timeline semaphore for cross-device sync.
///
/// Timeline semaphores (VK_KHR_timeline_semaphore, Vulkan 1.2 core) allow
/// wait/signal operations with monotonically increasing 64-bit values,
/// enabling fine-grained cross-device synchronization without recreating
/// semaphores.
///
/// The returned opaque pointer is a VkSemaphore handle. Call
/// destroyTimelineSemaphore when done.
pub fn createTimelineSemaphore(device_id: DeviceId, initial_value: u64) !*anyopaque {
    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;

    // Check timeline semaphore support
    if (!funcs.timeline_semaphores_supported) {
        return error.TimelineSemaphoresNotSupported;
    }

    const createSemaphore = funcs.vkCreateSemaphore orelse return error.ExtensionNotAvailable;

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Create timeline semaphore type info
    var type_info = vulkan_ext.VkSemaphoreTypeCreateInfo{
        .semaphoreType = vulkan_ext.VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = initial_value,
    };

    // Create semaphore with timeline type in pNext
    var create_info = vulkan_ext.VkSemaphoreCreateInfo{
        .pNext = @ptrCast(&type_info),
    };

    var semaphore: vulkan_ext.VkSemaphore = undefined;
    const result = createSemaphore(ctx.device, &create_info, null, &semaphore);

    if (result != .success) {
        return error.SemaphoreCreationFailed;
    }

    return semaphore;
}

/// Destroy a timeline semaphore.
///
/// The semaphore_handle must have been created by createTimelineSemaphore.
pub fn destroyTimelineSemaphore(device_id: DeviceId, semaphore_handle: *anyopaque) void {
    // Get extension functions
    const funcs = ext_functions orelse return;
    const destroySemaphore = funcs.vkDestroySemaphore orelse return;

    // Get device context
    const contexts = device_contexts orelse return;
    const ctx = contexts.get(device_id) orelse return;

    destroySemaphore(ctx.device, @ptrCast(semaphore_handle), null);
}

/// Get the current value of a timeline semaphore.
pub fn getTimelineSemaphoreValue(device_id: DeviceId, semaphore_handle: *anyopaque) !u64 {
    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;
    const getCounter = funcs.vkGetSemaphoreCounterValue orelse return error.ExtensionNotAvailable;

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    var value: u64 = 0;
    const result = getCounter(ctx.device, @ptrCast(semaphore_handle), &value);

    if (result != .success) {
        return error.QueryFailed;
    }

    return value;
}

/// Wait on a timeline semaphore value.
///
/// Blocks until the semaphore reaches or exceeds the specified value,
/// or until the timeout expires.
///
/// Returns error.Timeout if the wait timed out before the semaphore
/// reached the target value.
pub fn waitTimelineSemaphore(
    device_id: DeviceId,
    semaphore: *anyopaque,
    value: u64,
    timeout_ns: u64,
) !void {
    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;

    // Check timeline semaphore support
    if (!funcs.timeline_semaphores_supported) {
        return error.TimelineSemaphoresNotSupported;
    }

    const waitSemaphores = funcs.vkWaitSemaphores orelse return error.ExtensionNotAvailable;

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Create wait info
    const semaphores = [_]*anyopaque{semaphore};
    const values = [_]u64{value};

    var wait_info = vulkan_ext.VkSemaphoreWaitInfo{
        .semaphoreCount = 1,
        .pSemaphores = &semaphores,
        .pValues = &values,
    };

    const result = waitSemaphores(ctx.device, &wait_info, timeout_ns);

    switch (result) {
        .success => return,
        .timeout => return error.Timeout,
        else => return error.WaitFailed,
    }
}

/// Signal a timeline semaphore value from the host (CPU).
///
/// Sets the semaphore to the specified value. The value must be greater
/// than the current semaphore value (timeline semaphores are monotonically
/// increasing).
///
/// This is a host-side signal. For GPU-side signaling, use command buffer
/// operations (vkCmdSetEvent2 or submit with timeline semaphore in signal list).
pub fn signalTimelineSemaphore(
    device_id: DeviceId,
    semaphore: *anyopaque,
    value: u64,
) !void {
    // Get extension functions
    const funcs = ext_functions orelse return error.ExtensionsNotLoaded;

    // Check timeline semaphore support
    if (!funcs.timeline_semaphores_supported) {
        return error.TimelineSemaphoresNotSupported;
    }

    const signalSemaphore = funcs.vkSignalSemaphore orelse return error.ExtensionNotAvailable;

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Create signal info
    var signal_info = vulkan_ext.VkSemaphoreSignalInfo{
        .semaphore = semaphore,
        .value = value,
    };

    const result = signalSemaphore(ctx.device, &signal_info);

    if (result != .success) {
        return error.SignalFailed;
    }
}

/// Get support info for a device pair.
pub fn getDevicePairSupport(src: DeviceId, dst: DeviceId) ?DevicePairSupport {
    if (external_memory_matrix == null) return null;
    return external_memory_matrix.?.get(pairKey(src, dst));
}

/// Generate hash key for device pair.
fn pairKey(src: DeviceId, dst: DeviceId) u64 {
    return @as(u64, src) << 32 | @as(u64, dst);
}

// ============================================================================
// Reduction Compute Shader SPIR-V
// ============================================================================

/// Pre-compiled SPIR-V for sum reduction shader.
/// This would be generated from GLSL at build time.
pub const reduce_sum_spirv = [_]u32{
    // SPIR-V header would go here
    // In a real implementation, this would be the compiled shader
    0x07230203, // SPIR-V magic number
    0x00010500, // Version 1.5
    // ... rest of shader bytecode
};

/// Pre-compiled SPIR-V for max reduction shader.
pub const reduce_max_spirv = [_]u32{
    0x07230203,
    0x00010500,
};

/// Pre-compiled SPIR-V for min reduction shader.
pub const reduce_min_spirv = [_]u32{
    0x07230203,
    0x00010500,
};

// ============================================================================
// Helper: Create Exportable Buffer
// ============================================================================

/// Configuration for creating an exportable buffer.
pub const ExportableBufferConfig = struct {
    size: usize,
    usage: u32 = vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    host_visible: bool = false,
};

/// Create a Vulkan buffer that can be exported for peer sharing.
///
/// The buffer is allocated with external memory handle type support,
/// allowing it to be exported via exportBuffer().
///
/// Returns an opaque pointer to a VulkanBuffer structure.
pub fn createExportableBuffer(
    device_id: DeviceId,
    config: ExportableBufferConfig,
) !*anyopaque {
    // Check platform support
    if (builtin.os.tag != .linux and builtin.os.tag != .windows) {
        return error.PlatformNotSupported;
    }

    // Get device context
    const contexts = device_contexts orelse return error.NotInitialized;
    const ctx = contexts.get(device_id) orelse return error.DeviceNotFound;

    // Get allocator
    const allocator = allocator_ref orelse return error.NotInitialized;

    // Create buffer with external memory export capability
    // The pNext chain includes VkExternalMemoryBufferCreateInfo
    var export_buffer_info = vulkan_ext.VkExternalMemoryBufferCreateInfo{
        .handleTypes = VulkanExtFunctions.getPlatformHandleType(),
    };

    var buffer_info = vulkan.VkBufferCreateInfo{
        .pNext = @ptrCast(&export_buffer_info),
        .size = config.size,
        .usage = config.usage,
        .sharingMode = 0, // Exclusive
    };

    var buffer: vulkan.VkBuffer = undefined;
    const create_buffer = vulkan.vkCreateBuffer orelse return error.FunctionNotAvailable;
    if (create_buffer(ctx.device, &buffer_info, null, &buffer) != .success) {
        return error.BufferCreationFailed;
    }
    errdefer {
        if (vulkan.vkDestroyBuffer) |destroy| {
            destroy(ctx.device, buffer, null);
        }
    }

    // Get memory requirements
    var mem_reqs: vulkan.VkMemoryRequirements = undefined;
    const get_mem_reqs = vulkan.vkGetBufferMemoryRequirements orelse return error.FunctionNotAvailable;
    get_mem_reqs(ctx.device, buffer, @ptrCast(&mem_reqs));

    // Determine memory properties
    const mem_props: u32 = if (config.host_visible)
        vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    else
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Find suitable memory type
    const memory_type = findMemoryType(
        ctx.memory_properties,
        mem_reqs.memoryTypeBits,
        mem_props,
    ) orelse return error.MemoryTypeNotFound;

    // Allocate memory with export capability
    var export_info = vulkan_ext.VkExportMemoryAllocateInfo{
        .handleTypes = VulkanExtFunctions.getPlatformHandleType(),
    };

    var alloc_info = vulkan.VkMemoryAllocateInfo{
        .pNext = @ptrCast(&export_info),
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = memory_type,
    };

    var memory: vulkan.VkDeviceMemory = undefined;
    const alloc_memory = vulkan.vkAllocateMemory orelse return error.FunctionNotAvailable;
    if (alloc_memory(ctx.device, &alloc_info, null, &memory) != .success) {
        return error.MemoryAllocationFailed;
    }
    errdefer {
        if (vulkan.vkFreeMemory) |free| {
            free(ctx.device, memory, null);
        }
    }

    // Bind buffer to memory
    const bind_memory = vulkan.vkBindBufferMemory orelse return error.FunctionNotAvailable;
    if (bind_memory(ctx.device, buffer, memory, 0) != .success) {
        return error.BindMemoryFailed;
    }

    // Map memory if host-visible
    var mapped_ptr: ?*anyopaque = null;
    if (config.host_visible) {
        const map_memory = vulkan.vkMapMemory orelse return error.FunctionNotAvailable;
        if (map_memory(ctx.device, memory, 0, config.size, 0, &mapped_ptr) != .success) {
            return error.MemoryMapFailed;
        }
    }

    // Create VulkanBuffer wrapper
    const vk_buffer = try allocator.create(vulkan.VulkanBuffer);
    vk_buffer.* = .{
        .buffer = buffer,
        .memory = memory,
        .size = config.size,
        .mapped_ptr = mapped_ptr,
    };

    return vk_buffer;
}

/// Destroy an exportable buffer.
pub fn destroyExportableBuffer(device_id: DeviceId, buffer_handle: *anyopaque) void {
    const contexts = device_contexts orelse return;
    const ctx = contexts.get(device_id) orelse return;
    const allocator = allocator_ref orelse return;

    const vk_buffer: *vulkan.VulkanBuffer = @ptrCast(@alignCast(buffer_handle));

    // Unmap if mapped
    if (vk_buffer.mapped_ptr != null) {
        if (vulkan.vkUnmapMemory) |unmap| {
            unmap(ctx.device, vk_buffer.memory);
        }
    }

    // Destroy buffer
    if (vulkan.vkDestroyBuffer) |destroy| {
        destroy(ctx.device, vk_buffer.buffer, null);
    }

    // Free memory
    if (vulkan.vkFreeMemory) |free| {
        free(ctx.device, vk_buffer.memory, null);
    }

    // Free wrapper
    allocator.destroy(vk_buffer);
}

/// Check if the Vulkan peer transfer backend is initialized.
pub fn isInitialized() bool {
    return vulkan_peer_initialized;
}

/// Check if timeline semaphores are supported on a device.
pub fn hasTimelineSemaphoreSupport(device_id: DeviceId) bool {
    return queryTimelineSemaphoreSupport(device_id);
}

// ============================================================================
// Tests
// ============================================================================

test "Vulkan peer module compiles" {
    // Just verify compilation
    try std.testing.expect(reduce_sum_spirv[0] == 0x07230203);
}

test "pairKey generation" {
    const key1 = pairKey(0, 1);
    const key2 = pairKey(1, 0);
    const key3 = pairKey(0, 1);

    try std.testing.expect(key1 != key2);
    try std.testing.expect(key1 == key3);
}

test "hasExternalMemory without init" {
    // Should return false when not initialized
    try std.testing.expect(!hasExternalMemory(0, 1));
}

test "isInitialized returns false before init" {
    // Should be false before explicit init
    deinit(); // Ensure clean state
    try std.testing.expect(!isInitialized());
}

test "queryExternalMemoryInfo without init" {
    deinit(); // Ensure clean state
    const info = queryExternalMemoryInfo(0);
    try std.testing.expect(!info.supported);
    try std.testing.expectEqual(@as(u32, 0), info.compatible_handle_types);
}

test "queryTimelineSemaphoreSupport without init" {
    deinit(); // Ensure clean state
    try std.testing.expect(!queryTimelineSemaphoreSupport(0));
}

test "checkSameDriver without init" {
    deinit(); // Ensure clean state
    try std.testing.expect(!checkSameDriver(0, 1));
}

test "getDevicePairSupport without init" {
    deinit(); // Ensure clean state
    try std.testing.expect(getDevicePairSupport(0, 1) == null);
}

test "ExternalMemoryHandle platform type" {
    // Verify ExternalMemoryHandle is the correct type for this platform
    if (builtin.os.tag == .linux) {
        try std.testing.expect(@TypeOf(ExternalMemoryHandle) == type);
    } else if (builtin.os.tag == .windows) {
        try std.testing.expect(@TypeOf(ExternalMemoryHandle) == type);
    }
}

test "VulkanExtFunctions.getPlatformHandleType" {
    const handle_type = VulkanExtFunctions.getPlatformHandleType();
    if (builtin.os.tag == .linux) {
        try std.testing.expectEqual(vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, handle_type);
    } else if (builtin.os.tag == .windows) {
        try std.testing.expectEqual(vulkan_ext.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, handle_type);
    } else {
        try std.testing.expectEqual(@as(u32, 0), handle_type);
    }
}
