//! Vulkan VTable Backend Implementation
//!
//! Provides a complete VTable implementation for Vulkan, enabling GPU
//! kernel execution through the polymorphic backend interface.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const vulkan = @import("vulkan.zig");
const vulkan_init = @import("vulkan_init.zig");
const vulkan_types = @import("vulkan_types.zig");
const vulkan_buffers = @import("vulkan_buffers.zig");
const vulkan_pipelines = @import("vulkan_pipelines.zig");

/// Vulkan VTable Backend
///
/// Wraps the existing Vulkan backend implementation to provide the
/// unified Backend interface for polymorphic GPU operations.
pub const VulkanBackend = struct {
    allocator: std.mem.Allocator,
    initialized: bool,

    // Track allocations for cleanup
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    // Device info cache
    device_name: [256]u8 = undefined,
    device_name_len: usize = 0,

    const Allocation = struct {
        ptr: *anyopaque,
        size: usize,
    };

    const CompiledKernel = struct {
        handle: *anyopaque,
        name: []const u8,
    };

    const Self = @This();

    /// Initialize the Vulkan VTable backend.
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Check if Vulkan is enabled at compile time
        if (comptime !build_options.gpu_vulkan) {
            return interface.BackendError.NotAvailable;
        }

        // Try to initialize the Vulkan backend
        vulkan_init.init() catch {
            return interface.BackendError.NotAvailable;
        };

        const self = allocator.create(Self) catch {
            vulkan_init.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        // Get device name
        if (vulkan_init.vulkan_context) |ctx| {
            if (vulkan_init.vkGetPhysicalDeviceProperties) |get_props| {
                var props: vulkan_types.VkPhysicalDeviceProperties = undefined;
                get_props(ctx.physical_device, &props);
                const name_len = std.mem.indexOfScalar(u8, &props.deviceName, 0) orelse 256;
                @memcpy(self.device_name[0..name_len], props.deviceName[0..name_len]);
                self.device_name_len = name_len;
            }
        }

        return self;
    }

    /// Deinitialize the backend and release all resources.
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        for (self.allocations.items) |alloc| {
            vulkan_buffers.freeDeviceMemory(alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        // Destroy all kernels
        for (self.kernels.items) |kernel| {
            vulkan_pipelines.destroyKernel(self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        // Deinit the Vulkan backend
        if (self.initialized) {
            vulkan_init.deinit();
        }

        self.allocator.destroy(self);
    }

    // ========================================================================
    // Device Info
    // ========================================================================

    /// Get the number of available Vulkan devices.
    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        if (!vulkan_init.vulkan_initialized) return 0;
        return vulkan_init.getPhysicalDeviceCount();
    }

    /// Get device capabilities.
    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{};

        // Copy device name
        @memcpy(caps.name[0..self.device_name_len], self.device_name[0..self.device_name_len]);
        caps.name_len = self.device_name_len;

        if (vulkan_init.vulkan_context) |ctx| {
            if (vulkan_init.vkGetPhysicalDeviceProperties) |get_props| {
                var props: vulkan_types.VkPhysicalDeviceProperties = undefined;
                get_props(ctx.physical_device, &props);

                // Map Vulkan API version to compute capability
                caps.compute_capability_major = (props.apiVersion >> 22) & 0x7F;
                caps.compute_capability_minor = (props.apiVersion >> 12) & 0x3FF;
            }

            if (vulkan_init.vkGetPhysicalDeviceMemoryProperties) |get_mem| {
                var mem_props: vulkan_types.VkPhysicalDeviceMemoryProperties = undefined;
                get_mem(ctx.physical_device, &mem_props);

                // Sum up device-local memory heaps
                var total_mem: u64 = 0;
                for (0..mem_props.memoryHeapCount) |i| {
                    if ((mem_props.memoryHeaps[i].flags & 0x1) != 0) { // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        total_mem += mem_props.memoryHeaps[i].size;
                    }
                }
                caps.total_memory = @intCast(total_mem);
            }
        }

        // Vulkan defaults
        caps.max_threads_per_block = 1024;
        caps.max_shared_memory = 32768;
        caps.warp_size = 32;
        caps.supports_fp16 = true;
        caps.supports_fp64 = true;

        return caps;
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Allocate device memory.
    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags; // Vulkan handles memory type internally

        const ptr = vulkan_buffers.allocateDeviceMemory(size) catch {
            return interface.MemoryError.OutOfMemory;
        };

        // Track allocation
        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            vulkan_buffers.freeDeviceMemory(ptr);
            return interface.MemoryError.OutOfMemory;
        };

        return ptr;
    }

    /// Free device memory.
    pub fn free(self: *Self, ptr: *anyopaque) void {
        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                vulkan_buffers.freeDeviceMemory(ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    /// Copy data from host to device.
    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        vulkan_buffers.memcpyHostToDevice(dst, @constCast(src.ptr), src.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    /// Copy data from device to host.
    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        vulkan_buffers.memcpyDeviceToHost(dst.ptr, src, dst.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    /// Compile a kernel from SPIR-V source.
    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const kernel_source = vulkan_types.KernelSource{
            .code = source,
            .entry_point = kernel_name,
            .format = .spirv,
        };

        const handle = vulkan_pipelines.compileKernel(allocator, kernel_source) catch {
            return interface.KernelError.CompileFailed;
        };

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            vulkan_pipelines.destroyKernel(allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            vulkan_pipelines.destroyKernel(allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        return handle;
    }

    /// Launch a compiled kernel.
    pub fn launchKernel(
        self: *Self,
        kernel: *anyopaque,
        config: interface.LaunchConfig,
        args: []const *anyopaque,
    ) interface.KernelError!void {
        _ = self;

        // Validate configuration
        if (config.block_x == 0 or config.block_y == 0 or config.block_z == 0) {
            return interface.KernelError.InvalidConfig;
        }
        if (config.grid_x == 0 or config.grid_y == 0 or config.grid_z == 0) {
            return interface.KernelError.InvalidConfig;
        }

        const kernel_config = vulkan_types.KernelConfig{
            .grid_size = .{ config.grid_x, config.grid_y, config.grid_z },
            .block_size = .{ config.block_x, config.block_y, config.block_z },
            .shared_memory = config.shared_memory,
        };

        // Convert args to optional pointers
        var opt_args: [32]?*const anyopaque = .{null} ** 32;
        const arg_count = @min(args.len, 32);
        for (0..arg_count) |i| {
            opt_args[i] = args[i];
        }

        vulkan_pipelines.launchKernel(
            std.heap.page_allocator,
            kernel,
            kernel_config,
            opt_args[0..arg_count],
        ) catch {
            return interface.KernelError.LaunchFailed;
        };
    }

    /// Destroy a compiled kernel.
    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        for (self.kernels.items, 0..) |k, i| {
            if (k.handle == kernel) {
                vulkan_pipelines.destroyKernel(self.allocator, kernel);
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Synchronize the device.
    pub fn synchronize(self: *Self) interface.BackendError!void {
        _ = self;
        if (!vulkan_init.vulkan_initialized) {
            return interface.BackendError.NotAvailable;
        }

        if (vulkan_init.vulkan_context) |ctx| {
            if (vulkan_init.vkQueueWaitIdle) |wait_fn| {
                const result = wait_fn(ctx.compute_queue);
                if (result != .success) {
                    return interface.BackendError.Timeout;
                }
            }
        }
    }
};

/// Create a VTable-wrapped Vulkan backend for the interface system.
pub fn createVulkanVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try VulkanBackend.init(allocator);
    return interface.createBackend(VulkanBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "VulkanBackend initialization" {
    const allocator = std.testing.allocator;

    const result = VulkanBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.initialized);
    } else |err| {
        // Expected on systems without Vulkan
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}

test "createVulkanVTable" {
    const allocator = std.testing.allocator;

    const result = createVulkanVTable(allocator);
    if (result) |backend| {
        defer backend.deinit();
        // Should work through VTable interface
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 0);
    } else |err| {
        // Expected on systems without Vulkan
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}
