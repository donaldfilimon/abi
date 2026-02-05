//! WebGPU VTable Backend Implementation
//!
//! Provides a complete VTable implementation for WebGPU, enabling GPU
//! kernel execution through the polymorphic backend interface.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const webgpu = @import("webgpu.zig");
const types = @import("../kernel_types.zig");

/// WebGPU VTable Backend
///
/// Wraps the existing WebGPU backend implementation to provide the
/// unified Backend interface for polymorphic GPU operations.
pub const WebGpuBackend = struct {
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

    /// Initialize the WebGPU VTable backend.
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Check if WebGPU is enabled at compile time
        if (comptime !build_options.gpu_webgpu) {
            return interface.BackendError.NotAvailable;
        }

        // Try to initialize the WebGPU backend
        webgpu.init() catch {
            return interface.BackendError.NotAvailable;
        };

        const self = allocator.create(Self) catch {
            webgpu.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        // Set device name
        const name = "WebGPU Device";
        @memcpy(self.device_name[0..name.len], name);
        self.device_name_len = name.len;

        return self;
    }

    /// Deinitialize the backend and release all resources.
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        for (self.allocations.items) |alloc| {
            webgpu.freeDeviceMemory(self.allocator, alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        // Destroy all kernels
        for (self.kernels.items) |kernel| {
            webgpu.destroyKernel(self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        // Deinit the WebGPU backend
        if (self.initialized) {
            webgpu.deinit();
        }

        self.allocator.destroy(self);
    }

    // ========================================================================
    // Device Info
    // ========================================================================

    /// Get the number of available WebGPU devices.
    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        // WebGPU typically exposes 1 adapter
        return 1;
    }

    /// Get device capabilities.
    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{};

        // Copy device name
        @memcpy(caps.name[0..self.device_name_len], self.device_name[0..self.device_name_len]);
        caps.name_len = self.device_name_len;

        // WebGPU defaults (conservative cross-platform values)
        caps.max_threads_per_block = 256;
        caps.max_shared_memory = 16384;
        caps.warp_size = 32;
        caps.supports_fp16 = true;
        caps.supports_fp64 = false; // WebGPU doesn't support f64 in compute shaders

        return caps;
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Allocate device memory.
    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags; // WebGPU handles memory type internally

        const ptr = webgpu.allocateDeviceMemory(self.allocator, size) catch {
            return interface.MemoryError.OutOfMemory;
        };

        // Track allocation
        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            webgpu.freeDeviceMemory(self.allocator, ptr);
            return interface.MemoryError.OutOfMemory;
        };

        return ptr;
    }

    /// Free device memory.
    pub fn free(self: *Self, ptr: *anyopaque) void {
        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                webgpu.freeDeviceMemory(self.allocator, ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    /// Copy data from host to device.
    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        webgpu.memcpyHostToDevice(dst, @constCast(src.ptr), src.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    /// Copy data from device to host.
    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        webgpu.memcpyDeviceToHost(dst.ptr, src, dst.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyToDevice(dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyFromDevice(dst, src);
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    /// Compile a kernel from WGSL source.
    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const kernel_source = types.KernelSource{
            .code = source,
            .entry_point = kernel_name,
            .format = .wgsl,
        };

        const handle = webgpu.compileKernel(allocator, kernel_source) catch {
            return interface.KernelError.CompileFailed;
        };

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            webgpu.destroyKernel(allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            webgpu.destroyKernel(allocator, handle);
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

        const kernel_config = types.KernelConfig{
            .grid_dim = .{ config.grid_x, config.grid_y, config.grid_z },
            .block_dim = .{ config.block_x, config.block_y, config.block_z },
            .shared_memory = config.shared_memory,
        };

        // Convert args to optional pointers
        var opt_args: [32]?*const anyopaque = .{null} ** 32;
        const arg_count = @min(args.len, 32);
        for (0..arg_count) |i| {
            opt_args[i] = args[i];
        }

        webgpu.launchKernel(
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
                webgpu.destroyKernel(self.allocator, kernel);
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
        // Poll device to flush pending operations
        _ = webgpu.pollDevice(true);
    }
};

/// Create a VTable-wrapped WebGPU backend for the interface system.
pub fn createWebGpuVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try WebGpuBackend.init(allocator);
    return interface.createBackend(WebGpuBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "WebGpuBackend initialization" {
    const allocator = std.testing.allocator;

    const result = WebGpuBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.initialized);
    } else |err| {
        // Expected on systems without WebGPU
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}

test "createWebGpuVTable" {
    const allocator = std.testing.allocator;

    const result = createWebGpuVTable(allocator);
    if (result) |backend| {
        defer backend.deinit();
        // Should work through VTable interface
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 0);
    } else |err| {
        // Expected on systems without WebGPU
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}
