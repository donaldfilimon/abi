//! CUDA VTable Backend Implementation
//!
//! Provides a complete VTable implementation for CUDA, enabling real GPU
//! kernel execution through the polymorphic backend interface.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const loader = @import("loader.zig");
const nvrtc = @import("nvrtc.zig");

pub const CudaBackend = struct {
    allocator: std.mem.Allocator,
    device_id: i32,
    context: ?*anyopaque,
    functions: ?*const loader.CudaFunctions,

    // Track allocations for cleanup
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    pub const Allocation = struct {
        device_ptr: u64,
        size: usize,
        is_host_pinned: bool,
        host_ptr: ?*anyopaque,
    };

    pub const CompiledKernel = struct {
        module: ?*anyopaque,
        function: ?*anyopaque,
        name: []const u8,
    };

    const Self = @This();

    /// Initialize a CUDA backend for the specified device.
    ///
    /// Returns BackendError.NotAvailable if CUDA is disabled at compile time
    /// or the CUDA driver cannot be loaded.
    /// Returns BackendError.DeviceNotFound if the device_id is invalid.
    /// Returns BackendError.InitFailed if CUDA initialization fails.
    pub fn init(allocator: std.mem.Allocator, device_id: i32) interface.BackendError!*Self {
        // Check if CUDA is enabled at compile time
        if (comptime !build_options.gpu_cuda) {
            return interface.BackendError.NotAvailable;
        }

        // Try to load CUDA driver
        const functions = loader.load(allocator) catch {
            return interface.BackendError.NotAvailable;
        };

        // Initialize NVRTC
        nvrtc.init() catch {
            // Non-fatal, runtime compilation will just fail later if needed
        };

        // Check that core functions are available
        const cu_init = functions.core.cuInit orelse {
            return interface.BackendError.NotAvailable;
        };

        const cu_device_get_count = functions.core.cuDeviceGetCount orelse {
            return interface.BackendError.NotAvailable;
        };

        const cu_device_get = functions.core.cuDeviceGet orelse {
            return interface.BackendError.NotAvailable;
        };

        const cu_ctx_create = functions.core.cuCtxCreate orelse {
            return interface.BackendError.NotAvailable;
        };

        // Initialize CUDA
        const init_result = cu_init(0);
        if (init_result != .success) {
            return interface.BackendError.InitFailed;
        }

        // Check device count
        var device_count: i32 = 0;
        const count_result = cu_device_get_count(&device_count);
        if (count_result != .success or device_count == 0) {
            return interface.BackendError.DeviceNotFound;
        }

        if (device_id >= device_count or device_id < 0) {
            return interface.BackendError.DeviceNotFound;
        }

        // Get device handle
        var device: i32 = undefined;
        const device_result = cu_device_get(&device, device_id);
        if (device_result != .success) {
            return interface.BackendError.DeviceNotFound;
        }

        // Create context
        var context: ?*anyopaque = null;
        const ctx_result = cu_ctx_create(&context, 0, device);
        if (ctx_result != .success) {
            return interface.BackendError.InitFailed;
        }
        errdefer {
            // Destroy context on subsequent allocation failure
            if (functions.core.cuCtxDestroy) |cu_ctx_destroy| {
                _ = cu_ctx_destroy(context);
            }
        }

        const self = allocator.create(Self) catch {
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .device_id = device_id,
            .context = context,
            .functions = functions,
            .allocations = .empty,
            .kernels = .empty,
        };

        return self;
    }

    /// Deinitialize the backend, cleaning up all resources.
    ///
    /// This frees all tracked allocations, destroys all compiled kernels,
    /// and destroys the CUDA context.
    pub fn deinit(self: *Self) void {
        if (self.functions) |funcs| {
            // Free all device allocations
            if (funcs.memory.cuMemFree) |cu_mem_free| {
                for (self.allocations.items) |alloc| {
                    if (!alloc.is_host_pinned) {
                        _ = cu_mem_free(alloc.device_ptr);
                    }
                }
            }

            // Free all host-pinned allocations
            if (funcs.memory.cuMemFreeHost) |cu_mem_free_host| {
                for (self.allocations.items) |alloc| {
                    if (alloc.is_host_pinned) {
                        if (alloc.host_ptr) |ptr| {
                            _ = cu_mem_free_host(ptr);
                        }
                    }
                }
            }

            // Destroy all kernel modules
            if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                for (self.kernels.items) |kernel| {
                    if (kernel.module) |module| {
                        _ = cu_module_unload(module);
                    }
                    // Free the duplicated name string
                    self.allocator.free(kernel.name);
                }
            } else {
                // Still need to free names even if we can't unload modules
                for (self.kernels.items) |kernel| {
                    self.allocator.free(kernel.name);
                }
            }

            // Destroy context
            if (funcs.core.cuCtxDestroy) |cu_ctx_destroy| {
                if (self.context) |ctx| {
                    _ = cu_ctx_destroy(ctx);
                }
            }
        } else {
            // No functions, still free kernel names
            for (self.kernels.items) |kernel| {
                self.allocator.free(kernel.name);
            }
        }

        self.allocations.deinit(self.allocator);
        self.kernels.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Get the device count.
    pub fn getDeviceCount(self: *Self) u32 {
        const funcs = self.functions orelse return 0;
        const cu_device_get_count = funcs.core.cuDeviceGetCount orelse return 0;

        var count: i32 = 0;
        const result = cu_device_get_count(&count);
        if (result != .success) return 0;
        return if (count > 0) @intCast(count) else 0;
    }

    /// Get device capabilities.
    pub fn getDeviceCaps(self: *Self, requested_device_id: u32) interface.BackendError!interface.DeviceCaps {
        var caps = interface.DeviceCaps{};

        const funcs = self.functions orelse return interface.BackendError.NotAvailable;

        const device_id_i32: i32 = @intCast(requested_device_id);

        // Get device handle
        const cu_device_get = funcs.core.cuDeviceGet orelse return interface.BackendError.NotAvailable;
        var device: i32 = undefined;
        if (cu_device_get(&device, device_id_i32) != .success) {
            return interface.BackendError.DeviceNotFound;
        }

        // Get device name
        if (funcs.device.cuDeviceGetName) |cu_device_get_name| {
            if (cu_device_get_name(&caps.name, 256, device) == .success) {
                // Find the null terminator to set name_len
                for (caps.name, 0..) |c, i| {
                    if (c == 0) {
                        caps.name_len = i;
                        break;
                    }
                }
            }
        }

        // Get total memory
        if (funcs.device.cuDeviceTotalMem) |cu_device_total_mem| {
            var total_mem: usize = 0;
            if (cu_device_total_mem(&total_mem, device) == .success) {
                caps.total_memory = total_mem;
            }
        }

        // Get device attributes
        if (funcs.device.cuDeviceGetAttribute) |cu_device_get_attribute| {
            var val: i32 = 0;

            // Compute capability (attributes 75 and 76)
            if (cu_device_get_attribute(&val, 75, device) == .success) {
                caps.compute_capability_major = @intCast(val);
            }
            if (cu_device_get_attribute(&val, 76, device) == .success) {
                caps.compute_capability_minor = @intCast(val);
            }

            // Max threads per block (attribute 1)
            if (cu_device_get_attribute(&val, 1, device) == .success) {
                caps.max_threads_per_block = @intCast(val);
            }

            // Max shared memory per block (attribute 8)
            if (cu_device_get_attribute(&val, 8, device) == .success) {
                caps.max_shared_memory = @intCast(val);
            }

            // Warp size (attribute 10)
            if (cu_device_get_attribute(&val, 10, device) == .success) {
                caps.warp_size = @intCast(val);
            }

            // Async engine count (attribute 40)
            if (cu_device_get_attribute(&val, 40, device) == .success) {
                caps.async_engine_count = @intCast(val);
            }

            // Unified addressing (attribute 41) for unified memory support
            if (cu_device_get_attribute(&val, 41, device) == .success) {
                caps.unified_memory = val != 0;
            }

            // FP16 support: compute capability >= 5.3
            caps.supports_fp16 = caps.compute_capability_major > 5 or
                (caps.compute_capability_major == 5 and caps.compute_capability_minor >= 3);

            // FP64 support: compute capability >= 1.3
            caps.supports_fp64 = caps.compute_capability_major > 1 or
                (caps.compute_capability_major == 1 and caps.compute_capability_minor >= 3);

            // INT8 support: compute capability >= 6.1
            caps.supports_int8 = caps.compute_capability_major > 6 or
                (caps.compute_capability_major == 6 and caps.compute_capability_minor >= 1);
        }

        return caps;
    }

    /// Synchronize the CUDA context.
    pub fn synchronize(self: *Self) interface.BackendError!void {
        const funcs = self.functions orelse return interface.BackendError.NotAvailable;
        const cu_ctx_synchronize = funcs.core.cuCtxSynchronize orelse return interface.BackendError.NotAvailable;

        const result = cu_ctx_synchronize();
        if (result != .success) {
            return interface.BackendError.Timeout;
        }
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Allocate device memory.
    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        const funcs = self.functions orelse return interface.MemoryError.OutOfMemory;

        if (flags.host_visible) {
            // Allocate host-pinned memory
            const cu_mem_alloc_host = funcs.memory.cuMemAllocHost orelse return interface.MemoryError.OutOfMemory;
            var host_ptr: ?*anyopaque = null;
            const result = cu_mem_alloc_host(&host_ptr, size);
            if (result != .success or host_ptr == null) {
                return interface.MemoryError.OutOfMemory;
            }

            // Track allocation
            self.allocations.append(self.allocator, .{
                .device_ptr = 0,
                .size = size,
                .is_host_pinned = true,
                .host_ptr = host_ptr,
            }) catch {
                if (funcs.memory.cuMemFreeHost) |cu_mem_free_host| {
                    _ = cu_mem_free_host(host_ptr.?);
                }
                return interface.MemoryError.OutOfMemory;
            };

            return host_ptr.?;
        } else {
            // Allocate device memory
            const cu_mem_alloc = funcs.memory.cuMemAlloc orelse return interface.MemoryError.OutOfMemory;
            var device_ptr: u64 = 0;
            const result = cu_mem_alloc(&device_ptr, size);
            if (result != .success or device_ptr == 0) {
                return interface.MemoryError.OutOfMemory;
            }

            // Track allocation
            self.allocations.append(self.allocator, .{
                .device_ptr = device_ptr,
                .size = size,
                .is_host_pinned = false,
                .host_ptr = null,
            }) catch {
                if (funcs.memory.cuMemFree) |cu_mem_free| {
                    _ = cu_mem_free(device_ptr);
                }
                return interface.MemoryError.OutOfMemory;
            };

            return @ptrFromInt(device_ptr);
        }
    }

    /// Free device memory.
    pub fn free(self: *Self, ptr: *anyopaque) void {
        const funcs = self.functions orelse return;
        const target_addr = @intFromPtr(ptr);

        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            const matches = if (alloc.is_host_pinned)
                alloc.host_ptr == ptr
            else
                alloc.device_ptr == target_addr;

            if (matches) {
                if (alloc.is_host_pinned) {
                    if (funcs.memory.cuMemFreeHost) |cu_mem_free_host| {
                        if (alloc.host_ptr) |host_ptr| {
                            _ = cu_mem_free_host(host_ptr);
                        }
                    }
                } else {
                    if (funcs.memory.cuMemFree) |cu_mem_free| {
                        _ = cu_mem_free(alloc.device_ptr);
                    }
                }
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    /// Copy data from host to device.
    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;
        const cu_memcpy_htod = funcs.memory.cuMemcpyHtoD orelse return interface.MemoryError.TransferFailed;

        const device_ptr = @intFromPtr(dst);
        const result = cu_memcpy_htod(device_ptr, src.ptr, src.len);
        if (result != .success) {
            return interface.MemoryError.TransferFailed;
        }
    }

    /// Copy data from device to host.
    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;
        const cu_memcpy_dtoh = funcs.memory.cuMemcpyDtoH orelse return interface.MemoryError.TransferFailed;

        const device_ptr = @intFromPtr(src);
        const result = cu_memcpy_dtoh(dst.ptr, device_ptr, dst.len);
        if (result != .success) {
            return interface.MemoryError.TransferFailed;
        }
    }

    /// Copy data from host to device asynchronously.
    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;

        // Try async transfer first, fall back to sync if async not available
        if (funcs.memory.cuMemcpyHtoDAsync) |cu_memcpy_htod_async| {
            const device_ptr = @intFromPtr(dst);
            const result = cu_memcpy_htod_async(device_ptr, src.ptr, src.len, stream);
            if (result != .success) {
                // Fall back to synchronous transfer
                return self.copyToDevice(dst, src);
            }
            return;
        }

        // Fall back to synchronous transfer
        return self.copyToDevice(dst, src);
    }

    /// Copy data from device to host asynchronously.
    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;

        // Try async transfer first, fall back to sync if async not available
        if (funcs.memory.cuMemcpyDtoHAsync) |cu_memcpy_dtoh_async| {
            const device_ptr = @intFromPtr(src);
            const result = cu_memcpy_dtoh_async(dst.ptr, device_ptr, dst.len, stream);
            if (result != .success) {
                // Fall back to synchronous transfer
                return self.copyFromDevice(dst, src);
            }
            return;
        }

        // Fall back to synchronous transfer
        return self.copyFromDevice(dst, src);
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    /// Compile a kernel from source (CUDA C or PTX).
    /// If source is CUDA C, it is compiled to PTX using NVRTC.
    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const funcs = self.functions orelse return interface.KernelError.CompileFailed;

        const cu_module_load_data = funcs.kernel.cuModuleLoadData orelse return interface.KernelError.CompileFailed;
        const cu_module_get_function = funcs.kernel.cuModuleGetFunction orelse return interface.KernelError.CompileFailed;

        // Check if source is likely PTX (contains .version directive)
        const is_ptx = std.mem.indexOf(u8, source, ".version") != null;

        var ptx_source: []const u8 = source;
        var ptx_result: ?nvrtc.CompileResult = null;

        if (!is_ptx) {
            // Compile C to PTX
            ptx_result = nvrtc.compileToPTX(allocator, source, kernel_name, .{}) catch |err| {
                std.log.err("NVRTC compilation failed: {}", .{err});
                return interface.KernelError.CompileFailed;
            };
            ptx_source = ptx_result.?.ptx;
        }
        defer if (ptx_result) |res| {
            allocator.free(res.ptx);
            allocator.free(res.log);
        };

        // Load module from PTX
        var module: ?*anyopaque = null;
        const load_result = cu_module_load_data(&module, ptx_source.ptr);
        if (load_result != .success or module == null) {
            return interface.KernelError.CompileFailed;
        }
        errdefer {
            if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                _ = cu_module_unload(module);
            }
        }

        // Create null-terminated kernel name
        const name_z = allocator.allocSentinel(u8, kernel_name.len, 0) catch {
            return interface.KernelError.CompileFailed;
        };
        defer allocator.free(name_z);
        @memcpy(name_z, kernel_name);

        // Get function from module
        var function: ?*anyopaque = null;
        const func_result = cu_module_get_function(&function, module, name_z.ptr);
        if (func_result != .success or function == null) {
            if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                _ = cu_module_unload(module);
            }
            return interface.KernelError.CompileFailed;
        }

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                _ = cu_module_unload(module);
            }
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .module = module,
            .function = function,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                _ = cu_module_unload(module);
            }
            return interface.KernelError.CompileFailed;
        };

        return function.?;
    }

    /// Launch a compiled kernel.
    pub fn launchKernel(
        self: *Self,
        kernel: *anyopaque,
        config: interface.LaunchConfig,
        args: []const *anyopaque,
    ) interface.KernelError!void {
        const funcs = self.functions orelse return interface.KernelError.LaunchFailed;
        const cu_launch_kernel = funcs.kernel.cuLaunchKernel orelse return interface.KernelError.LaunchFailed;

        // Validate configuration
        if (config.block_x == 0 or config.block_y == 0 or config.block_z == 0) {
            return interface.KernelError.InvalidConfig;
        }
        if (config.grid_x == 0 or config.grid_y == 0 or config.grid_z == 0) {
            return interface.KernelError.InvalidConfig;
        }

        // Convert args slice to optional pointers array for CUDA
        // CUDA expects a pointer to an array of pointers to arguments
        var cuda_args: [32]?*anyopaque = .{null} ** 32;
        const arg_count = @min(args.len, 32);
        for (0..arg_count) |i| {
            cuda_args[i] = @constCast(args[i]);
        }

        const result = cu_launch_kernel(
            kernel,
            config.grid_x,
            config.grid_y,
            config.grid_z,
            config.block_x,
            config.block_y,
            config.block_z,
            config.shared_memory,
            config.stream, // null for default stream
            &cuda_args,
            null, // extra (not used)
        );

        if (result != .success) {
            return interface.KernelError.LaunchFailed;
        }
    }

    /// Destroy a compiled kernel.
    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        const funcs = self.functions orelse return;

        for (self.kernels.items, 0..) |k, i| {
            if (k.function == kernel) {
                if (funcs.kernel.cuModuleUnload) |cu_module_unload| {
                    if (k.module) |module| {
                        _ = cu_module_unload(module);
                    }
                }
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }
};

/// Create a VTable-wrapped CUDA backend for the interface system.
///
/// Returns a Backend interface that can be used polymorphically with other backends.
/// On systems without CUDA, returns BackendError.NotAvailable.
pub fn createCudaVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try CudaBackend.init(allocator, 0);
    return interface.createBackend(CudaBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "CudaBackend initialization" {
    const allocator = std.testing.allocator;

    // Should create backend or return NotAvailable if no CUDA
    const result = CudaBackend.init(allocator, 0);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.device_id == 0);
    } else |err| {
        // Expected on systems without CUDA
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed or err == error.DeviceNotFound);
    }
}

test "CudaBackend invalid device id" {
    const allocator = std.testing.allocator;

    // First check if CUDA is available at all
    const valid_result = CudaBackend.init(allocator, 0);
    if (valid_result) |backend| {
        backend.deinit();

        // Now try with an invalid device ID
        const invalid_result = CudaBackend.init(allocator, 9999);
        if (invalid_result) |invalid_backend| {
            // Shouldn't happen with device 9999
            invalid_backend.deinit();
            try std.testing.expect(false);
        } else |err| {
            try std.testing.expect(err == error.DeviceNotFound);
        }
    } else |_| {
        // CUDA not available, skip test
        return error.SkipZigTest;
    }
}

test "CudaBackend device count" {
    const allocator = std.testing.allocator;

    const result = CudaBackend.init(allocator, 0);
    if (result) |backend| {
        defer backend.deinit();
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 1);
    } else |_| {
        // CUDA not available, skip test
        return error.SkipZigTest;
    }
}

test "CudaBackend device caps" {
    const allocator = std.testing.allocator;

    const result = CudaBackend.init(allocator, 0);
    if (result) |backend| {
        defer backend.deinit();
        const caps = try backend.getDeviceCaps(0);
        // Basic sanity checks
        try std.testing.expect(caps.max_threads_per_block > 0);
        try std.testing.expect(caps.warp_size > 0);
        try std.testing.expect(caps.total_memory > 0);
    } else |_| {
        // CUDA not available, skip test
        return error.SkipZigTest;
    }
}
