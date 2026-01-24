//! OpenGL ES VTable Backend Implementation
//!
//! Provides a complete VTable implementation for OpenGL ES, enabling GPU
//! kernel execution through the polymorphic backend interface.
//! Requires OpenGL ES 3.1+ for compute shader support (mobile/embedded).

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const opengles = @import("opengles.zig");

/// OpenGL ES VTable Backend
///
/// Wraps the existing OpenGL ES backend implementation to provide the
/// unified Backend interface for polymorphic GPU operations.
pub const OpenGLESBackend = struct {
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

    /// Initialize the OpenGL ES VTable backend.
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Check if OpenGL ES is enabled at compile time
        if (comptime !build_options.gpu_opengles) {
            return interface.BackendError.NotAvailable;
        }

        // Try to initialize the OpenGL ES backend
        opengles.init() catch |err| {
            return switch (err) {
                opengles.OpenGlesError.LibraryNotFound => interface.BackendError.DriverNotFound,
                opengles.OpenGlesError.VersionNotSupported => interface.BackendError.NotAvailable,
                opengles.OpenGlesError.FunctionLoadFailed => interface.BackendError.InitFailed,
                else => interface.BackendError.InitFailed,
            };
        };

        const self = allocator.create(Self) catch {
            opengles.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        // Set device name based on OpenGL ES version
        const version = opengles.getVersion();
        const name = std.fmt.bufPrint(&self.device_name, "OpenGL ES {}.{} Compute Device", .{
            version.major,
            version.minor,
        }) catch "OpenGL ES Compute Device";
        self.device_name_len = name.len;

        // Set the buffer allocator for the OpenGL ES backend
        opengles.setBufferAllocator(allocator);

        return self;
    }

    /// Deinitialize the backend and release all resources.
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        for (self.allocations.items) |alloc| {
            opengles.freeDeviceMemory(alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        // Destroy all kernels
        for (self.kernels.items) |kernel| {
            opengles.destroyKernel(self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        // Deinit the OpenGL ES backend
        if (self.initialized) {
            opengles.deinit();
        }

        self.allocator.destroy(self);
    }

    // ========================================================================
    // Device Information
    // ========================================================================

    pub fn getDeviceCount(_: *Self) u32 {
        // OpenGL ES typically has only one device (the active GPU)
        // In multi-GPU systems, ES is usually tied to the primary GPU
        return 1;
    }

    pub fn getDeviceCaps(_: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) {
            return interface.BackendError.DeviceNotFound;
        }

        var caps = interface.DeviceCaps{};

        // Copy cached device name
        const name = "OpenGL ES Device (Mobile/Embedded)";
        const copy_len = @min(name.len, caps.name.len);
        @memcpy(caps.name[0..copy_len], name[0..copy_len]);
        caps.name_len = copy_len;

        // Query OpenGL ES capabilities
        const version = opengles.getVersion();
        const has_compute = version.major >= 3 and version.minor >= 1;

        // OpenGL ES compute limitations
        caps.total_memory = 0; // Unknown on mobile/embedded
        caps.max_threads_per_block = 128; // Lower than desktop GL
        caps.max_shared_memory = 16 * 1024; // 16KB typical on mobile
        caps.warp_size = 1; // No warps in GLES
        caps.supports_fp16 = true; // Many mobile GPUs support FP16
        caps.supports_fp64 = false; // Rare on mobile/embedded
        caps.supports_int8 = true; // Support for quantized ops
        caps.unified_memory = true; // UMA common on mobile SoCs
        caps.compute_capability_major = @intCast(version.major);
        caps.compute_capability_minor = @intCast(version.minor);
        caps.async_engine_count = if (has_compute) 1 else 0;

        return caps;
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags; // TODO: Use memory flags for GLES memory type selection

        const ptr = opengles.allocateDeviceMemory(size) catch {
            return interface.MemoryError.AllocationFailed;
        };

        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            opengles.freeDeviceMemory(ptr);
            return interface.MemoryError.AllocationFailed;
        };

        return ptr;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        // Remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                opengles.freeDeviceMemory(ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(_: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        opengles.copyToDevice(dst, src) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    pub fn copyFromDevice(_: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        opengles.copyFromDevice(dst, src) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream; // OpenGL ES doesn't have async transfers in compute context
        return self.copyToDevice(dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyFromDevice(dst, src);
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, kernel_name: []const u8) interface.KernelError!*anyopaque {
        // OpenGL ES uses GLSL ES compute shaders
        const kernel_handle = opengles.compileKernel(allocator, source) catch {
            return interface.KernelError.CompileFailed;
        };

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            opengles.destroyKernel(allocator, kernel_handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = kernel_handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            opengles.destroyKernel(allocator, kernel_handle);
            return interface.KernelError.CompileFailed;
        };

        return kernel_handle;
    }

    pub fn launchKernel(self: *Self, kernel: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        _ = self;

        // Convert to OpenGL ES dispatch parameters
        // GLES compute shaders use workgroup counts (not threads)
        const workgroup_x = config.grid_x;
        const workgroup_y = config.grid_y;
        const workgroup_z = config.grid_z;

        opengles.launchKernel(kernel, workgroup_x, workgroup_y, workgroup_z, args) catch {
            return interface.KernelError.LaunchFailed;
        };
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        // Remove from tracking
        for (self.kernels.items, 0..) |k, i| {
            if (k.handle == kernel) {
                _ = self.kernels.swapRemove(i);
                self.allocator.free(k.name);
                break;
            }
        }

        opengles.destroyKernel(self.allocator, kernel);
    }

    pub fn synchronize(_: *Self) interface.BackendError!void {
        opengles.synchronize() catch {
            return interface.BackendError.SynchronizationFailed;
        };
    }
};

/// Creates an OpenGL ES backend instance wrapped in the VTable interface.
///
/// Returns BackendError.NotAvailable if OpenGL ES is disabled at compile time
/// or the OpenGL ES driver cannot be loaded.
/// Returns BackendError.InitFailed if OpenGL ES initialization fails.
pub fn createOpenGLESVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try OpenGLESBackend.init(allocator);
    return interface.createBackend(OpenGLESBackend, impl);
}
