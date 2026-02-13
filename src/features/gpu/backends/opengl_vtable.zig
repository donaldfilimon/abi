//! OpenGL VTable Backend Implementation
//!
//! Provides a complete VTable implementation for OpenGL, enabling GPU
//! kernel execution through the polymorphic backend interface.
//! Requires OpenGL 4.3+ for compute shader support.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const opengl = @import("opengl.zig");

/// OpenGL VTable Backend
///
/// Wraps the existing OpenGL backend implementation to provide the
/// unified Backend interface for polymorphic GPU operations.
pub const OpenGLBackend = struct {
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

    /// Initialize the OpenGL VTable backend.
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Check if OpenGL is enabled at compile time
        if (comptime !build_options.gpu_opengl) {
            return interface.BackendError.NotAvailable;
        }

        // Try to initialize the OpenGL backend
        opengl.init() catch |err| {
            return switch (err) {
                opengl.OpenGlError.LibraryNotFound => interface.BackendError.DriverNotFound,
                opengl.OpenGlError.VersionNotSupported => interface.BackendError.NotAvailable,
                opengl.OpenGlError.FunctionLoadFailed => interface.BackendError.InitFailed,
                else => interface.BackendError.InitFailed,
            };
        };

        const self = allocator.create(Self) catch {
            opengl.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        // Set device name based on OpenGL version
        const version = opengl.getVersion();
        const name = std.fmt.bufPrint(&self.device_name, "OpenGL {}.{} Compute Device", .{
            version.major,
            version.minor,
        }) catch "OpenGL Compute Device";
        self.device_name_len = name.len;

        // Set the buffer allocator for the OpenGL backend
        opengl.setBufferAllocator(allocator);

        return self;
    }

    /// Deinitialize the backend and release all resources.
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        for (self.allocations.items) |alloc| {
            opengl.freeDeviceMemory(alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        // Destroy all kernels
        for (self.kernels.items) |kernel| {
            opengl.destroyKernel(self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        // Deinit the OpenGL backend
        if (self.initialized) {
            opengl.deinit();
        }

        self.allocator.destroy(self);
    }

    // ========================================================================
    // Device Info
    // ========================================================================

    /// Get the number of available OpenGL devices.
    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        // OpenGL typically exposes one device per context
        if (opengl.isAvailable()) {
            return 1;
        }
        return 0;
    }

    /// Get device capabilities.
    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{};

        // Copy device name
        @memcpy(caps.name[0..self.device_name_len], self.device_name[0..self.device_name_len]);
        caps.name_len = self.device_name_len;

        // Get OpenGL version info
        const version = opengl.getVersion();
        caps.compute_capability_major = @intCast(version.major);
        caps.compute_capability_minor = @intCast(version.minor);

        // OpenGL compute shader defaults (conservative values)
        caps.max_threads_per_block = 1024;
        caps.max_shared_memory = 32768; // 32KB typical minimum
        caps.warp_size = 32; // Typical work group size unit

        // OpenGL 4.3+ capabilities
        caps.supports_fp16 = false; // Not universally supported
        caps.supports_fp64 = version.major >= 4; // GLSL 4.0+ supports double
        caps.supports_int8 = true;
        caps.unified_memory = false; // OpenGL uses explicit buffer transfers

        return caps;
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Allocate device memory.
    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags; // OpenGL handles memory type internally

        const ptr = opengl.allocateDeviceMemoryWithAllocator(self.allocator, size) catch {
            return interface.MemoryError.OutOfMemory;
        };

        // Track allocation
        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            opengl.freeDeviceMemory(ptr);
            return interface.MemoryError.OutOfMemory;
        };

        return ptr;
    }

    /// Free device memory.
    pub fn free(self: *Self, ptr: *anyopaque) void {
        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                opengl.freeDeviceMemory(ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    /// Copy data from host to device.
    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        opengl.memcpyHostToDevice(dst, @constCast(src.ptr), src.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    /// Copy data from device to host.
    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        opengl.memcpyDeviceToHost(dst.ptr, src, dst.len) catch {
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

    /// Compile a kernel from GLSL compute shader source.
    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const kernel_types = @import("../kernel_types.zig");
        const backend_mod = @import("../backend.zig");

        // Create kernel source for OpenGL (GLSL compute shader)
        const kernel_source = kernel_types.KernelSource{
            .source = source,
            .name = kernel_name,
            .entry_point = "main",
            .backend = backend_mod.Backend.opengl,
        };

        const handle = opengl.compileKernel(allocator, kernel_source) catch {
            return interface.KernelError.CompileFailed;
        };

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            opengl.destroyKernel(allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            opengl.destroyKernel(allocator, handle);
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
        const kernel_types = @import("../kernel_types.zig");

        // Validate configuration
        if (config.block_x == 0 or config.block_y == 0 or config.block_z == 0) {
            return interface.KernelError.InvalidConfig;
        }
        if (config.grid_x == 0 or config.grid_y == 0 or config.grid_z == 0) {
            return interface.KernelError.InvalidConfig;
        }

        // Convert interface LaunchConfig to kernel_types.KernelConfig
        const kernel_config = kernel_types.KernelConfig{
            .grid_dim = .{ config.grid_x, config.grid_y, config.grid_z },
            .block_dim = .{ config.block_x, config.block_y, config.block_z },
            .shared_memory_bytes = config.shared_memory,
        };

        // Convert args to optional pointers for OpenGL backend
        var opt_args: [32]?*const anyopaque = .{null} ** 32;
        const arg_count = @min(args.len, 32);
        for (0..arg_count) |i| {
            opt_args[i] = args[i];
        }

        opengl.launchKernel(
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
                opengl.destroyKernel(self.allocator, kernel);
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Synchronize the device (wait for all pending operations to complete).
    pub fn synchronize(self: *Self) interface.BackendError!void {
        _ = self;
        if (!opengl.isAvailable()) {
            return interface.BackendError.NotAvailable;
        }
        opengl.synchronize();
    }
};

/// Create a VTable-wrapped OpenGL backend for the interface system.
pub fn createOpenGLVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try OpenGLBackend.init(allocator);
    return interface.createBackend(OpenGLBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "OpenGLBackend initialization" {
    const allocator = std.testing.allocator;

    const result = OpenGLBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.initialized);
    } else |err| {
        // Expected on systems without OpenGL 4.3+
        try std.testing.expect(err == error.NotAvailable or
            err == error.InitFailed or
            err == error.DriverNotFound);
    }
}

test "createOpenGLVTable" {
    const allocator = std.testing.allocator;

    const result = createOpenGLVTable(allocator);
    if (result) |backend| {
        defer backend.deinit();
        // Should work through VTable interface
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 0);
    } else |err| {
        // Expected on systems without OpenGL 4.3+
        try std.testing.expect(err == error.NotAvailable or
            err == error.InitFailed or
            err == error.DriverNotFound);
    }
}
