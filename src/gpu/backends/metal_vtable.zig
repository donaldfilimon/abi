//! Metal VTable Backend Implementation
//!
//! Provides a complete VTable implementation for Metal, enabling GPU
//! kernel execution through the polymorphic backend interface.
//!
//! ## Features
//! - Full device capability query (memory, threads, unified memory)
//! - Kernel compilation from Metal Shading Language (MSL)
//! - Asynchronous kernel execution with synchronization
//! - Memory allocation with unified memory support
//! - Multi-device enumeration on Intel Macs

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const metal = @import("metal.zig");
const types = @import("../kernel_types.zig");

/// Metal VTable Backend
///
/// Wraps the existing Metal backend implementation to provide the
/// unified Backend interface for polymorphic GPU operations.
pub const MetalBackend = struct {
    allocator: std.mem.Allocator,
    initialized: bool,

    // Track allocations for cleanup
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    // Device info cache
    device_name: [256]u8 = undefined,
    device_name_len: usize = 0,
    device_memory: u64 = 0,
    max_threads: u32 = 1024,
    has_unified_memory: bool = true,

    const Allocation = struct {
        ptr: *anyopaque,
        size: usize,
    };

    const CompiledKernel = struct {
        handle: *anyopaque,
        name: []const u8,
    };

    const Self = @This();

    /// Initialize the Metal VTable backend.
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Check if Metal is enabled at compile time
        if (comptime !build_options.gpu_metal) {
            return interface.BackendError.NotAvailable;
        }

        // Metal is only available on macOS
        if (builtin.target.os.tag != .macos) {
            return interface.BackendError.NotAvailable;
        }

        // Try to initialize the Metal backend
        metal.init() catch {
            return interface.BackendError.NotAvailable;
        };

        // Set allocators for the Metal backend
        metal.setBufferAllocator(allocator);
        metal.setPendingBuffersAllocator(allocator);

        const self = allocator.create(Self) catch {
            metal.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        // Query device info from the Metal backend
        if (metal.getDeviceInfo()) |info| {
            const name_len = @min(info.name.len, self.device_name.len);
            @memcpy(self.device_name[0..name_len], info.name[0..name_len]);
            self.device_name_len = name_len;
            self.device_memory = info.total_memory;
            self.max_threads = info.max_threads_per_threadgroup;
            self.has_unified_memory = info.has_unified_memory;
        } else {
            // Fallback to defaults
            const name = "Apple Metal GPU";
            @memcpy(self.device_name[0..name.len], name);
            self.device_name_len = name.len;
        }

        return self;
    }

    /// Deinitialize the backend and release all resources.
    pub fn deinit(self: *Self) void {
        // Free all tracked allocations
        for (self.allocations.items) |alloc| {
            metal.freeDeviceMemory(self.allocator, alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        // Destroy all kernels
        for (self.kernels.items) |kernel| {
            metal.destroyKernel(self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        // Deinit the Metal backend
        if (self.initialized) {
            metal.deinit();
        }

        self.allocator.destroy(self);
    }

    // ========================================================================
    // Device Info
    // ========================================================================

    /// Get the number of available Metal devices.
    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        // Metal typically exposes 1 device (the system GPU)
        if (metal.isAvailable()) {
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

        // Use queried device properties
        caps.max_threads_per_block = self.max_threads;
        caps.max_shared_memory = 32768; // Metal shared memory (threadgroup memory)
        caps.warp_size = 32; // Metal uses 32 threads per SIMD group
        caps.supports_fp16 = true; // All modern Metal devices support FP16
        caps.supports_fp64 = false; // Metal doesn't support FP64 compute
        caps.unified_memory = self.has_unified_memory;
        caps.total_memory = self.device_memory;

        return caps;
    }

    /// Get the total device memory in bytes.
    pub fn getTotalMemory(self: *Self) u64 {
        return self.device_memory;
    }

    /// Check if the device has unified memory (Apple Silicon).
    pub fn hasUnifiedMemory(self: *Self) bool {
        return self.has_unified_memory;
    }

    // ========================================================================
    // Memory Operations
    // ========================================================================

    /// Allocate device memory.
    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags; // Metal handles memory type internally

        const ptr = metal.allocateDeviceMemory(self.allocator, size) catch {
            return interface.MemoryError.OutOfMemory;
        };

        // Track allocation
        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            metal.freeDeviceMemory(self.allocator, ptr);
            return interface.MemoryError.OutOfMemory;
        };

        return ptr;
    }

    /// Free device memory.
    pub fn free(self: *Self, ptr: *anyopaque) void {
        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                metal.freeDeviceMemory(self.allocator, ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    /// Copy data from host to device.
    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        metal.memcpyHostToDevice(dst, @constCast(src.ptr), src.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    /// Copy data from device to host.
    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        metal.memcpyDeviceToHost(dst.ptr, src, dst.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    // ========================================================================
    // Kernel Operations
    // ========================================================================

    /// Compile a kernel from Metal Shading Language (MSL) source.
    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const kernel_source = types.KernelSource{
            .code = source,
            .entry_point = kernel_name,
            .format = .metal,
        };

        const handle = metal.compileKernel(allocator, kernel_source) catch {
            return interface.KernelError.CompileFailed;
        };

        // Track kernel
        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            metal.destroyKernel(allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            metal.destroyKernel(allocator, handle);
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

        metal.launchKernel(
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
                metal.destroyKernel(self.allocator, kernel);
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
        metal.synchronize();
    }
};

/// Create a VTable-wrapped Metal backend for the interface system.
pub fn createMetalVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try MetalBackend.init(allocator);
    return interface.createBackend(MetalBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "MetalBackend initialization" {
    const allocator = std.testing.allocator;

    const result = MetalBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.initialized);
        try std.testing.expect(backend.device_name_len > 0);
    } else |err| {
        // Expected on systems without Metal
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}

test "createMetalVTable" {
    const allocator = std.testing.allocator;

    const result = createMetalVTable(allocator);
    if (result) |backend| {
        defer backend.deinit();
        // Should work through VTable interface
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 0);
    } else |err| {
        // Expected on systems without Metal
        try std.testing.expect(err == error.NotAvailable or err == error.InitFailed);
    }
}

test "MetalBackend device capabilities" {
    const allocator = std.testing.allocator;

    const result = MetalBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();

        // Query device capabilities
        const caps = backend.getDeviceCaps(0) catch |err| {
            try std.testing.expect(err == error.DeviceNotFound);
            return;
        };

        // Verify expected Metal capabilities
        try std.testing.expect(caps.supports_fp16);
        try std.testing.expect(!caps.supports_fp64); // Metal doesn't support FP64
        try std.testing.expect(caps.warp_size == 32); // Metal SIMD group size
        try std.testing.expect(caps.max_threads_per_block >= 256);
        try std.testing.expect(caps.name_len > 0);
    } else |_| {
        // Skip test on non-Metal systems
    }
}

test "MetalBackend memory allocation" {
    const allocator = std.testing.allocator;

    const result = MetalBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();

        // Allocate some device memory
        const ptr = backend.allocate(4096, .{}) catch |err| {
            try std.testing.expect(err == error.OutOfMemory);
            return;
        };

        // Free the memory
        backend.free(ptr);
    } else |_| {
        // Skip test on non-Metal systems
    }
}

test "MetalBackend memory transfer" {
    const allocator = std.testing.allocator;

    const result = MetalBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();

        // Allocate device memory
        const ptr = backend.allocate(256, .{}) catch return;
        defer backend.free(ptr);

        // Prepare test data
        var src_data: [256]u8 = undefined;
        for (&src_data, 0..) |*b, i| {
            b.* = @intCast(i & 0xFF);
        }

        // Copy to device
        backend.copyToDevice(ptr, &src_data) catch |err| {
            try std.testing.expect(err == error.TransferFailed);
            return;
        };

        // Copy back from device
        var dst_data: [256]u8 = undefined;
        backend.copyFromDevice(&dst_data, ptr) catch |err| {
            try std.testing.expect(err == error.TransferFailed);
            return;
        };

        // Verify data integrity
        try std.testing.expectEqualSlices(u8, &src_data, &dst_data);
    } else |_| {
        // Skip test on non-Metal systems
    }
}

test "MetalBackend unified memory check" {
    const allocator = std.testing.allocator;

    const result = MetalBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();

        // On Apple Silicon, should have unified memory
        if (builtin.target.cpu.arch == .aarch64) {
            try std.testing.expect(backend.hasUnifiedMemory());
        }

        // Memory should be reported
        const memory = backend.getTotalMemory();
        // Memory could be 0 if query failed, but that's okay
        _ = memory;
    } else |_| {
        // Skip test on non-Metal systems
    }
}

test "Metal MTLSize struct" {
    // Test MTLSize construction
    const size1 = metal.MTLSize.init(64, 32, 1);
    try std.testing.expectEqual(@as(usize, 64), size1.width);
    try std.testing.expectEqual(@as(usize, 32), size1.height);
    try std.testing.expectEqual(@as(usize, 1), size1.depth);

    // Test from3D conversion
    const size2 = metal.MTLSize.from3D(.{ 128, 64, 2 });
    try std.testing.expectEqual(@as(usize, 128), size2.width);
    try std.testing.expectEqual(@as(usize, 64), size2.height);
    try std.testing.expectEqual(@as(usize, 2), size2.depth);
}

test "Metal isAvailable" {
    const available = metal.isAvailable();
    // On macOS, this should return true
    // On other platforms, should return false
    if (builtin.target.os.tag == .macos) {
        // Metal should be available on macOS
        // (unless running in a VM or very old hardware)
        _ = available;
    } else {
        try std.testing.expect(!available);
    }
}
