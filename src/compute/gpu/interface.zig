//! GPU Backend Interface
//!
//! Unified interface for all GPU backends (CUDA, Vulkan, Metal, etc.)
//! Enables runtime polymorphism and consistent behavior across backends.

const std = @import("std");

pub const BackendError = error{
    InitFailed,
    NotAvailable,
    DeviceNotFound,
    OutOfMemory,
    KernelCompileFailed,
    KernelLaunchFailed,
    InvalidOperation,
    Timeout,
};

pub const MemoryError = error{
    OutOfMemory,
    InvalidPointer,
    InvalidSize,
    TransferFailed,
};

pub const KernelError = error{
    CompileFailed,
    LaunchFailed,
    InvalidConfig,
    InvalidArgs,
};

/// Device capabilities
pub const DeviceCaps = struct {
    name: [256]u8 = undefined,
    name_len: usize = 0,
    total_memory: usize = 0,
    compute_capability_major: u32 = 0,
    compute_capability_minor: u32 = 0,
    max_threads_per_block: u32 = 512,
    max_shared_memory: u32 = 32768,
    warp_size: u32 = 32,
    supports_fp16: bool = false,
    supports_fp64: bool = false,
    supports_int8: bool = false,
    unified_memory: bool = false,
    async_engine_count: u32 = 0,

    pub fn getName(self: *const DeviceCaps) []const u8 {
        return self.name[0..self.name_len];
    }
};

/// Memory allocation flags
pub const MemoryFlags = packed struct {
    device: bool = true,
    host_visible: bool = false,
    host_coherent: bool = false,
    cached: bool = false,
    _padding: u4 = 0,
};

/// Kernel launch configuration
pub const LaunchConfig = struct {
    grid_x: u32 = 1,
    grid_y: u32 = 1,
    grid_z: u32 = 1,
    block_x: u32 = 256,
    block_y: u32 = 1,
    block_z: u32 = 1,
    shared_memory: u32 = 0,
    stream: ?*anyopaque = null,

    pub fn threads(self: *const LaunchConfig) u64 {
        return @as(u64, self.grid_x) * self.grid_y * self.grid_z *
            self.block_x * self.block_y * self.block_z;
    }
};

/// Backend type identifier
pub const BackendType = enum {
    cuda,
    vulkan,
    metal,
    webgpu,
    opengl,
    opengles,
    webgl2,
    stdgpu,
    simulated,

    pub fn name(self: BackendType) []const u8 {
        return switch (self) {
            .cuda => "cuda",
            .vulkan => "vulkan",
            .metal => "metal",
            .webgpu => "webgpu",
            .opengl => "opengl",
            .opengles => "opengles",
            .webgl2 => "webgl2",
            .stdgpu => "stdgpu",
            .simulated => "simulated",
        };
    }
};

/// GPU Backend interface (VTable pattern)
pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        // Lifecycle
        deinit: *const fn (*anyopaque) void,

        // Device info
        getDeviceCount: *const fn (*anyopaque) u32,
        getDeviceCaps: *const fn (*anyopaque, u32) BackendError!DeviceCaps,

        // Memory operations
        allocate: *const fn (*anyopaque, usize, MemoryFlags) MemoryError!*anyopaque,
        free: *const fn (*anyopaque, *anyopaque) void,
        copyToDevice: *const fn (*anyopaque, *anyopaque, []const u8) MemoryError!void,
        copyFromDevice: *const fn (*anyopaque, []u8, *anyopaque) MemoryError!void,

        // Kernel operations
        compileKernel: *const fn (*anyopaque, std.mem.Allocator, []const u8, []const u8) KernelError!*anyopaque,
        launchKernel: *const fn (*anyopaque, *anyopaque, LaunchConfig, []const *anyopaque) KernelError!void,
        destroyKernel: *const fn (*anyopaque, *anyopaque) void,

        // Synchronization
        synchronize: *const fn (*anyopaque) BackendError!void,
    };

    pub fn deinit(self: Backend) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn getDeviceCount(self: Backend) u32 {
        return self.vtable.getDeviceCount(self.ptr);
    }

    pub fn getDeviceCaps(self: Backend, device_id: u32) BackendError!DeviceCaps {
        return self.vtable.getDeviceCaps(self.ptr, device_id);
    }

    pub fn allocate(self: Backend, size: usize, flags: MemoryFlags) MemoryError!*anyopaque {
        return self.vtable.allocate(self.ptr, size, flags);
    }

    pub fn free(self: Backend, ptr: *anyopaque) void {
        self.vtable.free(self.ptr, ptr);
    }

    pub fn copyToDevice(self: Backend, dst: *anyopaque, src: []const u8) MemoryError!void {
        return self.vtable.copyToDevice(self.ptr, dst, src);
    }

    pub fn copyFromDevice(self: Backend, dst: []u8, src: *anyopaque) MemoryError!void {
        return self.vtable.copyFromDevice(self.ptr, dst, src);
    }

    pub fn compileKernel(self: Backend, allocator: std.mem.Allocator, source: []const u8, name: []const u8) KernelError!*anyopaque {
        return self.vtable.compileKernel(self.ptr, allocator, source, name);
    }

    pub fn launchKernel(self: Backend, kernel: *anyopaque, config: LaunchConfig, args: []const *anyopaque) KernelError!void {
        return self.vtable.launchKernel(self.ptr, kernel, config, args);
    }

    pub fn destroyKernel(self: Backend, kernel: *anyopaque) void {
        self.vtable.destroyKernel(self.ptr, kernel);
    }

    pub fn synchronize(self: Backend) BackendError!void {
        return self.vtable.synchronize(self.ptr);
    }
};

/// Backend registry for runtime selection
pub const Registry = struct {
    backends: std.StringHashMapUnmanaged(BackendInfo),
    allocator: std.mem.Allocator,

    pub const BackendInfo = struct {
        backend_type: BackendType,
        init_fn: *const fn (std.mem.Allocator) BackendError!Backend,
        available: bool,
        priority: u32,
    };

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .backends = std.StringHashMapUnmanaged(BackendInfo).empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Registry) void {
        self.backends.deinit(self.allocator);
    }

    pub fn register(self: *Registry, name: []const u8, info: BackendInfo) !void {
        try self.backends.put(self.allocator, name, info);
    }

    pub fn get(self: *const Registry, name: []const u8) ?BackendInfo {
        return self.backends.get(name);
    }

    pub fn getBest(self: *const Registry) ?BackendInfo {
        var best: ?BackendInfo = null;
        var it = self.backends.valueIterator();
        while (it.next()) |info| {
            if (info.available) {
                if (best == null or info.priority > best.?.priority) {
                    best = info.*;
                }
            }
        }
        return best;
    }
};

/// Create a backend implementation wrapper
pub fn createBackend(
    comptime Impl: type,
    impl: *Impl,
) Backend {
    const gen = struct {
        fn deinit(ptr: *anyopaque) void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            self.deinit();
        }

        fn getDeviceCount(ptr: *anyopaque) u32 {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.getDeviceCount();
        }

        fn getDeviceCaps(ptr: *anyopaque, device_id: u32) BackendError!DeviceCaps {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.getDeviceCaps(device_id);
        }

        fn allocate(ptr: *anyopaque, size: usize, flags: MemoryFlags) MemoryError!*anyopaque {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.allocate(size, flags);
        }

        fn free(ptr: *anyopaque, mem: *anyopaque) void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            self.free(mem);
        }

        fn copyToDevice(ptr: *anyopaque, dst: *anyopaque, src: []const u8) MemoryError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyToDevice(dst, src);
        }

        fn copyFromDevice(ptr: *anyopaque, dst: []u8, src: *anyopaque) MemoryError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyFromDevice(dst, src);
        }

        fn compileKernel(ptr: *anyopaque, allocator: std.mem.Allocator, source: []const u8, kernel_name: []const u8) KernelError!*anyopaque {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.compileKernel(allocator, source, kernel_name);
        }

        fn launchKernel(ptr: *anyopaque, kernel: *anyopaque, config: LaunchConfig, args: []const *anyopaque) KernelError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.launchKernel(kernel, config, args);
        }

        fn destroyKernel(ptr: *anyopaque, kernel: *anyopaque) void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            self.destroyKernel(kernel);
        }

        fn synchronize(ptr: *anyopaque) BackendError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.synchronize();
        }

        const vtable = Backend.VTable{
            .deinit = deinit,
            .getDeviceCount = getDeviceCount,
            .getDeviceCaps = getDeviceCaps,
            .allocate = allocate,
            .free = free,
            .copyToDevice = copyToDevice,
            .copyFromDevice = copyFromDevice,
            .compileKernel = compileKernel,
            .launchKernel = launchKernel,
            .destroyKernel = destroyKernel,
            .synchronize = synchronize,
        };
    };

    return .{
        .ptr = impl,
        .vtable = &gen.vtable,
    };
}

test "device caps" {
    var caps = DeviceCaps{};
    const name = "Test GPU";
    @memcpy(caps.name[0..name.len], name);
    caps.name_len = name.len;

    try std.testing.expectEqualStrings("Test GPU", caps.getName());
}

test "launch config" {
    const config = LaunchConfig{
        .grid_x = 4,
        .grid_y = 2,
        .block_x = 128,
    };
    try std.testing.expectEqual(@as(u64, 4 * 2 * 128), config.threads());
}
