//! GPU Backend Interface
//!
//! Unified interface for all GPU backends (CUDA, Vulkan, Metal, etc.)
//! Enables runtime polymorphism and consistent behavior across backends.

const std = @import("std");

// =============================================================================
// Standard GPU Error Types
// =============================================================================
// All backends should use these error types for consistency.
// Backend-specific errors should be mapped to these standard errors.

/// Backend lifecycle and device operation errors.
pub const BackendError = error{
    InitFailed,
    NotAvailable,
    DeviceNotFound,
    OutOfMemory,
    KernelCompileFailed,
    KernelLaunchFailed,
    InvalidOperation,
    Timeout,
    DriverNotFound,
    ContextCreationFailed,
    SynchronizationFailed,
};

/// Memory allocation and transfer errors.
pub const MemoryError = error{
    OutOfMemory,
    InvalidPointer,
    InvalidSize,
    TransferFailed,
    AllocationFailed,
    FreeFailed,
    BufferTooSmall,
    HostAccessDisabled,
    DeviceMemoryMissing,
    SizeMismatch,
};

/// Kernel compilation and execution errors.
pub const KernelError = error{
    CompileFailed,
    LaunchFailed,
    InvalidConfig,
    InvalidArgs,
    KernelNotFound,
    UnsupportedKernel,
    ArgumentCountMismatch,
};

/// Interface-level errors for stub implementations and feature detection.
pub const InterfaceError = error{
    NotImplemented,
    FeatureDisabled,
    UnsupportedOperation,
    UnsupportedBackend,
};

/// Unified GPU error type combining all error categories.
/// Use this when a function may return errors from multiple categories.
pub const GpuError = BackendError || MemoryError || KernelError || InterfaceError;

/// Maps backend-specific errors to standard GPU errors.
/// Backends should use this to convert their native errors.
pub fn mapToStandardError(err: anyerror) GpuError {
    // Check if already a standard error
    inline for (@typeInfo(GpuError).error_set.?) |e| {
        if (err == @field(anyerror, e.name)) {
            return @errorCast(err);
        }
    }
    // Default mapping for unknown errors
    return error.InvalidOperation;
}

/// Normalize backend-specific error naming to canonical `BackendError` values.
///
/// This is used at backend boundaries to absorb naming drift such as
/// `InitializationFailed` vs `InitFailed` and backend-specific "not available"
/// tags.
pub fn normalizeBackendError(err: anyerror) BackendError {
    if (err == error.NotAvailable or
        err == error.ValidationLayerNotAvailable or
        err == error.StdGpuNotAvailable or
        err == error.WebGpuNotAvailable or
        err == error.CudaNotAvailable or
        err == error.BankNotAvailable)
    {
        return error.NotAvailable;
    }
    if (err == error.DeviceNotFound) return error.DeviceNotFound;
    if (err == error.DriverNotFound) return error.DriverNotFound;
    if (err == error.OutOfMemory) return error.OutOfMemory;
    if (err == error.InitFailed or err == error.InitializationFailed or err == error.NotInitialized) {
        return error.InitFailed;
    }
    if (err == error.KernelCompileFailed or err == error.CompileFailed or err == error.ShaderCompilationFailed or err == error.CompilationFailed) {
        return error.KernelCompileFailed;
    }
    if (err == error.KernelLaunchFailed or err == error.LaunchFailed or err == error.ExecutionFailed or err == error.KernelExecutionFailed) {
        return error.KernelLaunchFailed;
    }
    if (err == error.Timeout) return error.Timeout;
    if (err == error.SynchronizationFailed) return error.SynchronizationFailed;

    const name = @errorName(err);
    if (std.mem.eql(u8, name, "NotAvailable") or
        std.mem.eql(u8, name, "ValidationLayerNotAvailable") or
        std.mem.eql(u8, name, "StdGpuNotAvailable") or
        std.mem.eql(u8, name, "WebGpuNotAvailable") or
        std.mem.eql(u8, name, "CudaNotAvailable") or
        std.mem.eql(u8, name, "BankNotAvailable"))
    {
        return error.NotAvailable;
    }
    if (std.mem.eql(u8, name, "DeviceNotFound")) return error.DeviceNotFound;
    if (std.mem.eql(u8, name, "DriverNotFound")) return error.DriverNotFound;
    if (std.mem.eql(u8, name, "OutOfMemory")) return error.OutOfMemory;
    if (std.mem.eql(u8, name, "InitFailed") or
        std.mem.eql(u8, name, "InitializationFailed") or
        std.mem.eql(u8, name, "NotInitialized"))
    {
        return error.InitFailed;
    }
    if (std.mem.eql(u8, name, "KernelCompileFailed") or
        std.mem.eql(u8, name, "CompileFailed") or
        std.mem.eql(u8, name, "ShaderCompilationFailed") or
        std.mem.eql(u8, name, "CompilationFailed"))
    {
        return error.KernelCompileFailed;
    }
    if (std.mem.eql(u8, name, "KernelLaunchFailed") or
        std.mem.eql(u8, name, "LaunchFailed") or
        std.mem.eql(u8, name, "ExecutionFailed") or
        std.mem.eql(u8, name, "KernelExecutionFailed"))
    {
        return error.KernelLaunchFailed;
    }
    if (std.mem.eql(u8, name, "Timeout")) return error.Timeout;
    if (std.mem.eql(u8, name, "SynchronizationFailed")) return error.SynchronizationFailed;

    return error.InitFailed;
}

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

    // Extended capabilities (CUDA sm_XX / Metal GPU Family)
    supports_bf16: bool = false,
    supports_tf32: bool = false,
    supports_fp8: bool = false,
    supports_tensor_cores: bool = false,
    supports_int8_tensor_cores: bool = false,
    supports_cooperative_groups: bool = false,
    supports_dynamic_parallelism: bool = false,
    supports_async_copy: bool = false,
    managed_memory: bool = false,
    supports_mesh_shaders: bool = false,
    supports_ray_tracing: bool = false,
    supports_neural_engine: bool = false,
    supports_mps: bool = false,
    metal_gpu_family: u32 = 0,

    // Hardware topology
    sm_count: u32 = 0,
    memory_clock_rate_khz: u32 = 0,
    memory_bus_width: u32 = 0,
    architecture_name: [64]u8 = .{0} ** 64,
    architecture_name_len: usize = 0,

    pub fn getName(self: *const DeviceCaps) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn getArchitectureName(self: *const DeviceCaps) []const u8 {
        return self.architecture_name[0..self.architecture_name_len];
    }

    /// Estimate memory bandwidth in GB/s from clock rate and bus width.
    pub fn memoryBandwidthGBps(self: *const DeviceCaps) f64 {
        if (self.memory_clock_rate_khz == 0 or self.memory_bus_width == 0)
            return 0;
        // bandwidth = 2 * clock_rate_hz * bus_width_bytes / 1e9
        const clock_hz: f64 = @as(f64, @floatFromInt(self.memory_clock_rate_khz)) * 1000.0;
        const bus_bytes: f64 = @as(f64, @floatFromInt(self.memory_bus_width)) / 8.0;
        return 2.0 * clock_hz * bus_bytes / 1_000_000_000.0;
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

/// Backend type identifier — canonical definition in backend.zig
pub const BackendType = @import("backend.zig").Backend;

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
        copyToDeviceAsync: *const fn (*anyopaque, *anyopaque, []const u8, ?*anyopaque) MemoryError!void,
        copyFromDeviceAsync: *const fn (*anyopaque, []u8, *anyopaque, ?*anyopaque) MemoryError!void,

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

    pub fn copyToDeviceAsync(self: Backend, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) MemoryError!void {
        return self.vtable.copyToDeviceAsync(self.ptr, dst, src, stream);
    }

    pub fn copyFromDeviceAsync(self: Backend, dst: []u8, src: *anyopaque, stream: ?*anyopaque) MemoryError!void {
        return self.vtable.copyFromDeviceAsync(self.ptr, dst, src, stream);
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

        fn copyToDeviceAsync(ptr: *anyopaque, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) MemoryError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyToDeviceAsync(dst, src, stream);
        }

        fn copyFromDeviceAsync(ptr: *anyopaque, dst: []u8, src: *anyopaque, stream: ?*anyopaque) MemoryError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyFromDeviceAsync(dst, src, stream);
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
            .copyToDeviceAsync = copyToDeviceAsync,
            .copyFromDeviceAsync = copyFromDeviceAsync,
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

test {
    std.testing.refAllDecls(@This());
}
