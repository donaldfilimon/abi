//! GPU backend detection, kernel management, and memory utilities.
const std = @import("std");
const backend = @import("backend.zig");
const kernels = @import("kernels.zig");
const memory = @import("memory.zig");

const build_options = @import("build_options");

const SimpleModuleLifecycle = struct {
    const Self = @This();
    initialized: bool = false,

    pub fn init(self: *Self, init_fn: fn () anyerror!void) !void {
        if (self.initialized) {
            return;
        }
        try init_fn();
        self.initialized = true;
        return;
    }

    pub fn ensureInitialized(self: *Self, init_fn: fn () anyerror!void) !void {
        if (self.isInitialized()) {
            return;
        }
        return self.init(init_fn);
    }

    pub fn deinit(self: *Self, deinit_fn: ?fn () void) void {
        if (!self.initialized) {
            return;
        }
        if (deinit_fn) |fn_ptr| {
            fn_ptr();
        }
        self.initialized = false;
    }

    pub fn isInitialized(self: *Self) bool {
        return self.initialized;
    }
};

var gpu_lifecycle = SimpleModuleLifecycle{};

var cuda_backend_initialized = false;

pub const MemoryError = memory.MemoryError;
pub const BufferFlags = memory.BufferFlags;
pub const GPUBuffer = memory.GPUBuffer;
pub const GPUMemoryPool = memory.GPUMemoryPool;
pub const MemoryStats = memory.MemoryStats;
pub const AsyncTransfer = memory.AsyncTransfer;
pub const GpuError = memory.MemoryError || error{GpuDisabled};

pub const KernelSource = kernels.KernelSource;
pub const KernelConfig = kernels.KernelConfig;
pub const CompiledKernel = kernels.CompiledKernel;
pub const Stream = kernels.Stream;
pub const KernelError = kernels.KernelError;
pub const compileKernel = kernels.compileKernel;
pub const createDefaultKernels = kernels.createDefaultKernels;

pub const Backend = backend.Backend;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const BackendInfo = backend.BackendInfo;
pub const DeviceCapability = backend.DeviceCapability;
pub const DeviceInfo = backend.DeviceInfo;
pub const Summary = backend.Summary;
pub const backendName = backend.backendName;
pub const backendDisplayName = backend.backendDisplayName;
pub const backendDescription = backend.backendDescription;
pub const backendFlag = backend.backendFlag;
pub const backendFromString = backend.backendFromString;
pub const backendSupportsKernels = backend.backendSupportsKernels;
pub const backendAvailability = backend.backendAvailability;
pub const availableBackends = backend.availableBackends;
pub const listBackendInfo = backend.listBackendInfo;
pub const listDevices = backend.listDevices;
pub const defaultDevice = backend.defaultDevice;
pub const defaultDeviceLabel = backend.defaultDeviceLabel;
pub const summary = backend.summary;
pub const moduleEnabled = backend.moduleEnabled;
pub const isEnabled = backend.isEnabled;

pub fn init(_: std.mem.Allocator) GpuError!void {
    if (!moduleEnabled()) return error.GpuDisabled;

    gpu_lifecycle.init(initCudaComponents) catch {
        return error.GpuDisabled;
    };
}

fn initCudaComponents() !void {
    if (comptime build_options.gpu_cuda) {
        if (!cuda_backend_initialized) {
            const cuda_module = @import("backends/cuda.zig");

            cuda_module.init() catch |err| {
                std.log.warn("CUDA backend initialization failed: {}. Using fallback mode.", .{err});
            };

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda_stream.zig");
                cuda_stream.init() catch |err| {
                    std.log.warn("CUDA stream initialization failed: {}", .{err});
                };

                const cuda_memory = @import("backends/cuda_memory.zig");
                cuda_memory.init() catch |err| {
                    std.log.warn("CUDA memory initialization failed: {}", .{err});
                };
            }

            cuda_backend_initialized = true;
        }
    }
}

fn deinitCudaComponents() void {
    if (cuda_backend_initialized) {
        if (comptime build_options.gpu_cuda) {
            const cuda_module = @import("backends/cuda.zig");
            cuda_module.deinit();

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda_stream.zig");
                cuda_stream.deinit();

                const cuda_memory = @import("backends/cuda_memory.zig");
                cuda_memory.deinit();
            }
        }
        cuda_backend_initialized = false;
    }
}

pub fn ensureInitialized(allocator: std.mem.Allocator) GpuError!void {
    if (!isInitialized()) {
        try init(allocator);
    }
}

pub fn deinit() void {
    deinitCudaComponents();
    gpu_lifecycle.deinit(null);
}

pub fn isInitialized() bool {
    return gpu_lifecycle.isInitialized();
}
