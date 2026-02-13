//! CUDA backend implementation
//!
//! Provides CUDA-specific kernel compilation and execution.
//! Uses consolidated cuda_loader for function management.
//!
//! This module aggregates all CUDA functionality:
//! - loader.zig: Consolidated CUDA library loading
//! - native.zig: Native CUDA backend with real GPU execution
//! - memory.zig: Device and pinned memory management
//! - stream.zig: CUDA stream management
//! - device_query.zig: Device capability queries
//! - nvrtc.zig: Runtime compilation support

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const build_options = @import("build_options");
const types = @import("../../kernel_types.zig");
const shared = @import("../shared.zig");
const fallback = @import("../fallback.zig");
const cuda_native = @import("native.zig");
const cuda_loader = @import("loader.zig");
const gpu = std.gpu;

// Re-export submodules
pub const loader = @import("loader.zig");
pub const native = @import("native.zig");
pub const memory = @import("memory.zig");
pub const stream = @import("stream.zig");
pub const device_query = @import("device_query.zig");
pub const nvrtc = @import("nvrtc.zig");
pub const cublas = if (build_options.enable_gpu and build_options.gpu_cuda)
    @import("cublas.zig")
else
    struct {
        pub const CublasContext = void;
        pub fn isAvailable() bool {
            return false;
        }
    };
pub const llm_kernels = @import("llm_kernels.zig");
pub const quantized_kernels = @import("quantized_kernels.zig");
pub const vtable = @import("vtable.zig");

// VTable backend exports
pub const CudaBackend = vtable.CudaBackend;
pub const createCudaVTable = vtable.createCudaVTable;

// Quantized kernel exports
pub const QuantizedKernelModule = quantized_kernels.QuantizedKernelModule;
pub const QuantConfig = quantized_kernels.QuantConfig;

// Re-export from loader for compatibility
pub const CuResult = cuda_loader.CuResult;

const CuStream = extern struct {
    ptr: *anyopaque,
};

const CuEvent = extern struct {
    ptr: *anyopaque,
};

const CuFunction = extern struct {
    ptr: *anyopaque,
};

const CuModule = extern struct {
    ptr: *anyopaque,
};

pub const CudaError = cuda_native.CudaError;

const CudaContext = struct {
    device_id: i32,
    context: ?*anyopaque,
    stream: ?*anyopaque,
    device_memory: std.ArrayListUnmanaged([]u8),
};

var cuda_initialized = false;
var cuda_context: ?CudaContext = null;
var use_native: bool = false;
var init_mutex = sync.Mutex{};
var cached_allocator: ?std.mem.Allocator = null;

/// Initialize the CUDA backend and create context.
/// Returns CudaError if CUDA driver is not available or initialization fails.
pub fn init(allocator: std.mem.Allocator) CudaError!void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (cuda_initialized) return;

    cached_allocator = allocator;
    const cuda_available = tryLoadCuda(allocator);
    if (cuda_available) {
        if (loadCudaFunctions(allocator)) {
            const funcs = cuda_loader.getFunctions() orelse {
                std.log.warn("CUDA functions not loaded, using simulation mode", .{});
                return initSimulationMode();
            };

            if (funcs.core.cuInit) |init_fn| {
                const result = init_fn(0);
                if (result != .success) {
                    std.log.warn("CUDA initialization failed with code {t}, using simulation mode", .{result});
                    return initSimulationMode();
                }
            }

            var device_count: i32 = 0;
            if (funcs.core.cuDeviceGetCount) |count_fn| {
                const result = count_fn(&device_count);
                if (result != .success or device_count == 0) {
                    std.log.warn("No CUDA devices found, using simulation mode", .{});
                    return initSimulationMode();
                }
                std.log.info("CUDA initialized with {d} device(s)", .{device_count});
            }
        } else {
            std.log.warn("CUDA runtime functions not available, using simulation mode", .{});
            return initSimulationMode();
        }
    } else {
        std.log.warn("CUDA driver library not found, using simulation mode", .{});
        return initSimulationMode();
    }

    cuda_context = CudaContext{
        .device_id = 0,
        .context = null,
        .stream = null,
        .device_memory = std.ArrayListUnmanaged([]u8).empty,
    };

    cuda_initialized = true;
}

fn initSimulationMode() CudaError!void {
    cuda_context = CudaContext{
        .device_id = 0,
        .context = null,
        .stream = null,
        .device_memory = std.ArrayListUnmanaged([]u8).empty,
    };
    cuda_initialized = true;
}

/// Deinitialize the CUDA backend and release context.
/// Safe to call multiple times.
pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (!cuda_initialized) return;

    if (use_native) {
        cuda_native.deinit();
        use_native = false;
    }

    if (cuda_context) |*ctx| {
        // Cleanup device memory if any was allocated
        // Note: In a real implementation, we should track and free device allocations
        _ = ctx;
    }

    cuda_context = null;
    cuda_initialized = false;

    // Unload CUDA library
    cuda_loader.unload();
}

fn ensureNativeInitialized() !void {
    if (!use_native) {
        cuda_native.init() catch |err| {
            std.log.warn("Failed to initialize native CUDA: {}. Using fallback.", .{err});
            return err;
        };
        use_native = true;
        std.log.info("Using native CUDA implementation", .{});
    }
}

/// Compile CUDA kernel source code.
/// @param allocator Memory allocator for compilation artifacts
/// @param source CUDA kernel source code and configuration
/// @return Opaque handle to compiled kernel or KernelError on failure
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (ensureNativeInitialized()) |_| {
        if (cuda_native.compileKernel(allocator, source)) |result| {
            return result;
        } else |err| {
            std.log.warn("Native kernel compilation failed: {}. Falling back to simulation.", .{err});
        }
    } else |err| {
        std.log.warn("Native initialization failed: {}. Using fallback.", .{err});
    }
    return fallback.compileKernel(allocator, source);
}

/// Launch a compiled CUDA kernel with specified configuration.
/// @param allocator Memory allocator (currently unused)
/// @param kernel_handle Opaque handle from compileKernel
/// @param config Kernel execution configuration (grid/block dimensions)
/// @param args Kernel arguments as array of pointers
/// @return KernelError on launch failure
pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    if (use_native) {
        if (cuda_native.launchKernel(allocator, kernel_handle, config, args)) |_| {
            return;
        } else |err| {
            std.log.warn("Native kernel launch failed: {}. Falling back to simulation.", .{err});
        }
    }
    return fallback.launchKernel(allocator, kernel_handle, config, args);
}

/// Destroy a compiled CUDA kernel and release resources.
/// @param allocator Memory allocator (currently unused)
/// @param kernel_handle Opaque handle from compileKernel to destroy
pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (use_native) {
        cuda_native.destroyKernel(allocator, kernel_handle);
        return;
    }
    fallback.destroyKernel(allocator, kernel_handle);
}

/// Create a new CUDA stream for asynchronous execution.
/// @return Opaque pointer to CUDA stream or CuResult error
pub fn createStream() !*anyopaque {
    if (use_native) {
        if (cuda_native.createStream()) |result| {
            return result;
        } else |err| {
            std.log.warn("Native stream creation failed: {}. Falling back to simulation.", .{err});
        }
    }
    return fallback.createOpaqueHandle(CuStream, .{ .ptr = null });
}

/// Destroy a CUDA stream.
/// @param stream_ Opaque pointer to CUDA stream to destroy
pub fn destroyStream(stream_: *anyopaque) void {
    if (use_native) {
        cuda_native.destroyStream(stream_);
        return;
    }
    fallback.destroyOpaqueHandle(CuStream, stream_);
}

/// Synchronize a CUDA stream, blocking until all operations complete.
/// @param stream_ Opaque pointer to CUDA stream to synchronize
/// @return CuResult error on synchronization failure
pub fn synchronizeStream(stream_: *anyopaque) !void {
    if (use_native) {
        return cuda_native.synchronizeStream(stream_);
    }
    const cu_stream: *CuStream = @ptrCast(@alignCast(stream_));
    _ = cu_stream;
}

/// Allocate device memory on CUDA GPU.
/// @param size Size in bytes to allocate
/// @return Opaque pointer to allocated memory or CuResult error
pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (use_native) {
        if (cuda_native.allocateDeviceMemory(size)) |result| {
            return result;
        } else |err| {
            std.log.warn("Native device memory allocation failed: {}. Falling back to simulation.", .{err});
        }
    }
    return fallback.allocateDeviceMemory(size);
}

/// Free device memory allocated by allocateDeviceMemory.
/// @param ptr Opaque pointer to memory to free
pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (use_native) {
        cuda_native.freeDeviceMemory(ptr);
        return;
    }
    fallback.freeDeviceMemory(ptr);
}

/// Copy data from host memory to CUDA device memory.
/// @param dst Device memory destination pointer
/// @param src Host memory source pointer
/// @param size Number of bytes to copy
/// @return CuResult error on transfer failure
pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (use_native) {
        if (cuda_native.memcpyHostToDevice(dst, @ptrCast(@alignCast(src)), size)) |_| {
            return;
        } else |err| {
            std.log.warn("Native host-to-device copy failed: {}. Falling back to simulation.", .{err});
        }
    }
    return fallback.memcpyHostToDevice(dst, @ptrCast(@alignCast(src)), size);
}

/// Copy data from CUDA device memory to host memory.
/// @param dst Host memory destination pointer
/// @param src Device memory source pointer
/// @param size Number of bytes to copy
/// @return CuResult error on transfer failure
pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (use_native) {
        if (cuda_native.memcpyDeviceToHost(dst, src, size)) |_| {
            return;
        } else |err| {
            std.log.warn("Native device-to-host copy failed: {}. Falling back to simulation.", .{err});
        }
    }
    return fallback.memcpyDeviceToHost(dst, @ptrCast(@alignCast(src)), size);
}

/// Check if CUDA is available
fn tryLoadCuda(allocator: std.mem.Allocator) bool {
    return cuda_loader.isAvailableWithAlloc(allocator);
}

/// Load CUDA functions using consolidated loader
fn loadCudaFunctions(allocator: std.mem.Allocator) bool {
    const funcs = cuda_loader.load(allocator) catch return false;
    return funcs.core.cuInit != null and funcs.core.cuDeviceGetCount != null;
}
