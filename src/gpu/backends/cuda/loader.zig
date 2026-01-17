//! CUDA Library Loader
//!
//! Consolidated CUDA function loading for all CUDA modules.

const std = @import("std");
const builtin = @import("builtin");

pub const CuResult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    deinitialized = 4,
    invalid_context = 201,
    invalid_handle = 400,
    not_found = 500,
    not_ready = 600,
    _,
};

// Core CUDA function types
pub const CuInitFn = *const fn (u32) callconv(.c) CuResult;
pub const CuDeviceGetCountFn = *const fn (*i32) callconv(.c) CuResult;
pub const CuDeviceGetFn = *const fn (*i32, i32) callconv(.c) CuResult;
pub const CuCtxCreateFn = *const fn (*?*anyopaque, u32, i32) callconv(.c) CuResult;
pub const CuCtxDestroyFn = *const fn (?*anyopaque) callconv(.c) CuResult;
pub const CuCtxSynchronizeFn = *const fn () callconv(.c) CuResult;

// Memory function types
pub const CuMemAllocFn = *const fn (*u64, usize) callconv(.c) CuResult;
pub const CuMemFreeFn = *const fn (u64) callconv(.c) CuResult;
pub const CuMemcpyHtoDFn = *const fn (u64, *const anyopaque, usize) callconv(.c) CuResult;
pub const CuMemcpyDtoHFn = *const fn (*anyopaque, u64, usize) callconv(.c) CuResult;
pub const CuMemcpyDtoDFn = *const fn (u64, u64, usize) callconv(.c) CuResult;
pub const CuMemAllocHostFn = *const fn (**anyopaque, usize) callconv(.c) CuResult;
pub const CuMemFreeHostFn = *const fn (*anyopaque) callconv(.c) CuResult;

// Stream function types
pub const CuStreamCreateFn = *const fn (*?*anyopaque, u32) callconv(.c) CuResult;
pub const CuStreamDestroyFn = *const fn (?*anyopaque) callconv(.c) CuResult;
pub const CuStreamSynchronizeFn = *const fn (?*anyopaque) callconv(.c) CuResult;

// Module/kernel function types
pub const CuModuleLoadDataFn = *const fn (*?*anyopaque, *const anyopaque) callconv(.c) CuResult;
pub const CuModuleUnloadFn = *const fn (?*anyopaque) callconv(.c) CuResult;
pub const CuModuleGetFunctionFn = *const fn (*?*anyopaque, ?*anyopaque, [*:0]const u8) callconv(.c) CuResult;
pub const CuLaunchKernelFn = *const fn (
    ?*anyopaque,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    ?*anyopaque,
    ?[*]?*anyopaque,
    ?[*]?*anyopaque,
) callconv(.c) CuResult;

// Device query function types
pub const CuDeviceGetNameFn = *const fn ([*]u8, i32, i32) callconv(.c) CuResult;
pub const CuDeviceGetAttributeFn = *const fn (*i32, i32, i32) callconv(.c) CuResult;
pub const CuDeviceTotalMemFn = *const fn (*usize, i32) callconv(.c) CuResult;

/// All CUDA core functions
pub const CoreFunctions = struct {
    cuInit: ?CuInitFn = null,
    cuDeviceGetCount: ?CuDeviceGetCountFn = null,
    cuDeviceGet: ?CuDeviceGetFn = null,
    cuCtxCreate: ?CuCtxCreateFn = null,
    cuCtxDestroy: ?CuCtxDestroyFn = null,
    cuCtxSynchronize: ?CuCtxSynchronizeFn = null,
};

/// All CUDA memory functions
pub const MemoryFunctions = struct {
    cuMemAlloc: ?CuMemAllocFn = null,
    cuMemFree: ?CuMemFreeFn = null,
    cuMemcpyHtoD: ?CuMemcpyHtoDFn = null,
    cuMemcpyDtoH: ?CuMemcpyDtoHFn = null,
    cuMemcpyDtoD: ?CuMemcpyDtoDFn = null,
    cuMemAllocHost: ?CuMemAllocHostFn = null,
    cuMemFreeHost: ?CuMemFreeHostFn = null,
};

/// All CUDA stream functions
pub const StreamFunctions = struct {
    cuStreamCreate: ?CuStreamCreateFn = null,
    cuStreamDestroy: ?CuStreamDestroyFn = null,
    cuStreamSynchronize: ?CuStreamSynchronizeFn = null,
};

/// All CUDA kernel functions
pub const KernelFunctions = struct {
    cuModuleLoadData: ?CuModuleLoadDataFn = null,
    cuModuleUnload: ?CuModuleUnloadFn = null,
    cuModuleGetFunction: ?CuModuleGetFunctionFn = null,
    cuLaunchKernel: ?CuLaunchKernelFn = null,
};

/// All CUDA device query functions
pub const DeviceFunctions = struct {
    cuDeviceGetName: ?CuDeviceGetNameFn = null,
    cuDeviceGetAttribute: ?CuDeviceGetAttributeFn = null,
    cuDeviceTotalMem: ?CuDeviceTotalMemFn = null,
};

/// Complete CUDA function set
pub const CudaFunctions = struct {
    core: CoreFunctions = .{},
    memory: MemoryFunctions = .{},
    stream: StreamFunctions = .{},
    kernel: KernelFunctions = .{},
    device: DeviceFunctions = .{},
};

var cuda_lib: ?std.DynLib = null;
var cuda_functions: CudaFunctions = .{};
var load_attempted: bool = false;

/// Load CUDA library and all functions
pub fn load() !*const CudaFunctions {
    if (load_attempted) {
        if (cuda_lib != null) return &cuda_functions;
        return error.LibraryNotFound;
    }
    load_attempted = true;

    // Try platform-specific library names
    const lib_names: []const []const u8 = switch (builtin.os.tag) {
        .windows => &.{"nvcuda.dll"},
        .linux => &.{ "libcuda.so.1", "libcuda.so" },
        else => return error.PlatformNotSupported,
    };

    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            cuda_lib = lib;
            break;
        } else |_| {}
    }

    if (cuda_lib == null) return error.LibraryNotFound;

    // Load core functions
    cuda_functions.core.cuInit = cuda_lib.?.lookup(CuInitFn, "cuInit");
    cuda_functions.core.cuDeviceGetCount = cuda_lib.?.lookup(CuDeviceGetCountFn, "cuDeviceGetCount");
    cuda_functions.core.cuDeviceGet = cuda_lib.?.lookup(CuDeviceGetFn, "cuDeviceGet");
    cuda_functions.core.cuCtxCreate = cuda_lib.?.lookup(CuCtxCreateFn, "cuCtxCreate_v2");
    cuda_functions.core.cuCtxDestroy = cuda_lib.?.lookup(CuCtxDestroyFn, "cuCtxDestroy_v2");
    cuda_functions.core.cuCtxSynchronize = cuda_lib.?.lookup(CuCtxSynchronizeFn, "cuCtxSynchronize");

    // Load memory functions
    cuda_functions.memory.cuMemAlloc = cuda_lib.?.lookup(CuMemAllocFn, "cuMemAlloc_v2");
    cuda_functions.memory.cuMemFree = cuda_lib.?.lookup(CuMemFreeFn, "cuMemFree_v2");
    cuda_functions.memory.cuMemcpyHtoD = cuda_lib.?.lookup(CuMemcpyHtoDFn, "cuMemcpyHtoD_v2");
    cuda_functions.memory.cuMemcpyDtoH = cuda_lib.?.lookup(CuMemcpyDtoHFn, "cuMemcpyDtoH_v2");
    cuda_functions.memory.cuMemcpyDtoD = cuda_lib.?.lookup(CuMemcpyDtoDFn, "cuMemcpyDtoD_v2");
    cuda_functions.memory.cuMemAllocHost = cuda_lib.?.lookup(CuMemAllocHostFn, "cuMemAllocHost_v2");
    cuda_functions.memory.cuMemFreeHost = cuda_lib.?.lookup(CuMemFreeHostFn, "cuMemFreeHost");

    // Load stream functions
    cuda_functions.stream.cuStreamCreate = cuda_lib.?.lookup(CuStreamCreateFn, "cuStreamCreate");
    cuda_functions.stream.cuStreamDestroy = cuda_lib.?.lookup(CuStreamDestroyFn, "cuStreamDestroy_v2");
    cuda_functions.stream.cuStreamSynchronize = cuda_lib.?.lookup(CuStreamSynchronizeFn, "cuStreamSynchronize");

    // Load kernel functions
    cuda_functions.kernel.cuModuleLoadData = cuda_lib.?.lookup(CuModuleLoadDataFn, "cuModuleLoadData");
    cuda_functions.kernel.cuModuleUnload = cuda_lib.?.lookup(CuModuleUnloadFn, "cuModuleUnload");
    cuda_functions.kernel.cuModuleGetFunction = cuda_lib.?.lookup(CuModuleGetFunctionFn, "cuModuleGetFunction");
    cuda_functions.kernel.cuLaunchKernel = cuda_lib.?.lookup(CuLaunchKernelFn, "cuLaunchKernel");

    // Load device functions
    cuda_functions.device.cuDeviceGetName = cuda_lib.?.lookup(CuDeviceGetNameFn, "cuDeviceGetName");
    cuda_functions.device.cuDeviceGetAttribute = cuda_lib.?.lookup(CuDeviceGetAttributeFn, "cuDeviceGetAttribute");
    cuda_functions.device.cuDeviceTotalMem = cuda_lib.?.lookup(CuDeviceTotalMemFn, "cuDeviceTotalMem_v2");

    return &cuda_functions;
}

/// Unload CUDA library
pub fn unload() void {
    if (cuda_lib) |*lib| {
        lib.close();
        cuda_lib = null;
    }
    cuda_functions = .{};
    load_attempted = false;
}

/// Check if CUDA is available
pub fn isAvailable() bool {
    if (!load_attempted) {
        _ = load() catch return false;
    }
    return cuda_lib != null and cuda_functions.core.cuInit != null;
}

/// Get loaded functions (returns null if not loaded)
pub fn getFunctions() ?*const CudaFunctions {
    if (cuda_lib == null) return null;
    return &cuda_functions;
}

/// Check CUDA result and convert to error
pub fn checkResult(result: CuResult) error{ CudaError, InvalidValue, OutOfMemory, NotInitialized }!void {
    return switch (result) {
        .success => {},
        .invalid_value => error.InvalidValue,
        .out_of_memory => error.OutOfMemory,
        .not_initialized => error.NotInitialized,
        else => error.CudaError,
    };
}
