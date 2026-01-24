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
// Async memory transfer function types
pub const CuMemcpyHtoDAsyncFn = *const fn (u64, *const anyopaque, usize, ?*anyopaque) callconv(.c) CuResult;
pub const CuMemcpyDtoHAsyncFn = *const fn (*anyopaque, u64, usize, ?*anyopaque) callconv(.c) CuResult;
pub const CuMemcpyDtoDAsyncFn = *const fn (u64, u64, usize, ?*anyopaque) callconv(.c) CuResult;

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
    // Async memory transfer functions
    cuMemcpyHtoDAsync: ?CuMemcpyHtoDAsyncFn = null,
    cuMemcpyDtoHAsync: ?CuMemcpyDtoHAsyncFn = null,
    cuMemcpyDtoDAsync: ?CuMemcpyDtoDAsyncFn = null,
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

// Global state for the loaded CUDA library and its symbols
var cuda_lib: ?std.DynLib = null;
var cuda_functions: CudaFunctions = .{};
var load_attempted: bool = false;

// Helper to lookup a symbol from the optional library.
fn bind(comptime T: type, name: []const u8) ?T {
    if (cuda_lib) |lib| {
        return lib.lookup(T, name);
    }
    return null;
}

/// Errors that can occur while loading the CUDA driver.
pub const LoadError = error{ LibraryNotFound, SymbolNotFound, PlatformNotSupported };

/// Load CUDA library and all functions
pub fn load(allocator: std.mem.Allocator) LoadError!*const CudaFunctions {
    // If we already attempted loading, return the existing state.
    if (load_attempted) {
        if (cuda_lib != null) return &cuda_functions;
        return error.LibraryNotFound;
    }
    load_attempted = true;

    // Platform‑specific library names; honour a CUDA_PATH env var if set.
    var lib_paths = std.ArrayList([]const u8).init(allocator);
    defer lib_paths.deinit();

    // Optional custom path via environment variable (e.g., "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcuda.dll")
    if (std.process.getEnvVarOwned(allocator, "CUDA_PATH")) |custom_path| {
        defer allocator.free(custom_path);
        // Append the file name for the appropriate OS.
        const file_name = switch (builtin.os.tag) {
            .windows => "nvcuda.dll",
            .linux => "libcuda.so",
            else => "",
        };
        if (file_name.len > 0) {
            const full = std.fs.path.join(allocator, &.{ custom_path, file_name }) catch "";
            if (full.len > 0) _ = lib_paths.append(full) catch {};
        }
    } else |_| {}

    // Default library names per platform.
    const default_names = switch (builtin.os.tag) {
        .windows => &.{"nvcuda.dll"},
        .linux => &.{ "libcuda.so.1", "libcuda.so" },
        else => return error.PlatformNotSupported,
    };
    for (default_names) |n| _ = lib_paths.append(n) catch {};

    // Attempt to open each candidate name.
    for (lib_paths.items) |name| {
        if (std.DynLib.open(name)) |lib| {
            cuda_lib = lib;
            break;
        } else |_| {}
    }

    if (cuda_lib == null) return error.LibraryNotFound;

    // Load core symbols – they are required for any CUDA operation.
    cuda_functions.core.cuInit = bind(CuInitFn, "cuInit");
    cuda_functions.core.cuDeviceGetCount = bind(CuDeviceGetCountFn, "cuDeviceGetCount");
    cuda_functions.core.cuDeviceGet = bind(CuDeviceGetFn, "cuDeviceGet");
    cuda_functions.core.cuCtxCreate = bind(CuCtxCreateFn, "cuCtxCreate_v2");
    cuda_functions.core.cuCtxDestroy = bind(CuCtxDestroyFn, "cuCtxDestroy_v2");
    cuda_functions.core.cuCtxSynchronize = bind(CuCtxSynchronizeFn, "cuCtxSynchronize");

    // Verify required core symbols are present.
    if (cuda_functions.core.cuInit == null or
        cuda_functions.core.cuDeviceGetCount == null or
        cuda_functions.core.cuDeviceGet == null)
    {
        return error.SymbolNotFound;
    }

    // Load optional memory symbols.
    cuda_functions.memory.cuMemAlloc = bind(CuMemAllocFn, "cuMemAlloc_v2");
    cuda_functions.memory.cuMemFree = bind(CuMemFreeFn, "cuMemFree_v2");
    cuda_functions.memory.cuMemcpyHtoD = bind(CuMemcpyHtoDFn, "cuMemcpyHtoD_v2");
    cuda_functions.memory.cuMemcpyDtoH = bind(CuMemcpyDtoHFn, "cuMemcpyDtoH_v2");
    cuda_functions.memory.cuMemcpyDtoD = bind(CuMemcpyDtoDFn, "cuMemcpyDtoD_v2");
    cuda_functions.memory.cuMemAllocHost = bind(CuMemAllocHostFn, "cuMemAllocHost_v2");
    cuda_functions.memory.cuMemFreeHost = bind(CuMemFreeHostFn, "cuMemFreeHost");
    // Load async memory transfer symbols (optional)
    cuda_functions.memory.cuMemcpyHtoDAsync = bind(CuMemcpyHtoDAsyncFn, "cuMemcpyHtoDAsync");
    cuda_functions.memory.cuMemcpyDtoHAsync = bind(CuMemcpyDtoHAsyncFn, "cuMemcpyDtoHAsync");
    cuda_functions.memory.cuMemcpyDtoDAsync = bind(CuMemcpyDtoDAsyncFn, "cuMemcpyDtoDAsync");

    // Load stream symbols.
    cuda_functions.stream.cuStreamCreate = bind(CuStreamCreateFn, "cuStreamCreate");
    cuda_functions.stream.cuStreamDestroy = bind(CuStreamDestroyFn, "cuStreamDestroy_v2");
    cuda_functions.stream.cuStreamSynchronize = bind(CuStreamSynchronizeFn, "cuStreamSynchronize");

    // Load kernel symbols.
    cuda_functions.kernel.cuModuleLoadData = bind(CuModuleLoadDataFn, "cuModuleLoadData");
    cuda_functions.kernel.cuModuleUnload = bind(CuModuleUnloadFn, "cuModuleUnload");
    cuda_functions.kernel.cuModuleGetFunction = bind(CuModuleGetFunctionFn, "cuModuleGetFunction");
    cuda_functions.kernel.cuLaunchKernel = bind(CuLaunchKernelFn, "cuLaunchKernel");

    // Load device query symbols.
    cuda_functions.device.cuDeviceGetName = bind(CuDeviceGetNameFn, "cuDeviceGetName");
    cuda_functions.device.cuDeviceGetAttribute = bind(CuDeviceGetAttributeFn, "cuDeviceGetAttribute");
    cuda_functions.device.cuDeviceTotalMem = bind(CuDeviceTotalMemFn, "cuDeviceTotalMem_v2");

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
