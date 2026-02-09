//! CUDA Library Loader
//!
//! Consolidated CUDA function loading for all CUDA modules.

const std = @import("std");
const builtin = @import("builtin");

// libc import for environment access (Zig 0.16 compatible)
const c = if (builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64)
    @cImport(@cInclude("stdlib.h"))
else
    struct {
        pub fn getenv(_: [*:0]const u8) ?[*:0]const u8 {
            return null;
        }
    };

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
pub const CuStreamWaitEventFn = *const fn (?*anyopaque, ?*anyopaque, u32) callconv(.c) CuResult;

// Event function types
pub const CuEventCreateFn = *const fn (*?*anyopaque, u32) callconv(.c) CuResult;
pub const CuEventDestroyFn = *const fn (?*anyopaque) callconv(.c) CuResult;
pub const CuEventRecordFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) CuResult;
pub const CuEventSynchronizeFn = *const fn (?*anyopaque) callconv(.c) CuResult;
pub const CuEventElapsedTimeFn = *const fn (*f32, ?*anyopaque, ?*anyopaque) callconv(.c) CuResult;

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
    cuStreamWaitEvent: ?CuStreamWaitEventFn = null,
};

/// All CUDA event functions
pub const EventFunctions = struct {
    cuEventCreate: ?CuEventCreateFn = null,
    cuEventDestroy: ?CuEventDestroyFn = null,
    cuEventRecord: ?CuEventRecordFn = null,
    cuEventSynchronize: ?CuEventSynchronizeFn = null,
    cuEventElapsedTime: ?CuEventElapsedTimeFn = null,
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
    event: EventFunctions = .{},
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

fn getEnv(name: [:0]const u8) ?[]const u8 {
    if (builtin.target.os.tag == .freestanding or
        builtin.target.cpu.arch == .wasm32 or
        builtin.target.cpu.arch == .wasm64)
    {
        return null;
    }
    const value_ptr = c.getenv(name.ptr);
    if (value_ptr) |ptr| {
        return std.mem.span(ptr);
    }
    return null;
}

fn appendOwnedPath(
    allocator: std.mem.Allocator,
    lib_paths: *std.ArrayListUnmanaged([]const u8),
    owned_paths: *std.ArrayListUnmanaged([]u8),
    full: []u8,
) void {
    if (owned_paths.append(allocator, full)) |_| {
        if (lib_paths.append(allocator, full)) |_| {
            return;
        }
        if (owned_paths.items.len > 0) {
            owned_paths.items.len -= 1;
        }
        allocator.free(full);
        return;
    }
    allocator.free(full);
}

fn appendEnvCandidates(
    allocator: std.mem.Allocator,
    lib_paths: *std.ArrayListUnmanaged([]const u8),
    owned_paths: *std.ArrayListUnmanaged([]u8),
    base_path: []const u8,
) void {
    if (base_path.len == 0) return;

    const file_names = switch (builtin.os.tag) {
        .windows => &.{"nvcuda.dll"},
        .linux => &.{ "libcuda.so.1", "libcuda.so" },
        else => &.{},
    };
    if (file_names.len == 0) return;

    const subdirs = switch (builtin.os.tag) {
        .windows => &.{ "", "bin" },
        .linux => &.{ "", "lib64", "lib" },
        else => &.{""},
    };

    for (subdirs) |subdir| {
        for (file_names) |file_name| {
            const parts = if (subdir.len == 0)
                &.{ base_path, file_name }
            else
                &.{ base_path, subdir, file_name };
            const full = std.fs.path.join(allocator, parts) catch continue;
            appendOwnedPath(allocator, lib_paths, owned_paths, full);
        }
    }
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
    var lib_paths = std.ArrayListUnmanaged([]const u8).empty;
    defer lib_paths.deinit(allocator);
    var owned_paths = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (owned_paths.items) |owned| {
            allocator.free(owned);
        }
        owned_paths.deinit(allocator);
    }

    // Optional custom paths via environment variables
    if (getEnv("CUDA_PATH")) |custom_path| {
        appendEnvCandidates(allocator, &lib_paths, &owned_paths, custom_path);
    }
    if (getEnv("CUDA_HOME")) |custom_path| {
        appendEnvCandidates(allocator, &lib_paths, &owned_paths, custom_path);
    }
    if (getEnv("CUDA_ROOT")) |custom_path| {
        appendEnvCandidates(allocator, &lib_paths, &owned_paths, custom_path);
    }

    // Default library names per platform.
    const default_names = switch (builtin.os.tag) {
        .windows => &.{"nvcuda.dll"},
        .linux => &.{ "libcuda.so.1", "libcuda.so" },
        else => return error.PlatformNotSupported,
    };
    for (default_names) |n| _ = lib_paths.append(allocator, n) catch |err| {
        std.log.debug("Failed to append CUDA library path '{s}': {t}", .{ n, err });
    };

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
    cuda_functions.stream.cuStreamWaitEvent = bind(CuStreamWaitEventFn, "cuStreamWaitEvent");

    // Load event symbols.
    cuda_functions.event.cuEventCreate = bind(CuEventCreateFn, "cuEventCreate");
    cuda_functions.event.cuEventDestroy = bind(CuEventDestroyFn, "cuEventDestroy");
    cuda_functions.event.cuEventRecord = bind(CuEventRecordFn, "cuEventRecord");
    cuda_functions.event.cuEventSynchronize = bind(CuEventSynchronizeFn, "cuEventSynchronize");
    cuda_functions.event.cuEventElapsedTime = bind(CuEventElapsedTimeFn, "cuEventElapsedTime");

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

/// Check if CUDA is available (without triggering load)
pub fn isAvailable() bool {
    // If already loaded, check the result
    if (load_attempted) {
        return cuda_lib != null and cuda_functions.core.cuInit != null;
    }
    // Not loaded yet - return false (caller should call load() with allocator first)
    return false;
}

/// Check if CUDA is available, attempting to load if necessary
pub fn isAvailableWithAlloc(allocator: std.mem.Allocator) bool {
    if (!load_attempted) {
        _ = load(allocator) catch return false;
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

// ============================================================================
// High-Level Kernel Launch API
// ============================================================================

/// Error type for kernel operations.
pub const KernelError = error{
    NotLoaded,
    NotAvailable,
    KernelLaunchFailed,
    OutOfMemory,
};

/// Launch a CUDA kernel with the given configuration.
pub fn launchKernel(
    kernel: *anyopaque,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    stream: ?*anyopaque,
    args: []const usize,
) KernelError!void {
    const funcs = getFunctions() orelse return error.NotLoaded;
    const launch_fn = funcs.kernel.cuLaunchKernel orelse return error.NotAvailable;

    // Convert usize args to ?*anyopaque array for CUDA API.
    // Use a small stack buffer to avoid heap allocation for common cases.
    var stack_args: [16]?*anyopaque = undefined;
    var arg_ptrs: []?*anyopaque = undefined;
    var heap_allocated = false;

    if (args.len <= stack_args.len) {
        arg_ptrs = stack_args[0..args.len];
    } else {
        const allocator = std.heap.page_allocator;
        arg_ptrs = allocator.alloc(?*anyopaque, args.len) catch return error.OutOfMemory;
        heap_allocated = true;
    }
    defer if (heap_allocated) std.heap.page_allocator.free(arg_ptrs);

    for (args, 0..) |_, i| {
        arg_ptrs[i] = @ptrCast(@constCast(&args[i]));
    }

    const result = launch_fn(
        kernel,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        stream,
        arg_ptrs.ptr,
        null,
    );

    if (result != .success) {
        return error.KernelLaunchFailed;
    }
}

/// Unload a CUDA module.
pub fn unloadModule(module: *anyopaque) void {
    const funcs = getFunctions() orelse return;
    if (funcs.kernel.cuModuleUnload) |unload_fn| {
        _ = unload_fn(module);
    }
}
