//! Native CUDA backend implementation with real GPU execution.
//!
//! Provides actual CUDA kernel compilation, execution, and memory management
//! using the CUDA Driver API instead of fallback simulation.

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const types = @import("../../kernel_types.zig");
const shared = @import("../shared.zig");

pub const CudaError = error{
    InitializationFailed,
    DriverNotFound,
    DeviceNotFound,
    OutOfMemory,
    KernelLaunchFailed,
    MemoryCopyFailed,
    InvalidKernel,
};

pub const CudaContext = struct {
    device_id: i32,
    context: ?*anyopaque,
    stream: ?*anyopaque,
    device_memory: std.ArrayListUnmanaged([]u8),
    allocator: std.mem.Allocator,
};

var cuda_initialized = false;
var cuda_context: ?CudaContext = null;
var cuda_lib: ?std.DynLib = null;
var init_mutex: sync.Mutex = .{};

const CuResult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_context = 6,
    invalid_device = 101,
    launch_failure = 700,
    launch_out_of_resources = 701,
    invalid_kernel = 209,
};

const CUdevice = i32;
const CUcontext = *anyopaque;
const CUstream = *anyopaque;
const CUmodule = *anyopaque;
const CUfunction = *anyopaque;
const CUdeviceptr = usize;

const CuInitFn = *const fn (u32) callconv(.c) CuResult;
const CuDeviceGetFn = *const fn (*CUdevice, i32) callconv(.c) CuResult;
const CuDeviceGetCountFn = *const fn (*i32) callconv(.c) CuResult;
const CuCtxCreateFn = *const fn (*CUcontext, u32, CUdevice) callconv(.c) CuResult;
const CuCtxGetCurrentFn = *const fn (*CUcontext) callconv(.c) CuResult;
const CuCtxSetCurrentFn = *const fn (CUcontext) callconv(.c) CuResult;
const CuCtxDestroyFn = *const fn (CUcontext) callconv(.c) CuResult;
const CuModuleLoadDataFn = *const fn (*CUmodule, *const anyopaque) callconv(.c) CuResult;
const CuModuleUnloadFn = *const fn (CUmodule) callconv(.c) CuResult;
const CuModuleGetFunctionFn = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CuResult;
const CuMemAllocFn = *const fn (*CUdeviceptr, usize) callconv(.c) CuResult;
const CuMemFreeFn = *const fn (CUdeviceptr) callconv(.c) CuResult;
const CuMemcpyH2DFn = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CuResult;
const CuMemcpyD2HFn = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CuResult;
const CuLaunchKernelFn = *const fn (
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    [*]*const anyopaque,
    [*]*const anyopaque,
) callconv(.c) CuResult;
const CuStreamCreateFn = *const fn (*CUstream, u32) callconv(.c) CuResult;
const CuStreamDestroyFn = *const fn (CUstream) callconv(.c) CuResult;
const CuStreamSynchronizeFn = *const fn (CUstream) callconv(.c) CuResult;
const CuMemcpyD2DFn = *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CuResult;
const CuMemcpyD2DAsyncFn = *const fn (CUdeviceptr, CUdeviceptr, usize, CUstream) callconv(.c) CuResult;
const CuMemcpyH2DAsyncFn = *const fn (CUdeviceptr, *const anyopaque, usize, CUstream) callconv(.c) CuResult;
const CuMemcpyD2HAsyncFn = *const fn (*anyopaque, CUdeviceptr, usize, CUstream) callconv(.c) CuResult;
const CuMemGetInfoFn = *const fn (*usize, *usize) callconv(.c) CuResult;
const CuDeviceGetAttributeFn = *const fn (*i32, i32, CUdevice) callconv(.c) CuResult;
const CuDeviceGetNameFn = *const fn ([*]u8, i32, CUdevice) callconv(.c) CuResult;
const CuDeviceTotalMemFn = *const fn (*usize, CUdevice) callconv(.c) CuResult;
const CuOccupancyMaxPotentialBlockSizeFn = *const fn (*i32, *i32, CUfunction, ?*anyopaque, usize, i32) callconv(.c) CuResult;
const CuFuncGetAttributeFn = *const fn (*i32, i32, CUfunction) callconv(.c) CuResult;
const CuMemsetD8Fn = *const fn (CUdeviceptr, u8, usize) callconv(.c) CuResult;
const CuMemsetD32Fn = *const fn (CUdeviceptr, u32, usize) callconv(.c) CuResult;

var cuInit: ?CuInitFn = null;
var cuDeviceGet: ?CuDeviceGetFn = null;
var cuDeviceGetCount: ?CuDeviceGetCountFn = null;
var cuCtxCreate: ?CuCtxCreateFn = null;
var cuCtxGetCurrent: ?CuCtxGetCurrentFn = null;
var cuCtxSetCurrent: ?CuCtxSetCurrentFn = null;
var cuCtxDestroy: ?CuCtxDestroyFn = null;
var cuModuleLoadData: ?CuModuleLoadDataFn = null;
var cuModuleUnload: ?CuModuleUnloadFn = null;
var cuModuleGetFunction: ?CuModuleGetFunctionFn = null;
var cuMemAlloc: ?CuMemAllocFn = null;
var cuMemFree: ?CuMemFreeFn = null;
var cuMemcpyH2D: ?CuMemcpyH2DFn = null;
var cuMemcpyD2H: ?CuMemcpyD2HFn = null;
var cuLaunchKernel: ?CuLaunchKernelFn = null;
var cuStreamCreate: ?CuStreamCreateFn = null;
var cuStreamDestroy: ?CuStreamDestroyFn = null;
var cuStreamSynchronize: ?CuStreamSynchronizeFn = null;
var cuMemcpyD2D: ?CuMemcpyD2DFn = null;
var cuMemcpyD2DAsync: ?CuMemcpyD2DAsyncFn = null;
var cuMemcpyH2DAsync: ?CuMemcpyH2DAsyncFn = null;
var cuMemcpyD2HAsync: ?CuMemcpyD2HAsyncFn = null;
var cuMemGetInfo: ?CuMemGetInfoFn = null;
var cuDeviceGetAttribute: ?CuDeviceGetAttributeFn = null;
var cuDeviceGetName: ?CuDeviceGetNameFn = null;
var cuDeviceTotalMem: ?CuDeviceTotalMemFn = null;
var cuOccupancyMaxPotentialBlockSize: ?CuOccupancyMaxPotentialBlockSizeFn = null;
var cuFuncGetAttribute: ?CuFuncGetAttributeFn = null;
var cuMemsetD8: ?CuMemsetD8Fn = null;
var cuMemsetD32: ?CuMemsetD32Fn = null;

pub const DeviceAttribute = enum(i32) {
    max_threads_per_block = 1,
    max_block_dim_x = 2,
    max_block_dim_y = 3,
    max_block_dim_z = 4,
    max_grid_dim_x = 5,
    max_grid_dim_y = 6,
    max_grid_dim_z = 7,
    max_shared_memory_per_block = 8,
    total_constant_memory = 9,
    warp_size = 10,
    max_registers_per_block = 12,
    clock_rate = 13,
    multiprocessor_count = 16,
    compute_capability_major = 75,
    compute_capability_minor = 76,
    max_threads_per_multiprocessor = 39,
    async_engine_count = 40,
    unified_addressing = 41,
    l2_cache_size = 38,
    memory_clock_rate = 36,
    global_memory_bus_width = 37,
    ecc_enabled = 32,
    tcc_driver = 35,
    memory_pools_supported = 115,
    tensor_core_count = 137,
};

pub const DeviceProperties = struct {
    name: [256]u8,
    total_memory: usize,
    shared_memory_per_block: usize,
    registers_per_block: i32,
    warp_size: i32,
    max_threads_per_block: i32,
    max_block_dim: [3]i32,
    max_grid_dim: [3]i32,
    clock_rate_khz: i32,
    multiprocessor_count: i32,
    compute_capability: struct { major: i32, minor: i32 },
    max_threads_per_multiprocessor: i32,
    l2_cache_size: i32,
    async_engine_count: i32,
    unified_addressing: bool,
    ecc_enabled: bool,
    tensor_core_count: i32,
};

pub fn init() !void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (cuda_initialized) return;

    if (!tryLoadCuda()) {
        return CudaError.DriverNotFound;
    }

    if (!loadCudaFunctions()) {
        return CudaError.DriverNotFound;
    }

    const init_fn = cuInit orelse return CudaError.DriverNotFound;
    if (init_fn(0) != .success) {
        return CudaError.InitializationFailed;
    }

    var device_count: i32 = 0;
    const count_fn = cuDeviceGetCount orelse return CudaError.DriverNotFound;
    if (count_fn(&device_count) != .success or device_count <= 0) {
        return CudaError.DeviceNotFound;
    }

    var device: CUdevice = 0;
    const get_fn = cuDeviceGet orelse return CudaError.DriverNotFound;
    if (get_fn(&device, 0) != .success) {
        return CudaError.DeviceNotFound;
    }

    var context: CUcontext = null;
    const ctx_fn = cuCtxCreate orelse return CudaError.DriverNotFound;
    if (ctx_fn(&context, 0, device) != .success) {
        return CudaError.InitializationFailed;
    }

    var stream: CUstream = null;
    const stream_fn = cuStreamCreate orelse return CudaError.DriverNotFound;
    if (stream_fn(&stream, 0) != .success) {
        if (cuCtxDestroy) |destroy_fn| {
            _ = destroy_fn(context);
        }
        return CudaError.InitializationFailed;
    }

    cuda_context = CudaContext{
        .device_id = @intCast(device),
        .context = context,
        .stream = stream,
        .device_memory = std.ArrayListUnmanaged([]u8).empty,
        .allocator = std.heap.page_allocator,
    };

    cuda_initialized = true;
}

pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (!cuda_initialized) return;

    if (cuda_context) |*ctx| {
        const destroy_fn = cuCtxDestroy orelse return;
        const stream_destroy_fn = cuStreamDestroy orelse return;

        if (ctx.stream) |stream| {
            _ = stream_destroy_fn(stream);
        }

        for (ctx.device_memory.items) |mem| {
            const free_fn = cuMemFree orelse return;
            _ = free_fn(@intCast(@intFromPtr(mem.ptr)));
        }
        ctx.device_memory.deinit(ctx.allocator);

        if (ctx.context) |context| {
            _ = destroy_fn(context);
        }
    }
    cuda_context = null;
    cuda_initialized = false;

    if (cuda_lib) |*lib| {
        lib.close();
    }

    cuda_context = null;
    cuda_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    _ = allocator;

    const load_fn = cuModuleLoadData orelse return types.KernelError.CompilationFailed;

    var module: CUmodule = undefined;
    const result = load_fn(&module, source.source.ptr);

    if (result != .success) {
        return types.KernelError.CompilationFailed;
    }

    const get_fn = cuModuleGetFunction orelse {
        const unload_fn = cuModuleUnload orelse return types.KernelError.CompilationFailed;
        unload_fn(module);
        return types.KernelError.CompilationFailed;
    };

    var function: CUfunction = undefined;
    const func_result = get_fn(&function, module, source.entry_point.ptr);

    if (func_result != .success) {
        const unload_fn = cuModuleUnload orelse return types.KernelError.CompilationFailed;
        unload_fn(module);
        return types.KernelError.CompilationFailed;
    }

    const handle = try std.heap.page_allocator.create(CudaKernel);
    handle.* = .{
        .module = module,
        .function = function,
        .name = try std.heap.page_allocator.dupe(u8, source.name),
    };

    return handle;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;

    const handle: *CudaKernel = @ptrCast(@alignCast(kernel_handle));
    const launch_fn = cuLaunchKernel orelse return types.KernelError.LaunchFailed;

    const stream = if (cuda_context) |ctx| ctx.stream else null;

    var kernel_args = try std.heap.page_allocator.alloc(*anyopaque, args.len);
    defer std.heap.page_allocator.free(kernel_args);

    for (args, 0..) |arg, i| {
        kernel_args[i] = @constCast(arg orelse null);
    }

    const result = launch_fn(
        handle.function,
        config.grid_dim[0],
        config.grid_dim[1],
        config.grid_dim[2],
        config.block_dim[0],
        config.block_dim[1],
        config.block_dim[2],
        config.shared_memory_bytes,
        stream,
        kernel_args.ptr,
        null,
    );

    if (result != .success) {
        return types.KernelError.LaunchFailed;
    }

    const sync_fn = cuStreamSynchronize orelse return types.KernelError.LaunchFailed;
    if (sync_fn(stream) != .success) {
        return types.KernelError.LaunchFailed;
    }
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    _ = allocator;

    const handle: *CudaKernel = @ptrCast(@alignCast(kernel_handle));
    const unload_fn = cuModuleUnload orelse return;

    unload_fn(handle.module);
    std.heap.page_allocator.free(handle.name);
    std.heap.page_allocator.destroy(handle);
}

pub fn createStream() !*anyopaque {
    if (cuda_context == null) return CudaError.InitializationFailed;

    const stream_fn = cuStreamCreate orelse return CudaError.InitializationFailed;
    var stream: CUstream = null;

    if (stream_fn(&stream, 0) != .success) {
        return CudaError.InitializationFailed;
    }

    const handle = try std.heap.page_allocator.create(CuStream);
    handle.* = .{ .ptr = stream };
    return handle;
}

pub fn destroyStream(stream: *anyopaque) void {
    const cu_stream: *CuStream = @ptrCast(@alignCast(stream));
    const destroy_fn = cuStreamDestroy orelse return;
    destroy_fn(cu_stream.ptr);
    std.heap.page_allocator.destroy(cu_stream);
}

pub fn synchronizeStream(stream: *anyopaque) !void {
    const cu_stream: *CuStream = @ptrCast(@alignCast(stream));
    const sync_fn = cuStreamSynchronize orelse return CudaError.KernelLaunchFailed;

    if (sync_fn(cu_stream.ptr) != .success) {
        return CudaError.KernelLaunchFailed;
    }
}

pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (cuda_context == null) return CudaError.OutOfMemory;

    const alloc_fn = cuMemAlloc orelse return CudaError.OutOfMemory;
    var ptr: CUdeviceptr = 0;

    if (alloc_fn(&ptr, size) != .success) {
        return CudaError.OutOfMemory;
    }

    const ctx = &cuda_context.?;
    const mem = try ctx.allocator.alloc(u8, size);
    ctx.device_memory.append(ctx.allocator, mem);

    mem.ptr = @ptrFromInt(ptr);
    return mem.ptr;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (cuda_context == null) return;
    if (@intFromPtr(ptr) == 0) return;

    const free_fn = cuMemFree orelse return;
    free_fn(@intCast(@intFromPtr(ptr)));

    const ctx = &cuda_context.?;
    for (ctx.device_memory.items, 0..) |mem, i| {
        if (mem.ptr == ptr) {
            ctx.allocator.free(mem);
            _ = ctx.device_memory.orderedRemove(i);
            break;
        }
    }
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) !void {
    const copy_fn = cuMemcpyH2D orelse return CudaError.MemoryCopyFailed;

    if (copy_fn(@intCast(@intFromPtr(dst)), src, size) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const copy_fn = cuMemcpyD2H orelse return CudaError.MemoryCopyFailed;

    if (copy_fn(dst, @intCast(@intFromPtr(src)), size) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const copy_fn = cuMemcpyD2D orelse return CudaError.MemoryCopyFailed;

    if (copy_fn(@intCast(@intFromPtr(dst)), @intCast(@intFromPtr(src)), size) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memcpyDeviceToDeviceAsync(dst: *anyopaque, src: *anyopaque, size: usize, stream: ?*anyopaque) !void {
    const copy_fn = cuMemcpyD2DAsync orelse return CudaError.MemoryCopyFailed;
    const stream_ptr = if (stream) |s| @as(CUstream, @ptrCast(s)) else if (cuda_context) |ctx| ctx.stream else null;

    if (copy_fn(@intCast(@intFromPtr(dst)), @intCast(@intFromPtr(src)), size, stream_ptr) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memcpyHostToDeviceAsync(dst: *anyopaque, src: *const anyopaque, size: usize, stream: ?*anyopaque) !void {
    const copy_fn = cuMemcpyH2DAsync orelse return CudaError.MemoryCopyFailed;
    const stream_ptr = if (stream) |s| @as(CUstream, @ptrCast(s)) else if (cuda_context) |ctx| ctx.stream else null;

    if (copy_fn(@intCast(@intFromPtr(dst)), src, size, stream_ptr) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memcpyDeviceToHostAsync(dst: *anyopaque, src: *anyopaque, size: usize, stream: ?*anyopaque) !void {
    const copy_fn = cuMemcpyD2HAsync orelse return CudaError.MemoryCopyFailed;
    const stream_ptr = if (stream) |s| @as(CUstream, @ptrCast(s)) else if (cuda_context) |ctx| ctx.stream else null;

    if (copy_fn(dst, @intCast(@intFromPtr(src)), size, stream_ptr) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memset(dst: *anyopaque, value: u8, size: usize) !void {
    const memset_fn = cuMemsetD8 orelse return CudaError.MemoryCopyFailed;

    if (memset_fn(@intCast(@intFromPtr(dst)), value, size) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn memsetU32(dst: *anyopaque, value: u32, count: usize) !void {
    const memset_fn = cuMemsetD32 orelse return CudaError.MemoryCopyFailed;

    if (memset_fn(@intCast(@intFromPtr(dst)), value, count) != .success) {
        return CudaError.MemoryCopyFailed;
    }
}

pub fn getMemoryInfo() !struct { free: usize, total: usize } {
    const info_fn = cuMemGetInfo orelse return CudaError.DriverNotFound;
    var free: usize = 0;
    var total: usize = 0;

    if (info_fn(&free, &total) != .success) {
        return CudaError.DriverNotFound;
    }

    return .{ .free = free, .total = total };
}

pub fn getDeviceProperties(device_id: i32) !DeviceProperties {
    var props = DeviceProperties{
        .name = undefined,
        .total_memory = 0,
        .shared_memory_per_block = 0,
        .registers_per_block = 0,
        .warp_size = 32,
        .max_threads_per_block = 1024,
        .max_block_dim = .{ 1024, 1024, 64 },
        .max_grid_dim = .{ 2147483647, 65535, 65535 },
        .clock_rate_khz = 0,
        .multiprocessor_count = 0,
        .compute_capability = .{ .major = 0, .minor = 0 },
        .max_threads_per_multiprocessor = 2048,
        .l2_cache_size = 0,
        .async_engine_count = 0,
        .unified_addressing = false,
        .ecc_enabled = false,
        .tensor_core_count = 0,
    };
    @memset(&props.name, 0);

    const get_fn = cuDeviceGet orelse return CudaError.DriverNotFound;
    var device: CUdevice = undefined;
    if (get_fn(&device, device_id) != .success) {
        return CudaError.DeviceNotFound;
    }

    if (cuDeviceGetName) |name_fn| {
        _ = name_fn(&props.name, 256, device);
    }

    if (cuDeviceTotalMem) |mem_fn| {
        _ = mem_fn(&props.total_memory, device);
    }

    const attr_fn = cuDeviceGetAttribute orelse return props;
    var val: i32 = 0;

    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_threads_per_block), device) == .success) {
        props.max_threads_per_block = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_block_dim_x), device) == .success) {
        props.max_block_dim[0] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_block_dim_y), device) == .success) {
        props.max_block_dim[1] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_block_dim_z), device) == .success) {
        props.max_block_dim[2] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_grid_dim_x), device) == .success) {
        props.max_grid_dim[0] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_grid_dim_y), device) == .success) {
        props.max_grid_dim[1] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_grid_dim_z), device) == .success) {
        props.max_grid_dim[2] = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_shared_memory_per_block), device) == .success) {
        props.shared_memory_per_block = @intCast(val);
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.warp_size), device) == .success) {
        props.warp_size = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_registers_per_block), device) == .success) {
        props.registers_per_block = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.clock_rate), device) == .success) {
        props.clock_rate_khz = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.multiprocessor_count), device) == .success) {
        props.multiprocessor_count = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.compute_capability_major), device) == .success) {
        props.compute_capability.major = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.compute_capability_minor), device) == .success) {
        props.compute_capability.minor = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.max_threads_per_multiprocessor), device) == .success) {
        props.max_threads_per_multiprocessor = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.l2_cache_size), device) == .success) {
        props.l2_cache_size = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.async_engine_count), device) == .success) {
        props.async_engine_count = val;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.unified_addressing), device) == .success) {
        props.unified_addressing = val != 0;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.ecc_enabled), device) == .success) {
        props.ecc_enabled = val != 0;
    }
    if (attr_fn(&val, @intFromEnum(DeviceAttribute.tensor_core_count), device) == .success) {
        props.tensor_core_count = val;
    }

    return props;
}

pub fn getOptimalBlockSize(kernel_handle: *anyopaque, dynamic_smem: usize) !struct { min_grid: i32, block_size: i32 } {
    const occupancy_fn = cuOccupancyMaxPotentialBlockSize orelse return CudaError.DriverNotFound;
    const handle: *CudaKernel = @ptrCast(@alignCast(kernel_handle));

    var min_grid_size: i32 = 0;
    var block_size: i32 = 0;

    if (occupancy_fn(&min_grid_size, &block_size, handle.function, null, dynamic_smem, 0) != .success) {
        return CudaError.KernelLaunchFailed;
    }

    return .{ .min_grid = min_grid_size, .block_size = block_size };
}

pub fn hasTensorCores(device_id: i32) bool {
    const props = getDeviceProperties(device_id) catch return false;
    return props.tensor_core_count > 0 or (props.compute_capability.major >= 7);
}

pub fn isInitialized() bool {
    return cuda_initialized;
}

pub fn getDeviceId() ?i32 {
    if (cuda_context) |ctx| {
        return ctx.device_id;
    }
    return null;
}

const CudaKernel = struct {
    module: CUmodule,
    function: CUfunction,
    name: []u8,
};

const CuStream = struct {
    ptr: CUstream,
};

fn tryLoadCuda() bool {
    const lib_names = [_][]const u8{ "nvcuda.dll", "libcuda.so.1", "libcuda.so" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            cuda_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadCudaFunctions() bool {
    if (cuda_lib == null) return false;

    cuInit = cuda_lib.?.lookup(CuInitFn, "cuInit") orelse return false;
    cuDeviceGet = cuda_lib.?.lookup(CuDeviceGetFn, "cuDeviceGet") orelse return false;
    cuDeviceGetCount = cuda_lib.?.lookup(CuDeviceGetCountFn, "cuDeviceGetCount") orelse return false;
    cuCtxCreate = cuda_lib.?.lookup(CuCtxCreateFn, "cuCtxCreate") orelse return false;
    cuCtxGetCurrent = cuda_lib.?.lookup(CuCtxGetCurrentFn, "cuCtxGetCurrent") orelse return false;
    cuCtxSetCurrent = cuda_lib.?.lookup(CuCtxSetCurrentFn, "cuCtxSetCurrent") orelse return false;
    cuCtxDestroy = cuda_lib.?.lookup(CuCtxDestroyFn, "cuCtxDestroy") orelse return false;
    cuModuleLoadData = cuda_lib.?.lookup(CuModuleLoadDataFn, "cuModuleLoadData") orelse return false;
    cuModuleUnload = cuda_lib.?.lookup(CuModuleUnloadFn, "cuModuleUnload") orelse return false;
    cuModuleGetFunction = cuda_lib.?.lookup(CuModuleGetFunctionFn, "cuModuleGetFunction") orelse return false;
    cuMemAlloc = cuda_lib.?.lookup(CuMemAllocFn, "cuMemAlloc") orelse return false;
    cuMemFree = cuda_lib.?.lookup(CuMemFreeFn, "cuMemFree") orelse return false;
    cuMemcpyH2D = cuda_lib.?.lookup(CuMemcpyH2DFn, "cuMemcpyHtoD") orelse return false;
    cuMemcpyD2H = cuda_lib.?.lookup(CuMemcpyD2HFn, "cuMemcpyDtoH") orelse return false;
    cuLaunchKernel = cuda_lib.?.lookup(CuLaunchKernelFn, "cuLaunchKernel") orelse return false;
    cuStreamCreate = cuda_lib.?.lookup(CuStreamCreateFn, "cuStreamCreate") orelse return false;
    cuStreamDestroy = cuda_lib.?.lookup(CuStreamDestroyFn, "cuStreamDestroy") orelse return false;
    cuStreamSynchronize = cuda_lib.?.lookup(CuStreamSynchronizeFn, "cuStreamSynchronize") orelse return false;

    // Optional advanced functions (don't fail if not found)
    cuMemcpyD2D = cuda_lib.?.lookup(CuMemcpyD2DFn, "cuMemcpyDtoD");
    cuMemcpyD2DAsync = cuda_lib.?.lookup(CuMemcpyD2DAsyncFn, "cuMemcpyDtoDAsync");
    cuMemcpyH2DAsync = cuda_lib.?.lookup(CuMemcpyH2DAsyncFn, "cuMemcpyHtoDAsync");
    cuMemcpyD2HAsync = cuda_lib.?.lookup(CuMemcpyD2HAsyncFn, "cuMemcpyDtoHAsync");
    cuMemGetInfo = cuda_lib.?.lookup(CuMemGetInfoFn, "cuMemGetInfo");
    cuDeviceGetAttribute = cuda_lib.?.lookup(CuDeviceGetAttributeFn, "cuDeviceGetAttribute");
    cuDeviceGetName = cuda_lib.?.lookup(CuDeviceGetNameFn, "cuDeviceGetName");
    cuDeviceTotalMem = cuda_lib.?.lookup(CuDeviceTotalMemFn, "cuDeviceTotalMem");
    cuOccupancyMaxPotentialBlockSize = cuda_lib.?.lookup(CuOccupancyMaxPotentialBlockSizeFn, "cuOccupancyMaxPotentialBlockSize");
    cuFuncGetAttribute = cuda_lib.?.lookup(CuFuncGetAttributeFn, "cuFuncGetAttribute");
    cuMemsetD8 = cuda_lib.?.lookup(CuMemsetD8Fn, "cuMemsetD8");
    cuMemsetD32 = cuda_lib.?.lookup(CuMemsetD32Fn, "cuMemsetD32");

    return true;
}
