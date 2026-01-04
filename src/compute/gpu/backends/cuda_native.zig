//! Native CUDA backend implementation with real GPU execution.
//!
//! Provides actual CUDA kernel compilation, execution, and memory management
//! using the CUDA Driver API instead of fallback simulation.

const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");

pub const CudaError = error{
    InitializationFailed,
    DriverNotFound,
    DeviceNotFound,
    OutOfMemory,
    LaunchFailed,
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

pub fn init() !void {
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
        cuCtxDestroy(context);
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
    if (cuda_context) |*ctx| {
        const destroy_fn = cuCtxDestroy orelse return;
        const stream_destroy_fn = cuStreamDestroy orelse return;

        if (ctx.stream != null) {
            stream_destroy_fn(ctx.stream);
        }

        for (ctx.device_memory.items) |mem| {
            const free_fn = cuMemFree orelse return;
            free_fn(@intCast(@intFromPtr(mem.ptr)));
        }
        ctx.device_memory.deinit(ctx.allocator);

        if (ctx.context != null) {
            destroy_fn(ctx.context);
        }
    }

    if (cuda_lib) |lib| {
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
    const sync_fn = cuStreamSynchronize orelse return CudaError.LaunchFailed;

    if (sync_fn(cu_stream.ptr) != .success) {
        return CudaError.LaunchFailed;
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

    return true;
}
