//! CUDA backend implementation
//!
//! Provides CUDA-specific kernel compilation and execution.

const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");
const cuda_native = @import("cuda_native.zig");
const gpu = std.gpu;
const CuResult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_context = 6,
    launch_failure = 700,
    launch_out_of_resources = 701,
};

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

const CudaContext = struct {
    device_id: i32,
    context: ?*anyopaque,
    stream: ?*anyopaque,
    device_memory: std.ArrayListUnmanaged([]u8),
};

var cuda_initialized = false;
var cuda_context: ?CudaContext = null;
var use_native: bool = false;

pub fn init() !void {
    if (cuda_initialized) return;

    const cuda_available = tryLoadCuda();
    if (cuda_available) {
        if (loadCudaFunctions()) {
            if (cuInit) |init_fn| {
                _ = init_fn(0);
            }
            var device_count: i32 = 0;
            if (cuDeviceGetCount) |count_fn| {
                _ = count_fn(&device_count);
            }
        } else {
            std.log.warn("CUDA runtime not available, using simulation mode", .{});
        }
    } else {
        std.log.warn("CUDA runtime not available, using simulation mode", .{});
    }

    cuda_context = CudaContext{
        .device_id = 0,
        .context = null,
        .stream = null,
        .device_memory = std.ArrayListUnmanaged([]u8).empty,
    };

    cuda_initialized = true;
}

pub fn deinit() void {
    if (use_native) {
        cuda_native.deinit();
        use_native = false;
    }
    if (cuda_context) |*ctx| {
        _ = ctx;
    }
    cuda_context = null;
    cuda_initialized = false;
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

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (use_native) {
        cuda_native.destroyKernel(allocator, kernel_handle);
        return;
    }
    fallback.destroyKernel(allocator, kernel_handle);
}

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

pub fn destroyStream(stream: *anyopaque) void {
    if (use_native) {
        cuda_native.destroyStream(stream);
        return;
    }
    fallback.destroyOpaqueHandle(CuStream, stream);
}

pub fn synchronizeStream(stream: *anyopaque) !void {
    if (use_native) {
        return cuda_native.synchronizeStream(stream);
    }
    const cu_stream: *CuStream = @ptrCast(@alignCast(stream));
    _ = cu_stream;
}

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

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (use_native) {
        cuda_native.freeDeviceMemory(ptr);
        return;
    }
    fallback.freeDeviceMemory(ptr);
}

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

fn tryLoadCuda() bool {
    const lib_names = [_][]const u8{ "nvcuda.dll", "libcuda.so.1", "libcuda.so" };
    return shared.tryLoadAny(lib_names[0..]);
}

const CuInitFn = *const fn (u32) callconv(.c) CuResult;
const CuDeviceGetCountFn = *const fn (*i32) callconv(.c) CuResult;
const CuMemAllocFn = *const fn (*?*anyopaque, usize) callconv(.c) CuResult;
const CuMemFreeFn = *const fn (*anyopaque) callconv(.c) CuResult;
const CuMemcpyH2DFn = *const fn (*anyopaque, *const anyopaque, usize) callconv(.c) CuResult;
const CuMemcpyD2HFn = *const fn (*anyopaque, *const anyopaque, usize) callconv(.c) CuResult;

var cuInit: ?CuInitFn = null;
var cuDeviceGetCount: ?CuDeviceGetCountFn = null;
var cuMemAlloc: ?CuMemAllocFn = null;
var cuMemFree: ?CuMemFreeFn = null;
var cuMemcpyH2D: ?CuMemcpyH2DFn = null;
var cuMemcpyD2H: ?CuMemcpyD2HFn = null;

fn loadCudaFunctions() bool {
    var lib = tryLoadCudaLib() orelse return false;
    defer lib.close();

    cuInit = lib.lookup(CuInitFn, "cuInit") orelse return false;
    cuDeviceGetCount = lib.lookup(CuDeviceGetCountFn, "cuDeviceGetCount") orelse return false;
    cuMemAlloc = lib.lookup(CuMemAllocFn, "cuMemAlloc") orelse return false;
    cuMemFree = lib.lookup(CuMemFreeFn, "cuMemFree") orelse return false;
    cuMemcpyH2D = lib.lookup(CuMemcpyH2DFn, "cuMemcpyHtoD") orelse return false;
    cuMemcpyD2H = lib.lookup(CuMemcpyD2HFn, "cuMemcpyDtoH") orelse return false;
    return true;
}

fn tryLoadCudaLib() ?std.DynLib {
    const lib_names = [_][]const u8{ "nvcuda.dll", "libcuda.so.1", "libcuda.so" };
    return shared.openFirst(lib_names[0..]);
}
