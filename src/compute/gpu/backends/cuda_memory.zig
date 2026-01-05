//! Native memory management for CUDA with proper device allocation.
//!
//! Implements real CUDA device memory allocation, pinned host memory,
//! and efficient transfer operations.

const std = @import("std");

pub const MemoryError = error{
    AllocationFailed,
    FreeFailed,
    CopyFailed,
    InvalidPointer,
    OutOfMemory,
    InitializationFailed,
};

const CUresult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_context = 6,
    invalid_device = 101,
    launch_failed = 700,
};

const CUdeviceptr = usize;
const CUstream = *anyopaque;

const CuMemAllocFn = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
const CuMemFreeFn = *const fn (CUdeviceptr) callconv(.c) CUresult;
const CuMemAllocHostFn = *const fn (*[*]u8, usize) callconv(.c) CUresult;
const CuMemFreeHostFn = *const fn ([*]u8) callconv(.c) CUresult;
const CuMemcpyH2DFn = *const fn (CUdeviceptr, [*]const u8, usize) callconv(.c) CUresult;
const CuMemcpyD2HFn = *const fn ([*]u8, CUdeviceptr, usize) callconv(.c) CUresult;
const CuMemcpyD2DFn = *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult;
const CuMemcpyH2DAsyncFn = *const fn (CUdeviceptr, [*]const u8, usize, CUstream) callconv(.c) CUresult;
const CuMemcpyD2HAsyncFn = *const fn ([*]u8, CUdeviceptr, usize, CUstream) callconv(.c) CUresult;

var cuMemAlloc: ?CuMemAllocFn = null;
var cuMemFree: ?CuMemFreeFn = null;
var cuMemAllocHost: ?CuMemAllocHostFn = null;
var cuMemFreeHost: ?CuMemFreeHostFn = null;
var cuMemcpyH2D: ?CuMemcpyH2DFn = null;
var cuMemcpyD2H: ?CuMemcpyD2HFn = null;
var cuMemcpyD2D: ?CuMemcpyD2DFn = null;
var cuMemcpyH2DAsync: ?CuMemcpyH2DAsyncFn = null;
var cuMemcpyD2HAsync: ?CuMemcpyD2HAsyncFn = null;

var cuda_lib: ?std.DynLib = null;
var initialized = false;

fn checkResult(result: CUresult) MemoryError!void {
    return switch (result) {
        .success => {},
        .invalid_value => MemoryError.InvalidPointer,
        .out_of_memory => MemoryError.OutOfMemory,
        .not_initialized => MemoryError.InitializationFailed,
        .invalid_context => MemoryError.InitializationFailed,
        .invalid_device => MemoryError.InvalidPointer,
        .launch_failed => MemoryError.CopyFailed,
        else => MemoryError.AllocationFailed,
    };
}

pub fn init() !void {
    if (initialized) return;

    const lib_names = [_][]const u8{ "nvcuda.dll", "libcuda.so.1", "libcuda.so" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            cuda_lib = lib;
            break;
        } else |_| {}
    }

    if (cuda_lib == null) {
        return MemoryError.InitializationFailed;
    }

    cuMemAlloc = cuda_lib.?.lookup(CuMemAllocFn, "cuMemAlloc") orelse return MemoryError.InitializationFailed;
    cuMemFree = cuda_lib.?.lookup(CuMemFreeFn, "cuMemFree") orelse return MemoryError.InitializationFailed;
    cuMemAllocHost = cuda_lib.?.lookup(CuMemAllocHostFn, "cuMemAllocHost") orelse return MemoryError.InitializationFailed;
    cuMemFreeHost = cuda_lib.?.lookup(CuMemFreeHostFn, "cuMemFreeHost") orelse return MemoryError.InitializationFailed;
    cuMemcpyH2D = cuda_lib.?.lookup(CuMemcpyH2DFn, "cuMemcpyHtoD") orelse return MemoryError.InitializationFailed;
    cuMemcpyD2H = cuda_lib.?.lookup(CuMemcpyD2HFn, "cuMemcpyDtoH") orelse return MemoryError.InitializationFailed;
    cuMemcpyD2D = cuda_lib.?.lookup(CuMemcpyD2DFn, "cuMemcpyDtoD") orelse return MemoryError.InitializationFailed;
    cuMemcpyH2DAsync = cuda_lib.?.lookup(CuMemcpyH2DAsyncFn, "cuMemcpyHtoDAsync") orelse return MemoryError.InitializationFailed;
    cuMemcpyD2HAsync = cuda_lib.?.lookup(CuMemcpyD2HAsyncFn, "cuMemcpyDtoHAsync") orelse return MemoryError.InitializationFailed;

    initialized = true;
}

pub fn deinit() void {
    if (cuda_lib) |*lib| {
        lib.close();
    }
    cuda_lib = null;
    initialized = false;
}

pub const DeviceMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) MemoryError!DeviceMemory {
        if (!initialized) {
            return MemoryError.InitializationFailed;
        }

        const alloc_fn = cuMemAlloc orelse return MemoryError.InitializationFailed;
        var device_ptr: CUdeviceptr = 0;

        const result = alloc_fn(&device_ptr, size);
        try checkResult(result);

        return .{
            .ptr = @ptrFromInt(device_ptr),
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DeviceMemory) void {
        if (self.ptr != null) {
            const device_ptr: CUdeviceptr = @intFromPtr(self.ptr);
            if (cuMemFree) |free_fn| {
                _ = free_fn(device_ptr);
            }
        }
        self.* = undefined;
    }
};

pub const PinnedMemory = struct {
    ptr: ?[*]u8,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) MemoryError!PinnedMemory {
        if (!initialized) {
            return MemoryError.InitializationFailed;
        }

        const alloc_fn = cuMemAllocHost orelse return MemoryError.InitializationFailed;
        var host_ptr: [*]u8 = undefined;

        const result = alloc_fn(&host_ptr, size);
        try checkResult(result);

        return .{
            .ptr = host_ptr,
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PinnedMemory) void {
        if (self.ptr != null) {
            if (cuMemFreeHost) |free_fn| {
                _ = free_fn(self.ptr.?);
            }
        }
        self.* = undefined;
    }

    pub fn slice(self: *const PinnedMemory) []u8 {
        return self.ptr.?[0..self.size];
    }
};

pub const MemoryPool = struct {
    allocations: std.ArrayListUnmanaged(DeviceMemory),
    total_size: usize,
    max_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) MemoryPool {
        return .{
            .allocations = std.ArrayListUnmanaged(DeviceMemory).empty,
            .total_size = 0,
            .max_size = max_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        for (self.allocations.items) |*alloc| {
            alloc.deinit();
        }
        self.allocations.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn allocate(self: *MemoryPool, size: usize) MemoryError!*DeviceMemory {
        if (self.max_size > 0 and self.total_size + size > self.max_size) {
            return MemoryError.OutOfMemory;
        }

        const mem = try self.allocator.create(DeviceMemory);
        mem.* = try DeviceMemory.init(self.allocator, size);
        try self.allocations.append(self.allocator, mem.*);
        self.total_size += size;
        return mem;
    }

    pub fn free(self: *MemoryPool, mem: *DeviceMemory) bool {
        for (self.allocations.items, 0..) |*item, i| {
            if (item.ptr == mem.ptr) {
                std.debug.assert(self.total_size >= item.size);
                self.total_size -= item.size;
                item.deinit();
                _ = self.allocations.orderedRemove(i);
                mem.deinit();
                self.allocator.destroy(mem);
                return true;
            }
        }
        return false;
    }

    pub fn stats(self: *const MemoryPool) struct {
        total_size: usize,
        max_size: usize,
        allocation_count: usize,
        usage_ratio: f64,
    } {
        const usage = if (self.max_size > 0)
            @as(f64, @floatFromInt(self.total_size)) / @as(f64, @floatFromInt(self.max_size))
        else
            0.0;
        return .{
            .total_size = self.total_size,
            .max_size = self.max_size,
            .allocation_count = self.allocations.items.len,
            .usage_ratio = usage,
        };
    }
};

pub fn DeviceMemoryInit(allocator: std.mem.Allocator, size: usize) MemoryError!DeviceMemory {
    return DeviceMemory.init(allocator, size);
}

pub fn DeviceMemoryDeinit(self: *DeviceMemory) void {
    self.deinit();
}

pub fn allocatePinned(allocator: std.mem.Allocator, size: usize) MemoryError!PinnedMemory {
    return PinnedMemory.init(allocator, size);
}

pub fn freePinned(self: *PinnedMemory) void {
    self.deinit();
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const copy_fn = cuMemcpyH2D orelse return MemoryError.InitializationFailed;
    const dst_ptr: CUdeviceptr = @intFromPtr(dst);
    const src_ptr: [*]const u8 = @ptrCast(@alignCast(src));

    const result = copy_fn(dst_ptr, src_ptr, size);
    try checkResult(result);
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const copy_fn = cuMemcpyD2H orelse return MemoryError.InitializationFailed;
    const dst_ptr: [*]u8 = @ptrCast(@alignCast(dst));
    const src_ptr: CUdeviceptr = @intFromPtr(src);

    const result = copy_fn(dst_ptr, src_ptr, size);
    try checkResult(result);
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const copy_fn = cuMemcpyD2D orelse return MemoryError.InitializationFailed;
    const dst_ptr: CUdeviceptr = @intFromPtr(dst);
    const src_ptr: CUdeviceptr = @intFromPtr(src);

    const result = copy_fn(dst_ptr, src_ptr, size);
    try checkResult(result);
}

pub fn memcpyAsync(
    dst: *anyopaque,
    src: *anyopaque,
    size: usize,
    stream: ?*anyopaque,
) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const stream_ptr: CUstream = @ptrCast(stream orelse null);

    const is_h2d = std.mem.isAligned(@intFromPtr(dst), 8);
    const is_d2h = std.mem.isAligned(@intFromPtr(src), 8);

    if (is_h2d and !is_d2h) {
        const copy_fn = cuMemcpyH2DAsync orelse return MemoryError.InitializationFailed;
        const dst_ptr: CUdeviceptr = @intFromPtr(dst);
        const src_ptr: [*]const u8 = @ptrCast(@alignCast(src));

        const result = copy_fn(dst_ptr, src_ptr, size, stream_ptr);
        try checkResult(result);
    } else if (!is_h2d and is_d2h) {
        const copy_fn = cuMemcpyD2HAsync orelse return MemoryError.InitializationFailed;
        const dst_ptr: [*]u8 = @ptrCast(@alignCast(dst));
        const src_ptr: CUdeviceptr = @intFromPtr(src);

        const result = copy_fn(dst_ptr, src_ptr, size, stream_ptr);
        try checkResult(result);
    } else {
        return MemoryError.CopyFailed;
    }
}
