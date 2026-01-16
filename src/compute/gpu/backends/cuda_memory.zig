//! Native memory management for CUDA with proper device allocation.
//!
//! Implements real CUDA device memory allocation, pinned host memory,
//! and efficient transfer operations. Uses consolidated cuda_loader.

const std = @import("std");
const cuda_loader = @import("cuda_loader.zig");

pub const MemoryError = error{
    AllocationFailed,
    FreeFailed,
    CopyFailed,
    InvalidPointer,
    OutOfMemory,
    InitializationFailed,
};

// Re-use types from loader
const CuResult = cuda_loader.CuResult;
const CUdeviceptr = u64;
const CUstream = *anyopaque;

var initialized = false;

fn checkResult(result: CuResult) MemoryError!void {
    return switch (result) {
        .success => {},
        .invalid_value => MemoryError.InvalidPointer,
        .out_of_memory => MemoryError.OutOfMemory,
        .not_initialized => MemoryError.InitializationFailed,
        .invalid_context => MemoryError.InitializationFailed,
        else => MemoryError.AllocationFailed,
    };
}

/// Get memory functions from consolidated loader
fn getMemoryFuncs() ?*const cuda_loader.MemoryFunctions {
    const funcs = cuda_loader.getFunctions() orelse return null;
    return &funcs.memory;
}

pub fn init() !void {
    if (initialized) return;

    // Use consolidated loader
    _ = cuda_loader.load() catch return MemoryError.InitializationFailed;

    const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
    if (mem_funcs.cuMemAlloc == null or mem_funcs.cuMemFree == null) {
        return MemoryError.InitializationFailed;
    }

    initialized = true;
}

pub fn deinit() void {
    // Don't unload - let main cuda.zig handle that
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

        const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
        const alloc_fn = mem_funcs.cuMemAlloc orelse return MemoryError.InitializationFailed;
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
            const mem_funcs = getMemoryFuncs() orelse return;
            if (mem_funcs.cuMemFree) |free_fn| {
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

        const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
        const alloc_fn = mem_funcs.cuMemAllocHost orelse return MemoryError.InitializationFailed;
        var host_ptr: *anyopaque = undefined;

        const result = alloc_fn(&host_ptr, size);
        try checkResult(result);

        return .{
            .ptr = @ptrCast(host_ptr),
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PinnedMemory) void {
        if (self.ptr != null) {
            const mem_funcs = getMemoryFuncs() orelse return;
            if (mem_funcs.cuMemFreeHost) |free_fn| {
                _ = free_fn(@ptrCast(self.ptr.?));
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
                // Remove from tracking list first (item is a copy, don't deinit it)
                _ = self.allocations.orderedRemove(i);
                // Then deinit and destroy the actual memory
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

    const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
    const copy_fn = mem_funcs.cuMemcpyHtoD orelse return MemoryError.InitializationFailed;
    const dst_ptr: CUdeviceptr = @intFromPtr(dst);

    const result = copy_fn(dst_ptr, src, size);
    try checkResult(result);
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
    const copy_fn = mem_funcs.cuMemcpyDtoH orelse return MemoryError.InitializationFailed;
    const dst_ptr: [*]u8 = @ptrCast(@alignCast(dst));
    const src_ptr: CUdeviceptr = @intFromPtr(src);

    const result = copy_fn(dst_ptr, src_ptr, size);
    try checkResult(result);
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) MemoryError!void {
    if (!initialized) {
        return MemoryError.InitializationFailed;
    }

    const mem_funcs = getMemoryFuncs() orelse return MemoryError.InitializationFailed;
    const copy_fn = mem_funcs.cuMemcpyDtoD orelse return MemoryError.InitializationFailed;
    const dst_ptr: CUdeviceptr = @intFromPtr(dst);
    const src_ptr: CUdeviceptr = @intFromPtr(src);

    const result = copy_fn(dst_ptr, src_ptr, size);
    try checkResult(result);
}
