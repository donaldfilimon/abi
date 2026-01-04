//! Fallback helpers for GPU backends without native runtime bindings.
//!
//! Provides simulated kernel compilation/execution and host-backed device
//! memory primitives used by backend stubs.
const std = @import("std");
const types = @import("../kernel_types.zig");
const simulated = @import("simulated.zig");

pub const DeviceMemoryError = error{
    BufferTooSmall,
    InvalidDeviceMemory,
    OutOfMemory,
};

const DeviceAllocation = struct {
    bytes: []u8,
};

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    return simulated.compile(allocator, source);
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    return simulated.launch(allocator, kernel_handle, config, args);
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    simulated.destroy(allocator, kernel_handle);
}

pub fn createOpaqueHandle(comptime T: type, value: T) !*anyopaque {
    const handle = try std.heap.page_allocator.create(T);
    handle.* = value;
    return handle;
}

pub fn destroyOpaqueHandle(comptime T: type, handle: *anyopaque) void {
    const typed: *T = @ptrCast(@alignCast(handle));
    std.heap.page_allocator.destroy(typed);
}

pub fn allocateDeviceMemory(size: usize) DeviceMemoryError!*anyopaque {
    const allocator = std.heap.page_allocator;
    const allocation = allocator.create(DeviceAllocation) catch
        return DeviceMemoryError.OutOfMemory;
    errdefer allocator.destroy(allocation);
    const bytes = allocator.alloc(u8, size) catch
        return DeviceMemoryError.OutOfMemory;
    allocation.* = .{ .bytes = bytes };
    return allocation;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (@intFromPtr(ptr) == 0) return;
    const allocation: *DeviceAllocation = @ptrCast(@alignCast(ptr));
    std.heap.page_allocator.free(allocation.bytes);
    std.heap.page_allocator.destroy(allocation);
}

pub fn memcpyHostToDevice(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
) DeviceMemoryError!void {
    const allocation = try getAllocation(dst);
    const src_bytes: [*]const u8 = @ptrCast(@alignCast(src));
    try validateCopy(allocation.bytes.len, size);
    std.mem.copyForwards(u8, allocation.bytes[0..size], src_bytes[0..size]);
}

pub fn memcpyDeviceToHost(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
) DeviceMemoryError!void {
    const allocation = try getAllocation(src);
    const dst_bytes: [*]u8 = @ptrCast(@alignCast(dst));
    try validateCopy(allocation.bytes.len, size);
    std.mem.copyForwards(u8, dst_bytes[0..size], allocation.bytes[0..size]);
}

pub fn memcpyDeviceToDevice(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
) DeviceMemoryError!void {
    const dst_allocation = try getAllocation(dst);
    const src_allocation = try getAllocation(src);
    try validateCopy(dst_allocation.bytes.len, size);
    try validateCopy(src_allocation.bytes.len, size);
    std.mem.copyForwards(u8, dst_allocation.bytes[0..size], src_allocation.bytes[0..size]);
}

pub fn deviceSlice(ptr: *anyopaque) DeviceMemoryError![]u8 {
    const allocation = try getAllocation(ptr);
    return allocation.bytes;
}

fn getAllocation(ptr: *anyopaque) DeviceMemoryError!*DeviceAllocation {
    if (@intFromPtr(ptr) == 0) return DeviceMemoryError.InvalidDeviceMemory;
    return @ptrCast(@alignCast(ptr));
}

fn validateCopy(available: usize, size: usize) DeviceMemoryError!void {
    if (size > available) return DeviceMemoryError.BufferTooSmall;
}

test "fallback device memory copies roundtrip" {
    const data = [_]u8{ 10, 20, 30, 40 };
    const device = try allocateDeviceMemory(data.len);
    defer freeDeviceMemory(device);

    try memcpyHostToDevice(device, @ptrCast(&data[0]), data.len);

    var output = [_]u8{ 0, 0, 0, 0 };
    try memcpyDeviceToHost(@ptrCast(&output[0]), device, output.len);
    try std.testing.expectEqualSlices(u8, &data, &output);
}

test "fallback device memory device-to-device copy" {
    const a = try allocateDeviceMemory(3);
    defer freeDeviceMemory(a);
    const b = try allocateDeviceMemory(3);
    defer freeDeviceMemory(b);

    const seed = [_]u8{ 7, 8, 9 };
    try memcpyHostToDevice(a, @ptrCast(&seed[0]), seed.len);
    try memcpyDeviceToDevice(b, a, seed.len);

    var output = [_]u8{ 0, 0, 0 };
    try memcpyDeviceToHost(@ptrCast(&output[0]), b, output.len);
    try std.testing.expectEqualSlices(u8, &seed, &output);
}

test "fallback opaque handle helpers" {
    const Dummy = struct {
        value: u32,
    };
    const handle = try createOpaqueHandle(Dummy, .{ .value = 42 });
    defer destroyOpaqueHandle(Dummy, handle);
    const dummy: *Dummy = @ptrCast(@alignCast(handle));
    try std.testing.expectEqual(@as(u32, 42), dummy.value);
}
