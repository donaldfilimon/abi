//! Metal buffer creation, management, and memory transfer operations.
//!
//! Handles allocating device memory (Metal buffers), freeing buffers,
//! and host-to-device / device-to-host / device-to-device memory copies.

const std = @import("std");
const s = @import("metal_state");
const metal_types = @import("metal_types");

/// Safely cast an opaque pointer to a MetalBuffer pointer with validation.
/// Returns null if the pointer is null or the magic value doesn't match.
fn safeCastToBuffer(ptr: ?*anyopaque) ?*s.MetalBuffer {
    const p = ptr orelse return null;
    const safe_buffer: *s.SafeMetalBuffer = @ptrCast(@alignCast(p));
    if (safe_buffer.magic != s.buffer_magic) {
        std.log.err("Invalid MetalBuffer pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ s.buffer_magic, safe_buffer.magic });
        return null;
    }
    return &safe_buffer.inner;
}

/// Safely cast a const opaque pointer to a MetalBuffer pointer with validation.
fn safeCastToBufferConst(ptr: ?*const anyopaque) ?*const s.MetalBuffer {
    const p = ptr orelse return null;
    const safe_buffer: *const s.SafeMetalBuffer = @ptrCast(@alignCast(p));
    if (safe_buffer.magic != s.buffer_magic) {
        std.log.err("Invalid MetalBuffer pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ s.buffer_magic, safe_buffer.magic });
        return null;
    }
    return &safe_buffer.inner;
}

pub fn allocateDeviceMemory(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    const actual_allocator = s.buffer_allocator orelse allocator;
    return allocateDeviceMemoryWithAllocator(actual_allocator, size);
}

pub fn allocateDeviceMemoryWithAllocator(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    if (!s.metal_initialized or s.metal_device == null) {
        return s.MetalError.BufferCreationFailed;
    }

    const device = s.metal_device.?;

    const msg_send_int = s.objc_msgSend_int orelse return s.MetalError.BufferCreationFailed;

    // [device newBufferWithLength:size options:MTLResourceStorageModeShared]
    const buffer = msg_send_int(device, s.sel_newBufferWithLength, size, s.MTLResourceStorageModeShared);
    if (buffer == null) {
        std.log.err("Failed to create Metal buffer of size {B}", .{size});
        return s.MetalError.BufferCreationFailed;
    }

    const safe_buffer = try allocator.create(s.SafeMetalBuffer);
    errdefer allocator.destroy(safe_buffer);

    safe_buffer.* = .{
        .magic = s.buffer_magic,
        .inner = .{
            .buffer = buffer,
            .size = size,
            .allocator = allocator,
        },
    };

    std.log.debug("Metal buffer allocated: size={B}", .{size});
    return safe_buffer;
}

pub fn freeDeviceMemory(allocator: std.mem.Allocator, ptr: *anyopaque) void {
    _ = allocator;

    const buffer = safeCastToBuffer(ptr) orelse {
        std.log.err("freeDeviceMemory: Invalid buffer pointer (null or corrupted), skipping free", .{});
        return;
    };
    const buffer_allocator_ref = buffer.allocator;

    if (s.objc_msgSend_void) |release_fn| {
        if (buffer.buffer != null) {
            release_fn(buffer.buffer, s.sel_release);
        }
    }

    const safe_buffer: *s.SafeMetalBuffer = @fieldParentPtr("inner", buffer);
    safe_buffer.magic = 0;

    buffer_allocator_ref.destroy(safe_buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const dst_buffer = safeCastToBuffer(dst) orelse {
        std.log.err("memcpyHostToDevice: Invalid destination buffer (null or corrupted)", .{});
        return s.MetalError.MemoryCopyFailed;
    };

    if (size > dst_buffer.size) {
        std.log.err("memcpyHostToDevice: Copy size ({}) exceeds buffer size ({})", .{ size, dst_buffer.size });
        return s.MetalError.MemoryCopyFailed;
    }

    const msg_send = s.objc_msgSend orelse return s.MetalError.MemoryCopyFailed;
    const contents = msg_send(dst_buffer.buffer, s.sel_contents);
    if (contents == null) {
        std.log.err("Failed to get Metal buffer contents for host->device copy", .{});
        return s.MetalError.MemoryCopyFailed;
    }

    const dst_slice = @as([*]u8, @ptrCast(contents.?))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(src))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy host->device: {B}", .{size});
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const src_buffer = safeCastToBuffer(src) orelse {
        std.log.err("memcpyDeviceToHost: Invalid source buffer (null or corrupted)", .{});
        return s.MetalError.MemoryCopyFailed;
    };

    if (size > src_buffer.size) {
        std.log.err("memcpyDeviceToHost: Copy size ({}) exceeds buffer size ({})", .{ size, src_buffer.size });
        return s.MetalError.MemoryCopyFailed;
    }

    const msg_send = s.objc_msgSend orelse return s.MetalError.MemoryCopyFailed;
    const contents = msg_send(src_buffer.buffer, s.sel_contents);
    if (contents == null) {
        std.log.err("Failed to get Metal buffer contents for device->host copy", .{});
        return s.MetalError.MemoryCopyFailed;
    }

    const dst_slice = @as([*]u8, @ptrCast(dst))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(contents.?))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy device->host: {B}", .{size});
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const src_buffer = safeCastToBuffer(src) orelse {
        std.log.err("memcpyDeviceToDevice: Invalid source buffer (null or corrupted)", .{});
        return s.MetalError.MemoryCopyFailed;
    };
    const dst_buffer = safeCastToBuffer(dst) orelse {
        std.log.err("memcpyDeviceToDevice: Invalid destination buffer (null or corrupted)", .{});
        return s.MetalError.MemoryCopyFailed;
    };

    if (size > src_buffer.size) {
        std.log.err("memcpyDeviceToDevice: Copy size ({}) exceeds source buffer size ({})", .{ size, src_buffer.size });
        return s.MetalError.MemoryCopyFailed;
    }
    if (size > dst_buffer.size) {
        std.log.err("memcpyDeviceToDevice: Copy size ({}) exceeds destination buffer size ({})", .{ size, dst_buffer.size });
        return s.MetalError.MemoryCopyFailed;
    }

    const msg_send = s.objc_msgSend orelse return s.MetalError.MemoryCopyFailed;
    const src_contents = msg_send(src_buffer.buffer, s.sel_contents);
    const dst_contents = msg_send(dst_buffer.buffer, s.sel_contents);

    if (src_contents == null or dst_contents == null) {
        std.log.err("Failed to get Metal buffer contents for device->device copy", .{});
        return s.MetalError.MemoryCopyFailed;
    }

    const dst_slice = @as([*]u8, @ptrCast(dst_contents.?))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(src_contents.?))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy device->device: {B}", .{size});
}
