//! Buffer allocation, pooling, and lifecycle management.
//!
//! Extracted from `unified.zig` to separate buffer concerns.

const std = @import("std");
const sync = @import("shared_services").sync;
const buffer_mod = @import("unified_buffer.zig");
const device_mod = @import("device.zig");

pub const Buffer = buffer_mod.Buffer;
pub const BufferOptions = buffer_mod.BufferOptions;
pub const MemoryMode = buffer_mod.MemoryMode;
pub const MemoryLocation = buffer_mod.MemoryLocation;
pub const AccessHint = buffer_mod.AccessHint;
pub const ElementType = buffer_mod.ElementType;

const Mutex = sync.Mutex;
const Device = device_mod.Device;

/// GPU memory information.
pub const MemoryInfo = struct {
    total_bytes: u64,
    used_bytes: u64,
    free_bytes: u64,
    peak_used_bytes: u64,
};

/// GPU statistics related to buffer operations.
pub const BufferStats = struct {
    buffers_created: u64 = 0,
    bytes_allocated: u64 = 0,
    host_to_device_transfers: u64 = 0,
    device_to_host_transfers: u64 = 0,
};

/// Create a new buffer on the active device.
pub fn createBuffer(
    allocator: std.mem.Allocator,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *Mutex,
    device: *const Device,
    size: usize,
    options: BufferOptions,
    global_memory_mode: MemoryMode,
) !*Buffer {
    var opts = options;
    if (opts.mode == .automatic and global_memory_mode != .automatic) {
        opts.mode = global_memory_mode;
    }

    const buffer = try allocator.create(Buffer);
    errdefer allocator.destroy(buffer);

    buffer.* = try Buffer.init(allocator, size, device, opts);

    buffer_mutex.lock();
    defer buffer_mutex.unlock();
    try buffers.append(allocator, buffer);

    return buffer;
}

/// Create a buffer from a typed slice.
pub fn createBufferFromSlice(
    allocator: std.mem.Allocator,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *Mutex,
    device: *const Device,
    comptime T: type,
    data: []const T,
    options: BufferOptions,
    global_memory_mode: MemoryMode,
) !*Buffer {
    var opts = options;
    if (opts.mode == .automatic and global_memory_mode != .automatic) {
        opts.mode = global_memory_mode;
    }

    const buffer = try allocator.create(Buffer);
    errdefer allocator.destroy(buffer);

    buffer.* = try buffer_mod.createFromSlice(allocator, T, data, device, opts);

    buffer_mutex.lock();
    defer buffer_mutex.unlock();
    try buffers.append(allocator, buffer);

    return buffer;
}

/// Destroy a buffer and remove it from tracking.
pub fn destroyBuffer(
    allocator: std.mem.Allocator,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *Mutex,
    buffer: *Buffer,
) void {
    buffer_mutex.lock();
    defer buffer_mutex.unlock();

    for (buffers.items, 0..) |b, i| {
        if (b == buffer) {
            _ = buffers.swapRemove(i);
            break;
        }
    }

    buffer.deinit();
    allocator.destroy(buffer);
}

/// Destroy all tracked buffers.
pub fn destroyAllBuffers(
    allocator: std.mem.Allocator,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *Mutex,
) void {
    buffer_mutex.lock();
    for (buffers.items) |buf| {
        buf.deinit();
        allocator.destroy(buf);
    }
    buffers.deinit(allocator);
    buffer_mutex.unlock();
}

/// Get memory information.
pub fn getMemoryInfo(
    active_device: ?*const Device,
    buffers: *std.ArrayListUnmanaged(*Buffer),
    buffer_mutex: *Mutex,
    peak_bytes_allocated: u64,
) MemoryInfo {
    var total: u64 = 0;
    var used: u64 = 0;

    if (active_device) |device| {
        total = device.total_memory orelse 0;
    }

    buffer_mutex.lock();
    defer buffer_mutex.unlock();
    for (buffers.items) |buf| {
        used += buf.getSize();
    }

    return .{
        .total_bytes = total,
        .used_bytes = used,
        .free_bytes = if (total > used) total - used else 0,
        .peak_used_bytes = peak_bytes_allocated,
    };
}
