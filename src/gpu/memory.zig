//! GPU memory management
//!
//! Provides buffer allocation, simulated device transfers, and pool helpers.
const std = @import("std");

pub const MemoryError = error{
    BufferTooSmall,
    InvalidOffset,
    HostAccessDisabled,
    DeviceMemoryMissing,
    SizeMismatch,
};

pub const BufferFlags = struct {
    read_only: bool = false,
    write_only: bool = false,
    host_visible: bool = true,
    device_local: bool = false,
    zero_init: bool = false,
};

pub const GpuBuffer = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    device_bytes: ?[]u8,
    device_ptr: ?*anyopaque,
    size: usize,
    flags: BufferFlags,
    host_dirty: bool,
    device_dirty: bool,

    pub fn init(allocator: std.mem.Allocator, size: usize, flags: BufferFlags) !GpuBuffer {
        const host_bytes = try allocator.alloc(u8, size);
        if (flags.zero_init) {
            @memset(host_bytes, 0);
        }

        var device_bytes: ?[]u8 = null;
        var device_ptr: ?*anyopaque = null;
        if (flags.device_local) {
            const allocated = try allocator.alloc(u8, size);
            if (flags.zero_init) {
                @memset(allocated, 0);
            }
            device_bytes = allocated;
            device_ptr = allocated.ptr;
        }

        return .{
            .allocator = allocator,
            .bytes = host_bytes,
            .device_bytes = device_bytes,
            .device_ptr = device_ptr,
            .size = size,
            .flags = flags,
            .host_dirty = false,
            .device_dirty = false,
        };
    }

    pub fn deinit(self: *GpuBuffer) void {
        if (self.device_bytes) |device_bytes| {
            self.allocator.free(device_bytes);
        }
        self.allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn len(self: *const GpuBuffer) usize {
        return self.bytes.len;
    }

    pub fn fill(self: *GpuBuffer, value: u8) MemoryError!void {
        if (!self.flags.host_visible) return MemoryError.HostAccessDisabled;
        @memset(self.bytes, value);
        self.host_dirty = true;
        if (self.device_bytes != null) self.device_dirty = true;
    }

    pub fn copyFrom(self: *GpuBuffer, data: []const u8) MemoryError!void {
        if (!self.flags.host_visible) return MemoryError.HostAccessDisabled;
        if (data.len != self.bytes.len) return MemoryError.SizeMismatch;
        std.mem.copyForwards(u8, self.bytes, data);
        self.host_dirty = true;
        if (self.device_bytes != null) self.device_dirty = true;
    }

    pub fn writeFromHost(self: *GpuBuffer, data: []const u8) MemoryError!void {
        if (!self.flags.host_visible) return MemoryError.HostAccessDisabled;
        if (data.len > self.bytes.len) return MemoryError.BufferTooSmall;
        @memcpy(self.bytes[0..data.len], data);
        self.host_dirty = true;
        if (self.device_bytes != null) self.device_dirty = true;
    }

    pub fn readToHost(self: *const GpuBuffer, offset: usize, size: usize) MemoryError![]const u8 {
        if (!self.flags.host_visible) return MemoryError.HostAccessDisabled;
        if (offset + size > self.bytes.len) return MemoryError.InvalidOffset;
        return self.bytes[offset..][0..size];
    }

    pub fn copyToDevice(self: *GpuBuffer) MemoryError!void {
        const device_bytes = self.device_bytes orelse return MemoryError.DeviceMemoryMissing;
        std.mem.copyForwards(u8, device_bytes, self.bytes);
        self.device_dirty = false;
        self.host_dirty = false;
    }

    pub fn copyToHost(self: *GpuBuffer) MemoryError!void {
        const device_bytes = self.device_bytes orelse return MemoryError.DeviceMemoryMissing;
        if (!self.flags.host_visible) return MemoryError.HostAccessDisabled;
        std.mem.copyForwards(u8, self.bytes, device_bytes);
        self.host_dirty = false;
        self.device_dirty = false;
    }

    pub fn asSlice(self: *GpuBuffer) []u8 {
        return self.bytes;
    }

    pub fn asConstSlice(self: *const GpuBuffer) []const u8 {
        return self.bytes;
    }
};

pub const MemoryStats = struct {
    total_size: usize,
    max_size: usize,
    buffer_count: usize,
    usage_ratio: f64,
};

pub const GpuMemoryPool = struct {
    buffers: std.ArrayListUnmanaged(*GpuBuffer),
    allocator: std.mem.Allocator,
    total_size: usize,
    max_size: usize,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) GpuMemoryPool {
        return .{
            .buffers = std.ArrayListUnmanaged(*GpuBuffer).empty,
            .allocator = allocator,
            .total_size = 0,
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *GpuMemoryPool) void {
        for (self.buffers.items) |buffer| {
            buffer.deinit();
            self.allocator.destroy(buffer);
        }
        self.buffers.deinit(self.allocator);
    }

    pub fn allocate(self: *GpuMemoryPool, size: usize, flags: BufferFlags) !*GpuBuffer {
        if (self.max_size > 0 and self.total_size + size > self.max_size) {
            return error.OutOfMemory;
        }

        const buffer = try self.allocator.create(GpuBuffer);
        errdefer self.allocator.destroy(buffer);
        buffer.* = try GpuBuffer.init(self.allocator, size, flags);
        try self.buffers.append(self.allocator, buffer);
        self.total_size += size;
        return buffer;
    }

    pub fn free(self: *GpuMemoryPool, buffer: *GpuBuffer) bool {
        var i: usize = 0;
        while (i < self.buffers.items.len) : (i += 1) {
            if (self.buffers.items[i] == buffer) {
                std.debug.assert(self.total_size >= buffer.size);
                self.total_size -= buffer.size;
                buffer.deinit();
                self.allocator.destroy(buffer);
                _ = self.buffers.orderedRemove(i);
                return true;
            }
        }
        return false;
    }

    pub fn stats(self: *const GpuMemoryPool) MemoryStats {
        return .{
            .total_size = self.total_size,
            .max_size = self.max_size,
            .buffer_count = self.buffers.items.len,
            .usage_ratio = self.getUsage(),
        };
    }

    pub fn getUsage(self: *const GpuMemoryPool) f64 {
        if (self.max_size == 0) return 0;
        return @as(f64, @floatFromInt(self.total_size)) /
            @as(f64, @floatFromInt(self.max_size));
    }
};

pub const AsyncTransfer = struct {
    source: *const GpuBuffer,
    destination: *GpuBuffer,
    size: usize,
    offset: usize,
    completed: std.atomic.Value(bool),

    pub fn init(
        source: *const GpuBuffer,
        destination: *GpuBuffer,
        size: usize,
        offset: usize,
    ) AsyncTransfer {
        return .{
            .source = source,
            .destination = destination,
            .size = size,
            .offset = offset,
            .completed = std.atomic.Value(bool).init(false),
        };
    }

    pub fn start(self: *AsyncTransfer) MemoryError!void {
        if (self.offset + self.size > self.source.bytes.len) return MemoryError.InvalidOffset;
        if (self.offset + self.size > self.destination.bytes.len) return MemoryError.InvalidOffset;
        const src = self.source.bytes[self.offset..][0..self.size];
        const dst = self.destination.bytes[self.offset..][0..self.size];
        @memcpy(dst, src);
        self.completed.store(true, .release);
    }

    pub fn wait(self: *AsyncTransfer) void {
        while (!self.completed.load(.acquire)) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn isComplete(self: *const AsyncTransfer) bool {
        return self.completed.load(.acquire);
    }
};

test "buffer host operations" {
    var buffer = try GpuBuffer.init(std.testing.allocator, 4, .{});
    defer buffer.deinit();

    try buffer.fill(0xaa);
    try std.testing.expectEqualSlices(u8, &.{ 0xaa, 0xaa, 0xaa, 0xaa }, buffer.bytes);

    try buffer.copyFrom(&.{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, buffer.bytes);
    const slice = try buffer.readToHost(1, 2);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3 }, slice);
}

test "buffer device copy" {
    var buffer = try GpuBuffer.init(std.testing.allocator, 3, .{ .device_local = true });
    defer buffer.deinit();

    try buffer.writeFromHost(&.{ 9, 8, 7 });
    try buffer.copyToDevice();
    buffer.bytes[0] = 1;
    try buffer.copyToHost();
    try std.testing.expectEqualSlices(u8, &.{ 9, 8, 7 }, buffer.bytes);
}

test "buffer dirty flags track synchronization" {
    var buffer = try GpuBuffer.init(std.testing.allocator, 2, .{ .device_local = true });
    defer buffer.deinit();

    try buffer.fill(0xaa);
    try std.testing.expect(buffer.host_dirty);
    try std.testing.expect(buffer.device_dirty);

    try buffer.copyToDevice();
    try std.testing.expect(!buffer.host_dirty);
    try std.testing.expect(!buffer.device_dirty);

    try buffer.writeFromHost(&.{ 1, 2 });
    try std.testing.expect(buffer.host_dirty);
    try std.testing.expect(buffer.device_dirty);
}

test "memory pool allocates and frees" {
    var pool = GpuMemoryPool.init(std.testing.allocator, 16);
    defer pool.deinit();

    const a = try pool.allocate(4, .{});
    const b = try pool.allocate(6, .{});
    try std.testing.expectEqual(@as(usize, 2), pool.stats().buffer_count);
    try std.testing.expect(pool.free(a));
    try std.testing.expect(pool.free(b));
}

test "async transfer copies data" {
    var source = try GpuBuffer.init(std.testing.allocator, 4, .{});
    defer source.deinit();
    var dest = try GpuBuffer.init(std.testing.allocator, 4, .{});
    defer dest.deinit();

    try source.copyFrom(&.{ 4, 3, 2, 1 });
    var transfer = AsyncTransfer.init(&source, &dest, 4, 0);
    try transfer.start();
    transfer.wait();
    try std.testing.expectEqualSlices(u8, &.{ 4, 3, 2, 1 }, dest.bytes);
}
