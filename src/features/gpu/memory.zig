//! GPU memory management
//!
//! Provides buffer allocation, simulated device transfers, and pool helpers.
const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

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
    /// HashMap for O(1) buffer lookup by pointer address
    buffer_lookup: std.AutoHashMapUnmanaged(usize, usize),
    allocator: std.mem.Allocator,
    total_size: usize,
    max_size: usize,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) GpuMemoryPool {
        return .{
            .buffers = std.ArrayListUnmanaged(*GpuBuffer).empty,
            .buffer_lookup = std.AutoHashMapUnmanaged(usize, usize){},
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
        self.buffer_lookup.deinit(self.allocator);
    }

    pub fn allocate(self: *GpuMemoryPool, size: usize, flags: BufferFlags) !*GpuBuffer {
        if (self.max_size > 0 and self.total_size + size > self.max_size) {
            return error.OutOfMemory;
        }

        const buffer = try self.allocator.create(GpuBuffer);
        errdefer self.allocator.destroy(buffer);
        buffer.* = try GpuBuffer.init(self.allocator, size, flags);
        errdefer buffer.deinit();

        // Insert into lookup map before appending to buffers array
        const buffer_addr = @intFromPtr(buffer);
        const buffer_idx = self.buffers.items.len;
        try self.buffer_lookup.put(self.allocator, buffer_addr, buffer_idx);
        errdefer _ = self.buffer_lookup.remove(buffer_addr);

        try self.buffers.append(self.allocator, buffer);
        self.total_size += size;
        return buffer;
    }

    /// Free a buffer from the pool.
    /// Uses O(1) hash map lookup and swapRemove for efficient removal.
    pub fn free(self: *GpuMemoryPool, buffer: *GpuBuffer) bool {
        const buffer_addr = @intFromPtr(buffer);

        // O(1) lookup using hash map
        const idx = self.buffer_lookup.get(buffer_addr) orelse return false;

        std.debug.assert(self.total_size >= buffer.size);
        self.total_size -= buffer.size;

        // Remove from lookup map
        _ = self.buffer_lookup.remove(buffer_addr);

        // If we're not removing the last element, update the swapped element's index
        const last_idx = self.buffers.items.len - 1;
        if (idx != last_idx) {
            const swapped_buffer = self.buffers.items[last_idx];
            const swapped_addr = @intFromPtr(swapped_buffer);
            // Update the swapped buffer's index in the lookup map
            if (self.buffer_lookup.getPtr(swapped_addr)) |entry| {
                entry.* = idx;
            }
        }

        buffer.deinit();
        self.allocator.destroy(buffer);
        // Use swapRemove for O(1) removal instead of orderedRemove O(n)
        _ = self.buffers.swapRemove(idx);
        return true;
    }

    /// Find a buffer in the pool by pointer (O(1) lookup).
    /// Returns the buffer if found, null otherwise.
    pub fn findBuffer(self: *const GpuMemoryPool, buffer: *const GpuBuffer) ?*GpuBuffer {
        const buffer_addr = @intFromPtr(buffer);
        const idx = self.buffer_lookup.get(buffer_addr) orelse return null;
        return self.buffers.items[idx];
    }

    /// Check if a buffer exists in the pool (O(1) lookup).
    pub fn contains(self: *const GpuMemoryPool, buffer: *const GpuBuffer) bool {
        const buffer_addr = @intFromPtr(buffer);
        return self.buffer_lookup.contains(buffer_addr);
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

test "memory pool O(1) buffer lookup" {
    var pool = GpuMemoryPool.init(std.testing.allocator, 0); // No size limit
    defer pool.deinit();

    // Allocate 1000 buffers
    const num_buffers = 1000;
    var handles: [num_buffers]*GpuBuffer = undefined;
    for (&handles) |*h| {
        h.* = try pool.allocate(64, .{});
    }

    // Verify all buffers can be found via O(1) lookup
    for (handles) |h| {
        try std.testing.expect(pool.contains(h));
        try std.testing.expect(pool.findBuffer(h) != null);
    }

    // Measure lookup time (should be O(1) per lookup)
    var timer = try time.Timer.start();
    for (handles) |h| {
        _ = pool.contains(h);
    }
    const elapsed = timer.read();

    // 1000 lookups should be < 1ms (O(1) each)
    // In practice, hash map lookups are ~100ns each, so 1000 should be ~100us
    try std.testing.expect(elapsed < 1_000_000); // 1ms in nanoseconds

    // Free all buffers and verify they're no longer found
    for (handles) |h| {
        try std.testing.expect(pool.free(h));
    }

    // Verify buffers are removed from lookup
    for (handles) |h| {
        try std.testing.expect(!pool.contains(h));
    }
}

test "memory pool swap remove updates lookup correctly" {
    var pool = GpuMemoryPool.init(std.testing.allocator, 0);
    defer pool.deinit();

    // Allocate 5 buffers: [0, 1, 2, 3, 4]
    const a = try pool.allocate(64, .{});
    const b = try pool.allocate(64, .{});
    const c = try pool.allocate(64, .{});
    const d = try pool.allocate(64, .{});
    const e = try pool.allocate(64, .{});

    try std.testing.expectEqual(@as(usize, 5), pool.buffers.items.len);

    // Free buffer at index 1 (b) - this should swap 'e' into position 1
    try std.testing.expect(pool.free(b));
    try std.testing.expectEqual(@as(usize, 4), pool.buffers.items.len);

    // All remaining buffers should still be findable
    try std.testing.expect(pool.contains(a));
    try std.testing.expect(!pool.contains(b)); // b was freed
    try std.testing.expect(pool.contains(c));
    try std.testing.expect(pool.contains(d));
    try std.testing.expect(pool.contains(e)); // e was swapped but should still be found

    // Free buffer at index 0 (a) - this should swap current last into position 0
    try std.testing.expect(pool.free(a));
    try std.testing.expectEqual(@as(usize, 3), pool.buffers.items.len);

    // Remaining buffers should still be findable
    try std.testing.expect(pool.contains(c));
    try std.testing.expect(pool.contains(d));
    try std.testing.expect(pool.contains(e));

    // Clean up
    try std.testing.expect(pool.free(c));
    try std.testing.expect(pool.free(d));
    try std.testing.expect(pool.free(e));

    try std.testing.expectEqual(@as(usize, 0), pool.buffers.items.len);
}
