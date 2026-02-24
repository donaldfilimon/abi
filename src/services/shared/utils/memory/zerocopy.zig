//! Zero-copy optimizations for data transfer.
const std = @import("std");

pub const ZeroCopyBuffer = struct {
    ptr: [*]u8,
    len: usize,
    capacity: usize,
    ref_count: *std.atomic.Value(u32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !ZeroCopyBuffer {
        const ref_count = try allocator.create(std.atomic.Value(u32));
        ref_count.* = std.atomic.Value(u32).init(1);

        const ptr = try allocator.alloc(u8, capacity);
        errdefer {
            allocator.free(ptr);
            allocator.destroy(ref_count);
        }

        return .{
            .ptr = ptr.ptr,
            .len = 0,
            .capacity = capacity,
            .ref_count = ref_count,
            .allocator = allocator,
        };
    }

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !ZeroCopyBuffer {
        const ref_count = try allocator.create(std.atomic.Value(u32));
        ref_count.* = std.atomic.Value(u32).init(1);

        const ptr = try allocator.dupe(u8, bytes);
        errdefer {
            allocator.free(ptr);
            allocator.destroy(ref_count);
        }

        return .{
            .ptr = ptr.ptr,
            .len = ptr.len,
            .capacity = ptr.len,
            .ref_count = ref_count,
            .allocator = allocator,
        };
    }

    pub fn clone(self: ZeroCopyBuffer) !ZeroCopyBuffer {
        const count = self.ref_count.fetchAdd(1, .monotonic) + 1;
        _ = count;
        return .{
            .ptr = self.ptr,
            .len = self.len,
            .capacity = self.capacity,
            .ref_count = self.ref_count,
            .allocator = self.allocator,
        };
    }

    pub fn slice(self: *const ZeroCopyBuffer, start: usize, end: usize) ![]const u8 {
        if (start > self.len or end > self.len or start > end) return error.InvalidSlice;
        return self.ptr[start..end];
    }

    pub fn asSlice(self: *const ZeroCopyBuffer) []const u8 {
        return self.ptr[0..self.len];
    }

    pub fn asMutSlice(self: *ZeroCopyBuffer) []u8 {
        return self.ptr[0..self.len];
    }

    pub fn append(self: *ZeroCopyBuffer, bytes: []const u8) !usize {
        const new_len = self.len + bytes.len;
        if (new_len > self.capacity) return error.BufferOverflow;

        @memcpy(self.ptr[self.len .. self.len + bytes.len], bytes);
        self.len = new_len;
        return bytes.len;
    }

    pub fn write(self: *ZeroCopyBuffer, offset: usize, bytes: []const u8) !void {
        if (offset + bytes.len > self.capacity) return error.OutOfCapacity;
        @memcpy(self.ptr[offset .. offset + bytes.len], bytes);
        if (offset + bytes.len > self.len) {
            self.len = offset + bytes.len;
        }
    }

    pub fn setLen(self: *ZeroCopyBuffer, len: usize) !void {
        if (len > self.capacity) return error.CapacityExceeded;
        self.len = len;
    }

    pub fn clear(self: *ZeroCopyBuffer) void {
        self.len = 0;
    }

    pub fn retain(self: *ZeroCopyBuffer) void {
        _ = self.ref_count.fetchAdd(1, .monotonic);
    }

    pub fn release(self: *ZeroCopyBuffer) void {
        const count = self.ref_count.fetchSub(1, .monotonic) - 1;
        if (count == 0) {
            self.allocator.destroy(self.ref_count);
            self.allocator.free(self.ptr[0..self.capacity]);
            self.* = undefined;
        }
    }

    pub fn refCount(self: *const ZeroCopyBuffer) u32 {
        return self.ref_count.load(.monotonic);
    }
};

pub const SharedBuffer = struct {
    buffer: ZeroCopyBuffer,
    readers: std.ArrayListUnmanaged(*Reader),
    writers: std.ArrayListUnmanaged(*Writer),

    const Reader = struct {
        buffer: *ZeroCopyBuffer,
        offset: usize,
    };

    const Writer = struct {
        buffer: *ZeroCopyBuffer,
        offset: usize,
    };

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !SharedBuffer {
        const buffer = try ZeroCopyBuffer.init(allocator, capacity);
        return .{
            .buffer = buffer,
            .readers = std.ArrayListUnmanaged(*Reader).empty,
            .writers = std.ArrayListUnmanaged(*Writer).empty,
        };
    }

    pub fn deinit(self: *SharedBuffer) void {
        for (self.readers.items) |reader| {
            self.buffer.allocator.destroy(reader);
        }
        self.readers.deinit(self.buffer.allocator);

        for (self.writers.items) |writer| {
            self.buffer.allocator.destroy(writer);
        }
        self.writers.deinit(self.buffer.allocator);

        self.buffer.release();
        self.* = undefined;
    }

    pub fn readAt(self: *SharedBuffer, offset: usize, out: []u8, timeout_ms: u64) !usize {
        _ = timeout_ms;
        if (offset + out.len > self.buffer.len) return error.OutOfBounds;

        @memcpy(out, self.buffer.ptr[offset .. offset + out.len]);
        return out.len;
    }

    pub fn writeAt(self: *SharedBuffer, offset: usize, bytes: []const u8, timeout_ms: u64) !usize {
        _ = timeout_ms;
        if (offset + bytes.len > self.buffer.capacity) return error.OutOfCapacity;

        @memcpy(self.buffer.ptr[offset .. offset + bytes.len], bytes);
        if (offset + bytes.len > self.buffer.len) {
            self.buffer.len = offset + bytes.len;
        }
        return bytes.len;
    }

    pub fn append(self: *SharedBuffer, bytes: []const u8) !usize {
        return self.buffer.append(bytes);
    }

    pub fn getStats(self: *const SharedBuffer) SharedBufferStats {
        return .{
            .total_readers = self.readers.items.len,
            .total_writers = self.writers.items.len,
            .buffer_length = self.buffer.len,
            .buffer_capacity = self.buffer.capacity,
            .ref_count = self.buffer.refCount(),
        };
    }
};

pub const SharedBufferStats = struct {
    total_readers: usize,
    total_writers: usize,
    buffer_length: usize,
    buffer_capacity: usize,
    ref_count: u32,
};

pub fn zeroCopySlice(allocator: std.mem.Allocator, src: []const u8) !ZeroCopyBuffer {
    return try ZeroCopyBuffer.fromBytes(allocator, src);
}

pub fn zeroCopyConcat(allocator: std.mem.Allocator, slices: []const []const u8) ![]u8 {
    var total: usize = 0;
    for (slices) |slice| {
        total += slice.len;
    }

    const result = try allocator.alloc(u8, total);
    var offset: usize = 0;
    for (slices) |slice| {
        @memcpy(result[offset .. offset + slice.len], slice);
        offset += slice.len;
    }
    return result;
}

pub fn zeroCopyTransfer(allocator: std.mem.Allocator, src: []const u8) !ZeroCopyBuffer {
    const buffer = try ZeroCopyBuffer.fromBytes(allocator, src);
    return buffer;
}

test "zero copy buffer init" {
    const allocator = std.testing.allocator;
    var buffer = try ZeroCopyBuffer.init(allocator, 1024);
    defer buffer.release();

    try std.testing.expectEqual(@as(usize, 0), buffer.len);
    try std.testing.expectEqual(@as(usize, 1024), buffer.capacity);
    try std.testing.expectEqual(@as(u32, 1), buffer.refCount());
}

test "zero copy buffer append" {
    const allocator = std.testing.allocator;
    var buffer = try ZeroCopyBuffer.init(allocator, 1024);
    defer buffer.release();

    const data = "hello world".*;
    try std.testing.expectEqual(@as(usize, 11), try buffer.append(&data));
    try std.testing.expectEqual(@as(usize, 11), buffer.len);
    try std.testing.expectEqualStrings("hello world", buffer.asSlice());
}

test "zero copy buffer clone" {
    const allocator = std.testing.allocator;
    var buffer = try ZeroCopyBuffer.init(allocator, 1024);
    defer buffer.release();

    var cloned = try buffer.clone();
    // Note: don't defer release here since we explicitly release below

    try std.testing.expectEqual(buffer.ptr, cloned.ptr);
    try std.testing.expectEqual(@as(u32, 2), cloned.refCount());
    cloned.release();
    try std.testing.expectEqual(@as(u32, 1), buffer.refCount());
}

test "shared buffer" {
    const allocator = std.testing.allocator;
    var shared = try SharedBuffer.init(allocator, 1024);
    defer shared.deinit();

    const data = "test data".*;
    try std.testing.expectEqual(@as(usize, 9), try shared.append(&data));

    const stats = shared.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_readers);
    try std.testing.expectEqual(@as(usize, 0), stats.total_writers);
}

test {
    std.testing.refAllDecls(@This());
}
