//! Ring buffer for sliding window attention.
//!
//! When the context length exceeds the model's capacity, a ring buffer
//! allows older tokens to be evicted while maintaining recency.

const std = @import("std");

/// Ring buffer for fixed-size sliding window.
pub fn RingBuffer(comptime T: type) type {
    return struct {
        data: []T,
        capacity: usize,
        head: usize, // Next write position
        len: usize, // Current number of elements

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            const data = try allocator.alloc(T, capacity);
            @memset(data, std.mem.zeroes(T));

            return .{
                .data = data,
                .capacity = capacity,
                .head = 0,
                .len = 0,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        /// Push a new element, evicting oldest if full.
        pub fn push(self: *Self, value: T) void {
            self.data[self.head] = value;
            self.head = (self.head + 1) % self.capacity;

            if (self.len < self.capacity) {
                self.len += 1;
            }
        }

        /// Push multiple elements.
        pub fn pushSlice(self: *Self, values: []const T) void {
            for (values) |v| {
                self.push(v);
            }
        }

        /// Get element at logical index (0 = oldest).
        pub fn get(self: *const Self, index: usize) ?T {
            if (index >= self.len) return null;

            // Calculate actual index in ring buffer
            const start = if (self.len < self.capacity)
                0
            else
                self.head;

            const actual_idx = (start + index) % self.capacity;
            return self.data[actual_idx];
        }

        /// Get the most recent element.
        pub fn latest(self: *const Self) ?T {
            if (self.len == 0) return null;
            const idx = if (self.head == 0) self.capacity - 1 else self.head - 1;
            return self.data[idx];
        }

        /// Get the oldest element.
        pub fn oldest(self: *const Self) ?T {
            return self.get(0);
        }

        /// Clear the buffer.
        pub fn clear(self: *Self) void {
            self.head = 0;
            self.len = 0;
        }

        /// Check if buffer is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        /// Check if buffer is full.
        pub fn isFull(self: *const Self) bool {
            return self.len == self.capacity;
        }

        /// Get current length.
        pub fn length(self: *const Self) usize {
            return self.len;
        }

        /// Convert to slice in order (oldest to newest).
        /// Caller owns returned memory.
        pub fn toSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const result = try allocator.alloc(T, self.len);

            for (0..self.len) |i| {
                result[i] = self.get(i).?;
            }

            return result;
        }
    };
}

/// Specialized ring buffer for KV cache with sliding window.
pub const KvRingBuffer = struct {
    k_buffer: []f32,
    v_buffer: []f32,
    capacity: u32, // Max sequence length
    kv_dim: u32,
    head: u32,
    len: u32,

    pub fn init(allocator: std.mem.Allocator, capacity: u32, kv_dim: u32) !KvRingBuffer {
        const buffer_size = @as(usize, capacity) * kv_dim;

        const k_buffer = try allocator.alloc(f32, buffer_size);
        errdefer allocator.free(k_buffer);
        const v_buffer = try allocator.alloc(f32, buffer_size);

        @memset(k_buffer, 0);
        @memset(v_buffer, 0);

        return .{
            .k_buffer = k_buffer,
            .v_buffer = v_buffer,
            .capacity = capacity,
            .kv_dim = kv_dim,
            .head = 0,
            .len = 0,
        };
    }

    pub fn deinit(self: *KvRingBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.k_buffer);
        allocator.free(self.v_buffer);
        self.* = undefined;
    }

    /// Push new K, V pair.
    pub fn push(self: *KvRingBuffer, k: []const f32, v: []const f32) void {
        const offset = @as(usize, self.head) * self.kv_dim;
        @memcpy(self.k_buffer[offset .. offset + self.kv_dim], k);
        @memcpy(self.v_buffer[offset .. offset + self.kv_dim], v);

        self.head = (self.head + 1) % self.capacity;
        if (self.len < self.capacity) {
            self.len += 1;
        }
    }

    /// Get K at logical index.
    pub fn getK(self: *const KvRingBuffer, index: u32) ?[]const f32 {
        if (index >= self.len) return null;

        const start: u32 = if (self.len < self.capacity) 0 else self.head;
        const actual_idx = (start + index) % self.capacity;
        const offset = @as(usize, actual_idx) * self.kv_dim;

        return self.k_buffer[offset .. offset + self.kv_dim];
    }

    /// Get V at logical index.
    pub fn getV(self: *const KvRingBuffer, index: u32) ?[]const f32 {
        if (index >= self.len) return null;

        const start: u32 = if (self.len < self.capacity) 0 else self.head;
        const actual_idx = (start + index) % self.capacity;
        const offset = @as(usize, actual_idx) * self.kv_dim;

        return self.v_buffer[offset .. offset + self.kv_dim];
    }

    /// Get all K values (contiguous copy).
    pub fn getAllK(self: *const KvRingBuffer, allocator: std.mem.Allocator) ![]f32 {
        const result = try allocator.alloc(f32, @as(usize, self.len) * self.kv_dim);

        for (0..self.len) |i| {
            const src = self.getK(@intCast(i)).?;
            const dst_offset = i * self.kv_dim;
            @memcpy(result[dst_offset .. dst_offset + self.kv_dim], src);
        }

        return result;
    }

    /// Get all V values (contiguous copy).
    pub fn getAllV(self: *const KvRingBuffer, allocator: std.mem.Allocator) ![]f32 {
        const result = try allocator.alloc(f32, @as(usize, self.len) * self.kv_dim);

        for (0..self.len) |i| {
            const src = self.getV(@intCast(i)).?;
            const dst_offset = i * self.kv_dim;
            @memcpy(result[dst_offset .. dst_offset + self.kv_dim], src);
        }

        return result;
    }

    pub fn length(self: *const KvRingBuffer) u32 {
        return self.len;
    }

    pub fn clear(self: *KvRingBuffer) void {
        self.head = 0;
        self.len = 0;
    }
};

test "ring buffer basic" {
    const allocator = std.testing.allocator;

    var rb = try RingBuffer(u32).init(allocator, 4);
    defer rb.deinit(allocator);

    try std.testing.expect(rb.isEmpty());

    rb.push(1);
    rb.push(2);
    rb.push(3);

    try std.testing.expectEqual(@as(usize, 3), rb.length());
    try std.testing.expectEqual(@as(?u32, 1), rb.oldest());
    try std.testing.expectEqual(@as(?u32, 3), rb.latest());
}

test "ring buffer overflow" {
    const allocator = std.testing.allocator;

    var rb = try RingBuffer(u32).init(allocator, 3);
    defer rb.deinit(allocator);

    rb.push(1);
    rb.push(2);
    rb.push(3);
    rb.push(4); // Evicts 1

    try std.testing.expectEqual(@as(usize, 3), rb.length());
    try std.testing.expect(rb.isFull());
    try std.testing.expectEqual(@as(?u32, 2), rb.oldest());
    try std.testing.expectEqual(@as(?u32, 4), rb.latest());
}

test "kv ring buffer" {
    const allocator = std.testing.allocator;

    var rb = try KvRingBuffer.init(allocator, 4, 8);
    defer rb.deinit(allocator);

    const k1 = [_]f32{1.0} ** 8;
    const v1 = [_]f32{2.0} ** 8;

    rb.push(&k1, &v1);
    try std.testing.expectEqual(@as(u32, 1), rb.length());

    const retrieved_k = rb.getK(0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), retrieved_k[0], 0.001);
}
