// SPDX-License-Identifier: MIT
// Copyright (c) 2024 ABI Framework
//
//! A fixed-capacity circular buffer (ring buffer) for time-series data.
//!
//! Ring buffers are ideal for dashboards and monitoring panels because they:
//! - Use fixed memory regardless of data volume
//! - Automatically evict oldest values when full
//! - Provide O(1) push and access operations
//!
//! ## Usage
//!
//! ```zig
//! // Create a buffer for 60 samples of f32 (e.g., 1 minute at 1 Hz)
//! var throughput = RingBuffer(f32, 60).init();
//!
//! // Add values as they arrive
//! throughput.push(45.2);
//! throughput.push(47.8);
//!
//! // Get latest value for display
//! const current = throughput.latest() orelse 0;
//!
//! // Get all values for sparkline rendering
//! var buf: [60]f32 = undefined;
//! const values = throughput.toSlice(&buf);
//! ```

const std = @import("std");

/// A fixed-capacity circular buffer that automatically evicts oldest values.
///
/// This is a comptime-parameterized generic - the capacity is fixed at compile
/// time, so no runtime allocations are needed.
pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    if (capacity == 0) {
        @compileError("RingBuffer capacity must be greater than 0");
    }

    return struct {
        const Self = @This();

        /// The maximum number of elements this buffer can hold.
        pub const max_capacity = capacity;

        /// Internal storage for elements.
        data: [capacity]T,
        /// Index where next element will be written.
        head: usize,
        /// Current number of elements stored (0 to capacity).
        count: usize,

        /// Initialize an empty ring buffer.
        pub fn init() Self {
            return .{
                .data = undefined,
                .head = 0,
                .count = 0,
            };
        }

        /// Add a value to the buffer, overwriting the oldest if at capacity.
        pub fn push(self: *Self, value: T) void {
            self.data[self.head] = value;
            self.head = (self.head + 1) % capacity;
            if (self.count < capacity) self.count += 1;
        }

        /// Copy all values to output buffer in FIFO order (oldest first).
        /// Returns a slice of the output buffer containing the copied values.
        pub fn toSlice(self: *const Self, buf: []T) []T {
            if (self.count == 0) return buf[0..0];

            // If buffer hasn't wrapped, data starts at 0
            // If wrapped (count == capacity), oldest is at head
            const start = if (self.count == capacity) self.head else 0;

            var i: usize = 0;
            var pos = start;
            while (i < self.count) : (i += 1) {
                buf[i] = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return buf[0..self.count];
        }

        /// Get the most recently added value, or null if empty.
        pub fn latest(self: *const Self) ?T {
            if (self.count == 0) return null;
            // Head points to next write position, so latest is at head - 1
            const idx = if (self.head == 0) capacity - 1 else self.head - 1;
            return self.data[idx];
        }

        /// Get the oldest value in the buffer, or null if empty.
        pub fn oldest(self: *const Self) ?T {
            if (self.count == 0) return null;
            const idx = if (self.count == capacity) self.head else 0;
            return self.data[idx];
        }

        /// Get a value by index where 0 is oldest and count-1 is newest.
        pub fn get(self: *const Self, index: usize) ?T {
            if (index >= self.count) return null;
            const start = if (self.count == capacity) self.head else 0;
            const pos = (start + index) % capacity;
            return self.data[pos];
        }

        /// Current number of values stored.
        pub fn len(self: *const Self) usize {
            return self.count;
        }

        /// Check if buffer is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        /// Check if buffer is at capacity.
        pub fn isFull(self: *const Self) bool {
            return self.count == capacity;
        }

        /// Remove all values.
        pub fn clear(self: *Self) void {
            self.head = 0;
            self.count = 0;
        }

        /// Calculate the arithmetic mean for numeric types.
        /// Returns 0 if buffer is empty.
        pub fn average(self: *const Self) f64 {
            if (self.count == 0) return 0;

            var total: f64 = 0;
            const start = if (self.count == capacity) self.head else 0;
            var pos = start;
            var i: usize = 0;
            while (i < self.count) : (i += 1) {
                total += switch (@typeInfo(T)) {
                    .int, .comptime_int => @as(f64, @floatFromInt(self.data[pos])),
                    .float, .comptime_float => @as(f64, @floatCast(self.data[pos])),
                    else => @compileError("average() requires numeric type"),
                };
                pos = (pos + 1) % capacity;
            }
            return total / @as(f64, @floatFromInt(self.count));
        }

        /// Find minimum value, or null if empty.
        pub fn min(self: *const Self) ?T {
            if (self.count == 0) return null;

            const start = if (self.count == capacity) self.head else 0;
            var result = self.data[start];
            var pos = (start + 1) % capacity;
            var i: usize = 1;
            while (i < self.count) : (i += 1) {
                if (self.data[pos] < result) result = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return result;
        }

        /// Find maximum value, or null if empty.
        pub fn max(self: *const Self) ?T {
            if (self.count == 0) return null;

            const start = if (self.count == capacity) self.head else 0;
            var result = self.data[start];
            var pos = (start + 1) % capacity;
            var i: usize = 1;
            while (i < self.count) : (i += 1) {
                if (self.data[pos] > result) result = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return result;
        }

        /// Calculate sum for numeric types.
        pub fn sum(self: *const Self) T {
            if (self.count == 0) return 0;

            var result: T = 0;
            const start = if (self.count == capacity) self.head else 0;
            var pos = start;
            var i: usize = 0;
            while (i < self.count) : (i += 1) {
                result += self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return result;
        }

        /// Iterator for reading values from oldest to newest.
        pub const Iterator = struct {
            buffer: *const Self,
            pos: usize,
            remaining: usize,

            pub fn next(self: *Iterator) ?T {
                if (self.remaining == 0) return null;
                const value = self.buffer.data[self.pos];
                self.pos = (self.pos + 1) % capacity;
                self.remaining -= 1;
                return value;
            }
        };

        /// Get an iterator over all values from oldest to newest.
        pub fn iterator(self: *const Self) Iterator {
            const start = if (self.count == capacity) self.head else 0;
            return .{
                .buffer = self,
                .pos = start,
                .remaining = self.count,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "RingBuffer push and retrieve maintains FIFO order" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(1);
    buf.push(2);
    buf.push(3);

    var out: [5]u32 = undefined;
    const slice = buf.toSlice(&out);

    try std.testing.expectEqual(@as(usize, 3), slice.len);
    try std.testing.expectEqual(@as(u32, 1), slice[0]);
    try std.testing.expectEqual(@as(u32, 2), slice[1]);
    try std.testing.expectEqual(@as(u32, 3), slice[2]);
}

test "RingBuffer overwrites oldest when full" {
    var buf = RingBuffer(u32, 3).init();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4); // Overwrites 1

    var out: [3]u32 = undefined;
    const slice = buf.toSlice(&out);

    try std.testing.expectEqual(@as(usize, 3), slice.len);
    try std.testing.expectEqual(@as(u32, 2), slice[0]);
    try std.testing.expectEqual(@as(u32, 3), slice[1]);
    try std.testing.expectEqual(@as(u32, 4), slice[2]);
}

test "RingBuffer latest returns most recent" {
    var buf = RingBuffer(u32, 5).init();
    try std.testing.expectEqual(@as(?u32, null), buf.latest());

    buf.push(10);
    try std.testing.expectEqual(@as(?u32, 10), buf.latest());

    buf.push(20);
    try std.testing.expectEqual(@as(?u32, 20), buf.latest());
}

test "RingBuffer oldest returns first added" {
    var buf = RingBuffer(u32, 3).init();
    try std.testing.expectEqual(@as(?u32, null), buf.oldest());

    buf.push(10);
    buf.push(20);
    buf.push(30);
    try std.testing.expectEqual(@as(?u32, 10), buf.oldest());

    buf.push(40); // Overwrites 10
    try std.testing.expectEqual(@as(?u32, 20), buf.oldest());
}

test "RingBuffer get by index" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(100);
    buf.push(200);
    buf.push(300);

    try std.testing.expectEqual(@as(?u32, 100), buf.get(0)); // oldest
    try std.testing.expectEqual(@as(?u32, 200), buf.get(1));
    try std.testing.expectEqual(@as(?u32, 300), buf.get(2)); // newest
    try std.testing.expectEqual(@as(?u32, null), buf.get(3)); // out of range
}

test "RingBuffer average calculation" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(10);
    buf.push(20);
    buf.push(30);

    try std.testing.expectEqual(@as(f64, 20.0), buf.average());
}

test "RingBuffer min and max" {
    var buf = RingBuffer(i32, 5).init();
    buf.push(5);
    buf.push(-3);
    buf.push(10);
    buf.push(2);

    try std.testing.expectEqual(@as(?i32, -3), buf.min());
    try std.testing.expectEqual(@as(?i32, 10), buf.max());
}

test "RingBuffer with floats" {
    var buf = RingBuffer(f32, 4).init();
    buf.push(1.5);
    buf.push(2.5);
    buf.push(3.5);
    buf.push(4.5);

    try std.testing.expectApproxEqAbs(@as(f64, 3.0), buf.average(), 0.001);
}

test "RingBuffer iterator" {
    var buf = RingBuffer(u32, 3).init();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4); // Wraps, now contains [2, 3, 4]

    var it = buf.iterator();
    try std.testing.expectEqual(@as(?u32, 2), it.next());
    try std.testing.expectEqual(@as(?u32, 3), it.next());
    try std.testing.expectEqual(@as(?u32, 4), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "RingBuffer clear" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(1);
    buf.push(2);
    try std.testing.expectEqual(@as(usize, 2), buf.len());

    buf.clear();
    try std.testing.expectEqual(@as(usize, 0), buf.len());
    try std.testing.expect(buf.isEmpty());
    try std.testing.expectEqual(@as(?u32, null), buf.latest());
}

test "RingBuffer isFull" {
    var buf = RingBuffer(u32, 3).init();
    try std.testing.expect(!buf.isFull());

    buf.push(1);
    buf.push(2);
    try std.testing.expect(!buf.isFull());

    buf.push(3);
    try std.testing.expect(buf.isFull());
}

test "RingBuffer sum" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(10);
    buf.push(20);
    buf.push(30);

    try std.testing.expectEqual(@as(u32, 60), buf.sum());
}

test {
    std.testing.refAllDecls(@This());
}
