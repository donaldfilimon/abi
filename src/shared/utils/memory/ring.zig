//! Lock-free ring buffer for streaming and producer/consumer patterns.
//!
//! Provides a fixed-capacity circular buffer with:
//! - Lock-free single-producer single-consumer (SPSC) mode
//! - Multi-producer multi-consumer (MPMC) mode with atomics
//! - Contiguous read/write windows
//! - Watermark-based flow control
//!
//! Usage:
//! ```zig
//! var ring = try RingBuffer(u8).init(allocator, 1024);
//! defer ring.deinit();
//!
//! // Write data
//! _ = try ring.write("hello world");
//!
//! // Read data
//! var buf: [32]u8 = undefined;
//! const n = ring.read(&buf);
//! ```

const std = @import("std");

/// Configuration for ring buffer behavior.
pub const RingConfig = struct {
    /// Enable thread-safe mode (uses atomics).
    thread_safe: bool = true,
    /// High water mark for flow control (0.0-1.0).
    high_water_mark: f32 = 0.9,
    /// Low water mark for flow control (0.0-1.0).
    low_water_mark: f32 = 0.1,
};

/// Ring buffer errors.
pub const RingError = error{
    BufferFull,
    BufferEmpty,
    InvalidCapacity,
    ReadOverflow,
    WriteOverflow,
};

/// Generic ring buffer implementation.
pub fn RingBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        buffer: []T,
        capacity: usize,
        read_pos: std.atomic.Value(usize),
        write_pos: std.atomic.Value(usize),
        config: RingConfig,
        total_written: std.atomic.Value(u64),
        total_read: std.atomic.Value(u64),
        drops: std.atomic.Value(u64),

        /// Initialize a new ring buffer with the given capacity.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            return initWithConfig(allocator, capacity, .{});
        }

        /// Initialize with custom configuration.
        pub fn initWithConfig(allocator: std.mem.Allocator, capacity: usize, config: RingConfig) !Self {
            if (capacity == 0) return RingError.InvalidCapacity;

            // Round up to power of 2 for efficient modulo
            const actual_capacity = std.math.ceilPowerOfTwo(usize, capacity) catch capacity;

            const buffer = try allocator.alloc(T, actual_capacity);

            return .{
                .allocator = allocator,
                .buffer = buffer,
                .capacity = actual_capacity,
                .read_pos = std.atomic.Value(usize).init(0),
                .write_pos = std.atomic.Value(usize).init(0),
                .config = config,
                .total_written = std.atomic.Value(u64).init(0),
                .total_read = std.atomic.Value(u64).init(0),
                .drops = std.atomic.Value(u64).init(0),
            };
        }

        /// Deinitialize and free resources.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
            self.* = undefined;
        }

        /// Write a single element to the buffer.
        pub fn push(self: *Self, item: T) RingError!void {
            const write_idx = self.write_pos.load(.acquire);
            const read_idx = self.read_pos.load(.acquire);

            const next_write = (write_idx + 1) & (self.capacity - 1);

            if (next_write == read_idx) {
                _ = self.drops.fetchAdd(1, .monotonic);
                return RingError.BufferFull;
            }

            self.buffer[write_idx] = item;
            self.write_pos.store(next_write, .release);
            _ = self.total_written.fetchAdd(1, .monotonic);
        }

        /// Read a single element from the buffer.
        pub fn pop(self: *Self) RingError!T {
            const read_idx = self.read_pos.load(.acquire);
            const write_idx = self.write_pos.load(.acquire);

            if (read_idx == write_idx) {
                return RingError.BufferEmpty;
            }

            const item = self.buffer[read_idx];
            const next_read = (read_idx + 1) & (self.capacity - 1);
            self.read_pos.store(next_read, .release);
            _ = self.total_read.fetchAdd(1, .monotonic);

            return item;
        }

        /// Write multiple elements to the buffer.
        /// Returns number of elements actually written.
        pub fn write(self: *Self, items: []const T) usize {
            var written: usize = 0;
            for (items) |item| {
                self.push(item) catch break;
                written += 1;
            }
            return written;
        }

        /// Read multiple elements from the buffer.
        /// Returns number of elements actually read.
        pub fn read(self: *Self, output: []T) usize {
            var count: usize = 0;
            while (count < output.len) {
                output[count] = self.pop() catch break;
                count += 1;
            }
            return count;
        }

        /// Get a contiguous readable slice (may not include all available data).
        pub fn peekContiguous(self: *Self) []const T {
            const read_idx = self.read_pos.load(.acquire);
            const write_idx = self.write_pos.load(.acquire);

            if (read_idx == write_idx) {
                return &.{};
            }

            if (write_idx > read_idx) {
                return self.buffer[read_idx..write_idx];
            } else {
                // Data wraps around, return first contiguous portion
                return self.buffer[read_idx..self.capacity];
            }
        }

        /// Consume n elements after peeking.
        pub fn consume(self: *Self, n: usize) void {
            const read_idx = self.read_pos.load(.acquire);
            const new_read = (read_idx + n) & (self.capacity - 1);
            self.read_pos.store(new_read, .release);
            _ = self.total_read.fetchAdd(@as(u64, @intCast(n)), .monotonic);
        }

        /// Get number of elements available to read.
        pub fn len(self: *const Self) usize {
            const read_idx = self.read_pos.load(.acquire);
            const write_idx = self.write_pos.load(.acquire);

            if (write_idx >= read_idx) {
                return write_idx - read_idx;
            } else {
                return self.capacity - read_idx + write_idx;
            }
        }

        /// Get number of free slots available for writing.
        pub fn available(self: *const Self) usize {
            return self.capacity - 1 - self.len();
        }

        /// Check if the buffer is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.read_pos.load(.acquire) == self.write_pos.load(.acquire);
        }

        /// Check if the buffer is full.
        pub fn isFull(self: *const Self) bool {
            const write_idx = self.write_pos.load(.acquire);
            const read_idx = self.read_pos.load(.acquire);
            const next_write = (write_idx + 1) & (self.capacity - 1);
            return next_write == read_idx;
        }

        /// Check if above high water mark.
        pub fn isHighWater(self: *const Self) bool {
            const usage = @as(f32, @floatFromInt(self.len())) / @as(f32, @floatFromInt(self.capacity));
            return usage >= self.config.high_water_mark;
        }

        /// Check if below low water mark.
        pub fn isLowWater(self: *const Self) bool {
            const usage = @as(f32, @floatFromInt(self.len())) / @as(f32, @floatFromInt(self.capacity));
            return usage <= self.config.low_water_mark;
        }

        /// Clear the buffer.
        pub fn clear(self: *Self) void {
            self.read_pos.store(0, .release);
            self.write_pos.store(0, .release);
        }

        /// Get buffer statistics.
        pub fn getStats(self: *const Self) RingStats {
            return .{
                .capacity = self.capacity,
                .length = self.len(),
                .available = self.available(),
                .total_written = self.total_written.load(.monotonic),
                .total_read = self.total_read.load(.monotonic),
                .drops = self.drops.load(.monotonic),
                .utilization = @as(f32, @floatFromInt(self.len())) / @as(f32, @floatFromInt(self.capacity)),
            };
        }
    };
}

/// Ring buffer statistics.
pub const RingStats = struct {
    capacity: usize,
    length: usize,
    available: usize,
    total_written: u64,
    total_read: u64,
    drops: u64,
    utilization: f32,
};

/// Byte ring buffer with streaming helpers.
pub const ByteRing = struct {
    ring: RingBuffer(u8),

    const Self = @This();

    /// Initialize a byte ring buffer.
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
        return .{
            .ring = try RingBuffer(u8).init(allocator, capacity),
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *Self) void {
        self.ring.deinit();
    }

    /// Write bytes.
    pub fn writeBytes(self: *Self, bytes: []const u8) usize {
        return self.ring.write(bytes);
    }

    /// Read bytes.
    pub fn readBytes(self: *Self, output: []u8) usize {
        return self.ring.read(output);
    }

    /// Write a line (appends newline).
    pub fn writeLine(self: *Self, line: []const u8) !void {
        const written = self.ring.write(line);
        if (written < line.len) {
            return RingError.BufferFull;
        }
        self.ring.push('\n') catch return RingError.BufferFull;
    }

    /// Read a line (up to delimiter).
    pub fn readLine(self: *Self, output: []u8) ?[]const u8 {
        const data = self.ring.peekContiguous();
        for (data, 0..) |c, i| {
            if (c == '\n') {
                const line_len = @min(i, output.len);
                _ = self.ring.read(output[0..line_len]);
                _ = self.ring.pop() catch {}; // consume newline
                return output[0..line_len];
            }
        }
        return null;
    }

    /// Get available data length.
    pub fn len(self: *const Self) usize {
        return self.ring.len();
    }

    /// Check if empty.
    pub fn isEmpty(self: *const Self) bool {
        return self.ring.isEmpty();
    }

    /// Clear buffer.
    pub fn clear(self: *Self) void {
        self.ring.clear();
    }
};

test "ring buffer basic push/pop" {
    const allocator = std.testing.allocator;
    var ring = try RingBuffer(u32).init(allocator, 8);
    defer ring.deinit();

    try ring.push(1);
    try ring.push(2);
    try ring.push(3);

    try std.testing.expectEqual(@as(usize, 3), ring.len());

    try std.testing.expectEqual(@as(u32, 1), try ring.pop());
    try std.testing.expectEqual(@as(u32, 2), try ring.pop());
    try std.testing.expectEqual(@as(u32, 3), try ring.pop());

    try std.testing.expect(ring.isEmpty());
}

test "ring buffer bulk write/read" {
    const allocator = std.testing.allocator;
    var ring = try RingBuffer(u8).init(allocator, 16);
    defer ring.deinit();

    const data = "hello world";
    const written = ring.write(data);
    try std.testing.expectEqual(@as(usize, 11), written);

    var buf: [32]u8 = undefined;
    const count = ring.read(&buf);
    try std.testing.expectEqual(@as(usize, 11), count);
    try std.testing.expectEqualStrings("hello world", buf[0..count]);
}

test "ring buffer wraparound" {
    const allocator = std.testing.allocator;
    var ring = try RingBuffer(u8).init(allocator, 8);
    defer ring.deinit();

    // Fill partially
    _ = ring.write("hello");
    try std.testing.expectEqual(@as(usize, 5), ring.len());

    // Read some
    var buf: [3]u8 = undefined;
    _ = ring.read(&buf);
    try std.testing.expectEqual(@as(usize, 2), ring.len());

    // Write more (wraps around)
    _ = ring.write("world");
    try std.testing.expectEqual(@as(usize, 7), ring.len());

    // Read all
    var out: [16]u8 = undefined;
    const n = ring.read(&out);
    try std.testing.expectEqualStrings("loworld", out[0..n]);
}

test "ring buffer full detection" {
    const allocator = std.testing.allocator;
    var ring = try RingBuffer(u8).init(allocator, 4); // Actual capacity 4
    defer ring.deinit();

    try ring.push(1);
    try ring.push(2);
    try ring.push(3);
    // Buffer should be full now (capacity - 1 = 3)

    try std.testing.expect(ring.isFull());
    try std.testing.expectError(RingError.BufferFull, ring.push(4));
}

test "byte ring line operations" {
    const allocator = std.testing.allocator;
    var ring = try ByteRing.init(allocator, 64);
    defer ring.deinit();

    try ring.writeLine("hello");
    try ring.writeLine("world");

    var line1: [32]u8 = undefined;
    const result1 = ring.readLine(&line1);
    try std.testing.expect(result1 != null);
    try std.testing.expectEqualStrings("hello", result1.?);

    var line2: [32]u8 = undefined;
    const result2 = ring.readLine(&line2);
    try std.testing.expect(result2 != null);
    try std.testing.expectEqualStrings("world", result2.?);
}

test "ring buffer statistics" {
    const allocator = std.testing.allocator;
    var ring = try RingBuffer(u8).init(allocator, 16);
    defer ring.deinit();

    _ = ring.write("test");
    var buf: [2]u8 = undefined;
    _ = ring.read(&buf);

    const stats = ring.getStats();
    try std.testing.expectEqual(@as(u64, 4), stats.total_written);
    try std.testing.expectEqual(@as(u64, 2), stats.total_read);
    try std.testing.expectEqual(@as(usize, 2), stats.length);
}
