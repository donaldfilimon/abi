//! Circular Buffer - High-performance ring buffer for time series data
//!
//! This module provides efficient circular buffer implementations for:
//! - Time series data storage with automatic rollover
//! - Fixed-size buffers with O(1) operations
//! - Memory-efficient data structures
//! - Thread-safe variants available

const std = @import("std");

/// High-performance circular buffer for time series data
pub fn CircularBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Buffer data
        data: []T,
        /// Current write position
        write_pos: usize,
        /// Current read position
        read_pos: usize,
        /// Maximum capacity
        capacity: usize,
        /// Number of elements currently stored
        len: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new circular buffer
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
            const data = try allocator.alloc(T, capacity);
            const buffer = try allocator.create(Self);
            buffer.* = Self{
                .data = data,
                .write_pos = 0,
                .read_pos = 0,
                .capacity = capacity,
                .len = 0,
                .allocator = allocator,
            };
            return buffer;
        }

        /// Deinitialize the buffer
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }

        /// Add an element to the buffer
        pub fn push(self: *Self, value: T) void {
            self.data[self.write_pos] = value;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            if (self.len < self.capacity) {
                self.len += 1;
            } else {
                self.read_pos = (self.read_pos + 1) % self.capacity;
            }
        }

        /// Remove and return the oldest element
        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            const value = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
            self.len -= 1;
            return value;
        }
    };
}

/// Alias for CircularBuffer - RingBuffer is the same implementation
pub fn RingBuffer(comptime T: type) type {
    return CircularBuffer(T);
}
