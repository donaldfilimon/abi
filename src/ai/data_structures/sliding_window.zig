//! Sliding Window - Moving window data structures
//!
//! This module provides sliding window implementations for:
//! - Fixed-size moving windows
//! - Efficient data updates
//! - Statistical calculations over windows

const std = @import("std");

/// Sliding window data structure
pub const SlidingWindow = struct {
    const Self = @This();

    /// Window data
    data: std.ArrayList(f32),
    /// Maximum window size
    max_size: usize,
    /// Current sum (for efficient average calculation)
    sum: f32,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new sliding window
    pub fn init(allocator: std.mem.Allocator, max_size: usize) !*Self {
        const window = try allocator.create(Self);
        window.* = Self{
            .data = try std.ArrayList(f32).initCapacity(allocator, max_size),
            .max_size = max_size,
            .sum = 0.0,
            .allocator = allocator,
        };
        return window;
    }

    /// Deinitialize the window
    pub fn deinit(self: *Self) void {
        self.data.deinit();
        self.allocator.destroy(self);
    }

    /// Add a value to the window
    pub fn add(self: *Self, value: f32) void {
        if (self.data.items.len >= self.max_size) {
            // Remove oldest value
            const oldest = self.data.items[0];
            self.sum -= oldest;
            _ = self.data.orderedRemove(0);
        }

        self.data.append(value) catch return;
        self.sum += value;
    }

    /// Get current window size
    pub fn size(self: *Self) usize {
        return self.data.items.len;
    }

    /// Check if window is full
    pub fn isFull(self: *Self) bool {
        return self.data.items.len == self.max_size;
    }

    /// Get average of current window
    pub fn average(self: *Self) f32 {
        if (self.data.items.len == 0) return 0.0;
        return self.sum / @as(f32, @floatFromInt(self.data.items.len));
    }

    /// Get minimum value in window
    pub fn min(self: *Self) f32 {
        if (self.data.items.len == 0) return 0.0;
        var min_val = self.data.items[0];
        for (self.data.items[1..]) |val| {
            min_val = @min(min_val, val);
        }
        return min_val;
    }

    /// Get maximum value in window
    pub fn max(self: *Self) f32 {
        if (self.data.items.len == 0) return 0.0;
        var max_val = self.data.items[0];
        for (self.data.items[1..]) |val| {
            max_val = @max(max_val, val);
        }
        return max_val;
    }

    /// Get standard deviation of current window
    pub fn stdDev(self: *Self) f32 {
        if (self.data.items.len < 2) return 0.0;

        const avg = self.average();
        var variance: f32 = 0.0;

        for (self.data.items) |val| {
            const diff = val - avg;
            variance += diff * diff;
        }

        variance /= @as(f32, @floatFromInt(self.data.items.len - 1));
        return @sqrt(variance);
    }

    /// Get value at specific index (0 = newest)
    pub fn get(self: *Self, index: usize) ?f32 {
        const actual_index = self.data.items.len - 1 - index;
        if (actual_index >= self.data.items.len) return null;
        return self.data.items[actual_index];
    }

    /// Clear all data from the window
    pub fn clear(self: *Self) void {
        self.data.clearAndFree();
        self.sum = 0.0;
    }

    /// Get all values as a slice
    pub fn getAll(self: *Self) []f32 {
        return self.data.items;
    }
};
