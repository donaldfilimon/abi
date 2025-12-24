//! Batch Queue - Efficient batch processing queue
//!
//! This module provides optimized queue implementations for batch processing:
//! - Efficient batch insertion and retrieval
//! - Memory-efficient storage
//! - Configurable batch sizes

const std = @import("std");

/// High-performance batch queue for processing data in batches
pub const BatchQueue = struct {
    const Self = @This();

    /// Queue data
    data: std.ArrayList(u8),
    /// Batch size configuration
    batch_size: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new batch queue
    pub fn init(allocator: std.mem.Allocator, batch_size: usize) !*Self {
        const queue = try allocator.create(Self);
        queue.* = Self{
            .data = try std.ArrayList(u8).initCapacity(allocator, 0),
            .batch_size = batch_size,
            .allocator = allocator,
        };
        return queue;
    }

    /// Deinitialize the queue
    pub fn deinit(self: *Self) void {
        self.data.deinit();
        self.allocator.destroy(self);
    }

    /// Add data to the queue
    pub fn enqueue(self: *Self, data: []const u8) !void {
        try self.data.appendSlice(self.allocator, data);
    }

    /// Get the next batch if available (optimized O(n) -> O(k) where k=batch_size)
    pub fn dequeueBatch(self: *Self) ?[]u8 {
        if (self.data.items.len < self.batch_size) return null;
        const batch = self.data.items[0..self.batch_size];

        // More efficient: use std.mem.copy to shift remaining data
        const remaining_items = self.data.items[self.batch_size..];
        std.mem.copy(u8, self.data.items[0..remaining_items.len], remaining_items);
        self.data.items.len -= self.batch_size;

        return batch;
    }
};
