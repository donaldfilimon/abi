//! Compressed Vector - Memory-efficient vector storage
//!
//! This module provides compressed vector implementations for:
//! - Sparse vector storage
//! - Quantization-based compression
//! - Memory-efficient operations

const std = @import("std");

/// Compressed vector implementation using sparse storage
pub const CompressedVector = struct {
    const Self = @This();

    /// Non-zero indices
    indices: std.ArrayList(usize),
    /// Non-zero values
    values: std.ArrayList(f32),
    /// Original vector dimension
    dimension: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new compressed vector
    pub fn init(allocator: std.mem.Allocator, dimension: usize) !*Self {
        const vector = try allocator.create(Self);
        vector.* = Self{
            .indices = try std.ArrayList(usize).initCapacity(allocator, 0),
            .values = try std.ArrayList(f32).initCapacity(allocator, 0),
            .dimension = dimension,
            .allocator = allocator,
        };
        return vector;
    }

    /// Deinitialize the vector
    pub fn deinit(self: *Self) void {
        self.indices.deinit();
        self.values.deinit();
        self.allocator.destroy(self);
    }

    /// Set a value at the specified index
    pub fn set(self: *Self, index: usize, value: f32) !void {
        if (index >= self.dimension) return error.OutOfBounds;

        // Check if index already exists
        for (0..self.indices.items.len) |i| {
            if (self.indices.items[i] == index) {
                if (value == 0.0) {
                    // Remove zero value
                    _ = self.indices.orderedRemove(i);
                    _ = self.values.orderedRemove(i);
                } else {
                    self.values.items[i] = value;
                }
                return;
            }
        }

        // Add new non-zero value
        if (value != 0.0) {
            try self.indices.append(index);
            try self.values.append(value);
        }
    }

    /// Get a value at the specified index
    pub fn get(self: *Self, index: usize) f32 {
        if (index >= self.dimension) return 0.0;

        for (0..self.indices.items.len) |i| {
            if (self.indices.items[i] == index) {
                return self.values.items[i];
            }
        }

        return 0.0;
    }

    /// Get the number of non-zero elements
    pub fn nnz(self: *Self) usize {
        return self.values.items.len;
    }

    /// Calculate dot product with another compressed vector
    pub fn dot(self: *Self, other: *const Self) f32 {
        if (self.dimension != other.dimension) return 0.0;

        var result: f32 = 0.0;
        var i: usize = 0;
        var j: usize = 0;

        // Merge join on sorted indices
        while (i < self.indices.items.len and j < other.indices.items.len) {
            const idx_a = self.indices.items[i];
            const idx_b = other.indices.items[j];

            if (idx_a == idx_b) {
                result += self.values.items[i] * other.values.items[j];
                i += 1;
                j += 1;
            } else if (idx_a < idx_b) {
                i += 1;
            } else {
                j += 1;
            }
        }

        return result;
    }

    /// Add another compressed vector to this one
    pub fn add(self: *Self, other: *const Self) !void {
        if (self.dimension != other.dimension) return error.IncompatibleDimensions;

        // Add all values from other vector
        for (0..other.indices.items.len) |i| {
            const index = other.indices.items[i];
            const value = other.values.items[i];
            const current = self.get(index);
            try self.set(index, current + value);
        }
    }
};
