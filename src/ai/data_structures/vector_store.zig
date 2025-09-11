//! Vector Store - Efficient storage for high-dimensional vectors
//!
//! This module provides vector storage implementations for:
//! - High-dimensional vector storage
//! - Similarity search capabilities
//! - Memory-efficient storage

const std = @import("std");

/// Vector store for embedding storage and similarity search
pub fn VectorStore(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Vector data storage
        vectors: std.ArrayList([]T),
        /// Vector dimensions
        dimensions: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new vector store
        pub fn init(allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*Self {
            const store = try allocator.create(Self);
            store.* = Self{
                .vectors = try std.ArrayList([]T).initCapacity(allocator, capacity),
                .dimensions = dimensions,
                .allocator = allocator,
            };
            return store;
        }

        /// Deinitialize the vector store
        pub fn deinit(self: *Self) void {
            for (self.vectors.items) |vector| {
                self.allocator.free(vector);
            }
            self.vectors.deinit();
            self.allocator.destroy(self);
        }

        /// Add a vector to the store
        pub fn addVector(self: *Self, vector: []const T) !void {
            if (vector.len != self.dimensions) return error.InvalidDimensions;
            const copy = try self.allocator.dupe(T, vector);
            try self.vectors.append(copy);
        }

        /// Get a vector by index
        pub fn getVector(self: *Self, index: usize) ?[]T {
            if (index >= self.vectors.items.len) return null;
            return self.vectors.items[index];
        }

        /// Calculate cosine similarity between two vectors
        pub fn cosineSimilarity(self: *Self, a: []const T, b: []const T) f32 {
            _ = self;
            if (a.len != b.len) return 0.0;

            var dot_product: f32 = 0.0;
            var norm_a: f32 = 0.0;
            var norm_b: f32 = 0.0;

            for (0..a.len) |i| {
                const ai = if (T == f32) a[i] else @as(f32, @floatFromInt(a[i]));
                const bi = if (T == f32) b[i] else @as(f32, @floatFromInt(b[i]));
                dot_product += ai * bi;
                norm_a += ai * ai;
                norm_b += bi * bi;
            }

            if (norm_a == 0.0 or norm_b == 0.0) return 0.0;

            return dot_product / (@sqrt(norm_a) * @sqrt(norm_b));
        }
    };
}
