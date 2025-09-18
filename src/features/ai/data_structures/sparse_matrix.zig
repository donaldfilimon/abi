//! Sparse Matrix - Efficient storage for sparse matrices
//!
//! This module provides sparse matrix implementations for:
//! - Memory-efficient storage of sparse data
//! - Fast matrix operations on sparse matrices
//! - Various sparse matrix formats (COO, CSR, CSC)

const std = @import("std");

/// Sparse matrix implementation using COO (Coordinate) format
pub fn SparseMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Row indices
        row_indices: std.ArrayList(usize),
        /// Column indices
        col_indices: std.ArrayList(usize),
        /// Values
        values: std.ArrayList(T),
        /// Matrix dimensions
        rows: usize,
        /// Matrix dimensions
        cols: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,

        /// Initialize a new sparse matrix
        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
            const matrix = try allocator.create(Self);
            matrix.* = Self{
                .row_indices = try std.ArrayList(usize).initCapacity(allocator, 0),
                .col_indices = try std.ArrayList(usize).initCapacity(allocator, 0),
                .values = try std.ArrayList(T).initCapacity(allocator, 0),
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
            };
            return matrix;
        }

        /// Deinitialize the matrix
        pub fn deinit(self: *Self) void {
            self.row_indices.deinit();
            self.col_indices.deinit();
            self.values.deinit();
            self.allocator.destroy(self);
        }

        /// Set a value at the specified position
        pub fn set(self: *Self, row: usize, col: usize, value: T) !void {
            if (row >= self.rows or col >= self.cols) return error.OutOfBounds;

            // Check if element already exists
            for (0..self.row_indices.items.len) |i| {
                if (self.row_indices.items[i] == row and self.col_indices.items[i] == col) {
                    self.values.items[i] = value;
                    return;
                }
            }

            // Add new element
            try self.row_indices.append(row);
            try self.col_indices.append(col);
            try self.values.append(value);
        }

        /// Get a value at the specified position
        pub fn get(self: *Self, row: usize, col: usize) T {
            if (row >= self.rows or col >= self.cols) return 0;

            for (0..self.row_indices.items.len) |i| {
                if (self.row_indices.items[i] == row and self.col_indices.items[i] == col) {
                    return self.values.items[i];
                }
            }

            return 0;
        }

        /// Get the number of non-zero elements
        pub fn nnz(self: *Self) usize {
            return self.values.items.len;
        }
    };
}
