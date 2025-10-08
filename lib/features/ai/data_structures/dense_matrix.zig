//! Dense Matrix - Standard matrix implementation
//!
//! This module provides dense matrix implementations for:
//! - Standard matrix operations
//! - Memory-efficient contiguous storage
//! - SIMD-accelerated operations where applicable

const std = @import("std");

/// Dense matrix implementation with contiguous memory layout
pub const DenseMatrix = struct {
    const Self = @This();

    /// Matrix data (row-major order)
    data: []f32,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new dense matrix
    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
        const matrix = try allocator.create(Self);
        matrix.* = Self{
            .data = try allocator.alloc(f32, rows * cols),
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
        // Initialize to zeros
        @memset(matrix.data, 0.0);
        return matrix;
    }

    /// Deinitialize the matrix
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    /// Get element at position (i, j)
    pub fn get(self: *Self, i: usize, j: usize) f32 {
        if (i >= self.rows or j >= self.cols) return 0.0;
        return self.data[i * self.cols + j];
    }

    /// Set element at position (i, j)
    pub fn set(self: *Self, i: usize, j: usize, value: f32) void {
        if (i >= self.rows or j >= self.cols) return;
        self.data[i * self.cols + j] = value;
    }

    /// Get a row as a slice
    pub fn getRow(self: *Self, i: usize) ?[]f32 {
        if (i >= self.rows) return null;
        const start = i * self.cols;
        return self.data[start .. start + self.cols];
    }

    /// Get a column as a slice (creates a copy)
    pub fn getCol(self: *Self, j: usize) ![]f32 {
        if (j >= self.cols) return error.OutOfBounds;
        const col = try self.allocator.alloc(f32, self.rows);
        for (0..self.rows) |i| {
            col[i] = self.get(i, j);
        }
        return col;
    }

    /// Matrix multiplication (self * other)
    pub fn mul(self: *Self, other: *const Self) !*Self {
        if (self.cols != other.rows) return error.IncompatibleDimensions;

        const result = try Self.init(self.allocator, self.rows, other.cols);

        for (0..self.rows) |i| {
            for (0..other.cols) |j| {
                var sum: f32 = 0.0;
                for (0..self.cols) |k| {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        return result;
    }

    /// Element-wise addition
    pub fn add(self: *Self, other: *const Self) !void {
        if (self.rows != other.rows or self.cols != other.cols) return error.IncompatibleDimensions;

        for (0..self.data.len) |i| {
            self.data[i] += other.data[i];
        }
    }

    /// Scalar multiplication
    pub fn mulScalar(self: *Self, scalar: f32) void {
        for (self.data) |*val| {
            val.* *= scalar;
        }
    }
};
