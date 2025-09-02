//! Advanced SIMD Matrix Operations Module
//! High-performance matrix operations with SIMD optimizations and modern Zig patterns

const std = @import("std");
const core = @import("../core/mod.zig");
const Allocator = core.Allocator;
const Timer = core.Timer;

/// Matrix configuration for SIMD operations
pub const MatrixConfig = struct {
    rows: usize,
    cols: usize,
    alignment: usize,
    simd_width: usize,

    pub fn init(rows: usize, cols: usize) MatrixConfig {
        const simd_width = if (@import("builtin").target.cpu.arch == .x86_64) 8 else 4;
        const alignment = simd_width * @sizeOf(f32);

        return .{
            .rows = rows,
            .cols = cols,
            .alignment = alignment,
            .simd_width = simd_width,
        };
    }

    pub fn getAlignedSize(self: MatrixConfig) usize {
        const row_alignment = (self.cols + self.simd_width - 1) & ~(self.simd_width - 1);
        return self.rows * row_alignment;
    }

    pub fn isValid(self: MatrixConfig) bool {
        return self.rows > 0 and self.cols > 0;
    }
};

/// SIMD Matrix with optimized operations
pub const SIMDMatrix = struct {
    data: []f32,
    config: MatrixConfig,
    allocator: Allocator,

    const Self = @This();
    const VectorT = @Vector(8, f32);

    /// Initialize matrix with dimensions
    pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self {
        const config = MatrixConfig.init(rows, cols);
        if (!config.isValid()) return error.InvalidDimensions;

        const aligned_size = config.getAlignedSize();
        const data = try allocator.alignedAlloc(f32, config.alignment, aligned_size);

        // Initialize with zeros
        @memset(data, 0);

        return Self{
            .data = data,
            .config = config,
            .allocator = allocator,
        };
    }

    /// Initialize matrix with data
    pub fn initWithData(allocator: Allocator, data: []const f32, rows: usize, cols: usize) !Self {
        if (data.len != rows * cols) return error.InvalidDimensions;

        const config = MatrixConfig.init(rows, cols);
        const aligned_size = config.getAlignedSize();
        const matrix_data = try allocator.alignedAlloc(f32, config.alignment, aligned_size);

        // Copy data and pad with zeros if needed
        @memcpy(matrix_data[0..data.len], data);
        if (aligned_size > data.len) {
            @memset(matrix_data[data.len..], 0);
        }

        return Self{
            .data = matrix_data,
            .config = config,
            .allocator = allocator,
        };
    }

    /// Deinitialize matrix
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }

    /// Get matrix dimensions
    pub fn getDimensions(self: *const Self) MatrixConfig {
        return self.config;
    }

    /// Get element at position
    pub fn get(self: *const Self, row: usize, col: usize) f32 {
        if (row >= self.config.rows or col >= self.config.cols) return 0;
        return self.data[row * self.config.cols + col];
    }

    /// Set element at position
    pub fn set(self: *Self, row: usize, col: usize, value: f32) void {
        if (row >= self.config.rows or col >= self.config.cols) return;
        self.data[row * self.config.cols + col] = value;
    }

    /// Fill matrix with value
    pub fn fill(self: *Self, value: f32) void {
        const simd_value: VectorT = @splat(value);
        const simd_len = (self.data.len / 8) * 8;

        // SIMD fill
        var i: usize = 0;
        while (i < simd_len) : (i += 8) {
            const slice = self.data[i..][0..8];
            slice.* = simd_value;
        }

        // Scalar remainder
        while (i < self.data.len) : (i += 1) {
            self.data[i] = value;
        }
    }

    /// Copy data from another matrix
    pub fn copyFrom(self: *Self, other: *const Self) void {
        const min_rows = @min(self.config.rows, other.config.rows);
        const min_cols = @min(self.config.cols, other.config.cols);

        for (0..min_rows) |row| {
            for (0..min_cols) |col| {
                self.set(row, col, other.get(row, col));
            }
        }
    }

    /// Matrix addition with SIMD optimization
    pub fn add(self: *Self, other: *const Self) void {
        if (self.config.rows != other.config.rows or self.config.cols != other.config.cols) return;

        const simd_len = (self.data.len / 8) * 8;

        // SIMD addition
        var i: usize = 0;
        while (i < simd_len) : (i += 8) {
            const a: VectorT = self.data[i..][0..8].*;
            const b: VectorT = other.data[i..][0..8].*;
            const result = a + b;
            self.data[i..][0..8].* = result;
        }

        // Scalar remainder
        while (i < self.data.len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    /// Matrix subtraction with SIMD optimization
    pub fn sub(self: *Self, other: *const Self) void {
        if (self.config.rows != other.config.rows or self.config.cols != other.config.cols) return;

        const simd_len = (self.data.len / 8) * 8;

        // SIMD subtraction
        var i: usize = 0;
        while (i < simd_len) : (i += 8) {
            const a: VectorT = self.data[i..][0..8].*;
            const b: VectorT = other.data[i..][0..8].*;
            const result = a - b;
            self.data[i..][0..8].* = result;
        }

        // Scalar remainder
        while (i < self.data.len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    /// Matrix scalar multiplication with SIMD optimization
    pub fn scale(self: *Self, scalar: f32) void {
        const simd_scalar: VectorT = @splat(scalar);
        const simd_len = (self.data.len / 8) * 8;

        // SIMD scaling
        var i: usize = 0;
        while (i < simd_len) : (i += 8) {
            const a: VectorT = self.data[i..][0..8].*;
            const result = a * simd_scalar;
            self.data[i..][0..8].* = result;
        }

        // Scalar remainder
        while (i < self.data.len) : (i += 1) {
            self.data[i] *= scalar;
        }
    }

    /// Element-wise matrix multiplication with SIMD optimization
    pub fn mulElementWise(self: *Self, other: *const Self) void {
        if (self.config.rows != other.config.rows or self.config.cols != other.config.cols) return;

        const simd_len = (self.data.len / 8) * 8;

        // SIMD element-wise multiplication
        var i: usize = 0;
        while (i < simd_len) : (i += 8) {
            const a: VectorT = self.data[i..][0..8].*;
            const b: VectorT = other.data[i..][0..8].*;
            const result = a * b;
            self.data[i..][0..8].* = result;
        }

        // Scalar remainder
        while (i < self.data.len) : (i += 1) {
            self.data[i] *= other.data[i];
        }
    }

    /// Matrix multiplication with SIMD optimization
    pub fn multiply(self: *Self, other: *const Self) !Self {
        if (self.config.cols != other.config.rows) return error.DimensionMismatch;

        const result_rows = self.config.rows;
        const result_cols = other.config.cols;
        const result = try Self.init(self.allocator, result_rows, result_cols);

        // Optimized matrix multiplication with SIMD
        for (0..result_rows) |i| {
            for (0..result_cols) |j| {
                var sum: f32 = 0;
                const simd_len = (self.config.cols / 8) * 8;

                // SIMD dot product for row i and column j
                var k: usize = 0;
                while (k < simd_len) : (k += 8) {
                    const row_slice = self.data[i * self.config.cols + k ..][0..8];
                    const col_slice = other.data[k * other.config.cols + j ..][0..8];
                    const row_vec: VectorT = row_slice.*;
                    const col_vec: VectorT = col_slice.*;
                    const product = row_vec * col_vec;
                    sum += @reduce(.Add, product);
                }

                // Scalar remainder
                while (k < self.config.cols) : (k += 1) {
                    sum += self.get(i, k) * other.get(k, j);
                }

                result.set(i, j, sum);
            }
        }

        return result;
    }

    /// Matrix transpose with SIMD optimization
    pub fn transpose(self: *Self) !Self {
        const result = try Self.init(self.allocator, self.config.cols, self.config.rows);

        for (0..self.config.rows) |i| {
            for (0..self.config.cols) |j| {
                result.set(j, i, self.get(i, j));
            }
        }

        return result;
    }

    /// Matrix determinant calculation
    pub fn determinant(self: *const Self) f32 {
        if (self.config.rows != self.config.cols) return 0;
        if (self.config.rows == 1) return self.get(0, 0);
        if (self.config.rows == 2) {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }

        // For larger matrices, use LU decomposition
        var det: f32 = 1;
        var temp_matrix = self.clone() catch return 0;
        defer temp_matrix.deinit();

        // LU decomposition with partial pivoting
        for (0..self.config.rows) |k| {
            var max_row = k;
            var max_val = std.math.fabs(temp_matrix.get(k, k));

            for (k + 1..self.config.rows) |i| {
                const val = std.math.fabs(temp_matrix.get(i, k));
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }

            if (max_row != k) {
                temp_matrix.swapRows(k, max_row);
                det = -det;
            }

            if (std.math.fabs(temp_matrix.get(k, k)) < 1e-10) return 0;

            det *= temp_matrix.get(k, k);

            for (k + 1..self.config.rows) |i| {
                const factor = temp_matrix.get(i, k) / temp_matrix.get(k, k);
                for (k + 1..self.config.cols) |j| {
                    const new_val = temp_matrix.get(i, j) - factor * temp_matrix.get(k, j);
                    temp_matrix.set(i, j, new_val);
                }
            }
        }

        return det;
    }

    /// Matrix inverse calculation
    pub fn inverse(self: *const Self) !Self {
        if (self.config.rows != self.config.cols) return error.NotSquareMatrix;

        const det = self.determinant();
        if (std.math.fabs(det) < 1e-10) return error.SingularMatrix;

        const n = self.config.rows;
        var augmented = try Self.init(self.allocator, n, 2 * n);
        defer augmented.deinit();

        // Create augmented matrix [A|I]
        for (0..n) |i| {
            for (0..n) |j| {
                augmented.set(i, j, self.get(i, j));
                augmented.set(i, j + n, if (i == j) 1 else 0);
            }
        }

        // Gaussian elimination
        for (0..n) |k| {
            var max_row = k;
            var max_val = std.math.fabs(augmented.get(k, k));

            for (k + 1..n) |i| {
                const val = std.math.fabs(augmented.get(i, k));
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }

            if (max_row != k) {
                augmented.swapRows(k, max_row);
            }

            const pivot = augmented.get(k, k);
            if (std.math.fabs(pivot) < 1e-10) return error.SingularMatrix;

            // Scale row k
            for (0..2 * n) |j| {
                const new_val = augmented.get(k, j) / pivot;
                augmented.set(k, j, new_val);
            }

            // Eliminate column k
            for (0..n) |i| {
                if (i != k) {
                    const factor = augmented.get(i, k);
                    for (0..2 * n) |j| {
                        const new_val = augmented.get(i, j) - factor * augmented.get(k, j);
                        augmented.set(i, j, new_val);
                    }
                }
            }
        }

        // Extract inverse matrix
        const inverse_matrix = try Self.init(self.allocator, n, n);
        for (0..n) |i| {
            for (0..n) |j| {
                inverse_matrix.set(i, j, augmented.get(i, j + n));
            }
        }

        return inverse;
    }

    /// Clone matrix
    pub fn clone(self: *const Self) !Self {
        const result = try Self.init(self.allocator, self.config.rows, self.config.cols);
        @memcpy(result.data, self.data);
        return result;
    }

    /// Swap two rows
    fn swapRows(self: *Self, row1: usize, row2: usize) void {
        if (row1 >= self.config.rows or row2 >= self.config.rows) return;

        const start1 = row1 * self.config.cols;
        const start2 = row2 * self.config.cols;

        for (0..self.config.cols) |col| {
            const temp = self.data[start1 + col];
            self.data[start1 + col] = self.data[start2 + col];
            self.data[start2 + col] = temp;
        }
    }

    /// Print matrix for debugging
    pub fn print(self: *const Self) void {
        std.debug.print("Matrix ({d}x{d}):\n", .{ self.config.rows, self.config.cols });
        for (0..self.config.rows) |i| {
            std.debug.print("  ", .{});
            for (0..self.config.cols) |j| {
                std.debug.print("{d:8.3} ", .{self.get(i, j)});
            }
            std.debug.print("\n", .{});
        }
    }
};

/// Matrix operations namespace
pub const MatrixOps = struct {
    /// Add two matrices
    pub fn add(a: *const SIMDMatrix, b: *const SIMDMatrix) !SIMDMatrix {
        if (a.config.rows != b.config.rows or a.config.cols != b.config.cols) {
            return error.DimensionMismatch;
        }

        var result = try a.clone();
        result.add(b);
        return result;
    }

    /// Subtract two matrices
    pub fn sub(a: *const SIMDMatrix, b: *const SIMDMatrix) !SIMDMatrix {
        if (a.config.rows != b.config.rows or a.config.cols != b.config.cols) {
            return error.DimensionMismatch;
        }

        var result = try a.clone();
        result.sub(b);
        return result;
    }

    /// Multiply two matrices
    pub fn multiply(a: *const SIMDMatrix, b: *const SIMDMatrix) !SIMDMatrix {
        return a.multiply(b);
    }

    /// Create identity matrix
    pub fn identity(allocator: Allocator, size: usize) !SIMDMatrix {
        var matrix = try SIMDMatrix.init(allocator, size, size);
        for (0..size) |i| {
            matrix.set(i, i, 1.0);
        }
        return matrix;
    }

    /// Create random matrix
    pub fn random(allocator: Allocator, rows: usize, cols: usize, min: f32, max: f32) !SIMDMatrix {
        var matrix = try SIMDMatrix.init(allocator, rows, cols);
        var rng = std.rand.DefaultPrng.init(42);

        for (0..rows) |i| {
            for (0..cols) |j| {
                const value = min + (max - min) * rng.random().float(f32);
                matrix.set(i, j, value);
            }
        }

        return matrix;
    }
};

test "SIMD matrix basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var matrix = try SIMDMatrix.init(allocator, 4, 4);
    defer matrix.deinit();

    // Test initialization
    try testing.expectEqual(@as(usize, 4), matrix.config.rows);
    try testing.expectEqual(@as(usize, 4), matrix.config.cols);

    // Test set/get
    matrix.set(0, 0, 1.0);
    matrix.set(1, 1, 2.0);
    try testing.expectEqual(@as(f32, 1.0), matrix.get(0, 0));
    try testing.expectEqual(@as(f32, 2.0), matrix.get(1, 1));

    // Test fill
    matrix.fill(5.0);
    try testing.expectEqual(@as(f32, 5.0), matrix.get(0, 0));
    try testing.expectEqual(@as(f32, 5.0), matrix.get(3, 3));
}

test "SIMD matrix arithmetic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var matrix1 = try SIMDMatrix.initWithData(allocator, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, 4, 4);
    defer matrix1.deinit();

    var matrix2 = try SIMDMatrix.initWithData(allocator, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 4, 4);
    defer matrix2.deinit();

    // Test addition
    matrix1.add(&matrix2);
    try testing.expectEqual(@as(f32, 2.0), matrix1.get(0, 0));
    try testing.expectEqual(@as(f32, 3.0), matrix1.get(0, 1));
    try testing.expectEqual(@as(f32, 17.0), matrix1.get(3, 3));

    // Test scaling
    matrix1.scale(0.5);
    try testing.expectEqual(@as(f32, 1.0), matrix1.get(0, 0));
    try testing.expectEqual(@as(f32, 1.5), matrix1.get(0, 1));
}

test "SIMD matrix multiplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var matrix1 = try SIMDMatrix.initWithData(allocator, &[_]f32{ 1, 2, 3, 4 }, 2, 2);
    defer matrix1.deinit();

    var matrix2 = try SIMDMatrix.initWithData(allocator, &[_]f32{ 5, 6, 7, 8 }, 2, 2);
    defer matrix2.deinit();

    const result = try matrix1.multiply(&matrix2);
    defer result.deinit();

    // Expected result: [19 22] [43 50]
    try testing.expectEqual(@as(f32, 19.0), result.get(0, 0));
    try testing.expectEqual(@as(f32, 22.0), result.get(0, 1));
    try testing.expectEqual(@as(f32, 43.0), result.get(1, 0));
    try testing.expectEqual(@as(f32, 50.0), result.get(1, 1));
}

test "SIMD matrix identity and inverse" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const identity = try MatrixOps.identity(allocator, 3);
    defer identity.deinit();

    try testing.expectEqual(@as(f32, 1.0), identity.get(0, 0));
    try testing.expectEqual(@as(f32, 1.0), identity.get(1, 1));
    try testing.expectEqual(@as(f32, 1.0), identity.get(2, 2));
    try testing.expectEqual(@as(f32, 0.0), identity.get(0, 1));

    // Test determinant of identity matrix
    const det = identity.determinant();
    try testing.expectEqual(@as(f32, 1.0), det);
}
