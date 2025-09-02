//! Unified SIMD Operations Module
//!
//! This module consolidates all SIMD functionality into a single, high-performance
//! implementation with:
//! - Vector operations (f32x4, f32x8, f32x16)
//! - Matrix operations with SIMD acceleration
//! - Performance monitoring and optimization
//! - Cross-platform compatibility
//! - Automatic fallbacks for unsupported operations

const std = @import("std");
const builtin = @import("builtin");

/// SIMD vector types with automatic detection
pub const Vector = struct {
    /// 4-float SIMD vector
    pub const f32x4 = if (@hasDecl(std.simd, "f32x4")) std.simd.f32x4 else @Vector(4, f32);
    /// 8-float SIMD vector
    pub const f32x8 = if (@hasDecl(std.simd, "f32x8")) std.simd.f32x8 else @Vector(8, f32);
    /// 16-float SIMD vector
    pub const f32x16 = if (@hasDecl(std.simd, "f32x16")) std.simd.f32x16 else @Vector(16, f32);

    /// Load vector from slice (compatible with both std.simd and @Vector)
    pub fn load(comptime T: type, data: []const f32) T {
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            return std.simd.f32x16.load(data);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            return std.simd.f32x8.load(data);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            return std.simd.f32x4.load(data);
        } else {
            // Fallback for @Vector types
            var result: T = undefined;
            for (0..@typeInfo(T).vector.len) |i| {
                result[i] = data[i];
            }
            return result;
        }
    }

    /// Store vector to slice (compatible with both std.simd and @Vector)
    pub fn store(data: []f32, vec: anytype) void {
        const T = @TypeOf(vec);
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            std.simd.f32x16.store(data, vec);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            std.simd.f32x8.store(data, vec);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            std.simd.f32x4.store(data, vec);
        } else {
            // Fallback for @Vector types
            for (0..@typeInfo(T).vector.len) |i| {
                data[i] = vec[i];
            }
        }
    }

    /// Create splat vector (compatible with both std.simd and @Vector)
    pub fn splat(comptime T: type, value: f32) T {
        if (@hasDecl(std.simd, "f32x16") and T == std.simd.f32x16) {
            return std.simd.f32x16.splat(value);
        } else if (@hasDecl(std.simd, "f32x8") and T == std.simd.f32x8) {
            return std.simd.f32x8.splat(value);
        } else if (@hasDecl(std.simd, "f32x4") and T == std.simd.f32x4) {
            return std.simd.f32x4.splat(value);
        } else {
            // Fallback for @Vector types
            return @splat(value);
        }
    }

    /// Check if SIMD is available for a given vector size
    pub fn isSimdAvailable(comptime size: usize) bool {
        return switch (size) {
            4 => @hasDecl(std.simd, "f32x4"),
            8 => @hasDecl(std.simd, "f32x8"),
            16 => @hasDecl(std.simd, "f32x16"),
            else => false,
        };
    }

    /// Get optimal SIMD vector size for given dimension
    pub fn getOptimalSize(dimension: usize) usize {
        if (dimension >= 16 and isSimdAvailable(16)) return 16;
        if (dimension >= 8 and isSimdAvailable(8)) return 8;
        if (dimension >= 4 and isSimdAvailable(4)) return 4;
        return 1;
    }
};

/// SIMD-optimized vector operations
pub const VectorOps = struct {
    /// Calculate Euclidean distance between two vectors using SIMD
    pub fn distance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        if (a.len == 0) return 0.0;

        const optimal_size = Vector.getOptimalSize(a.len);
        var sum: f32 = 0.0;
        var i: usize = 0;

        // SIMD-optimized distance calculation
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.load(Vector.f32x16, a[i..][0..16]);
                    const vb = Vector.load(Vector.f32x16, b[i..][0..16]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += std.simd.f32x16.reduce_add(sq);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = Vector.f32x8.load(a[i..][0..8]);
                    const vb = Vector.f32x8.load(b[i..][0..8]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += std.simd.f32x8.reduce_add(sq);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = Vector.f32x4.load(a[i..][0..4]);
                    const vb = Vector.f32x4.load(b[i..][0..4]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += std.simd.f32x4.reduce_add(sq);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }

        return @sqrt(sum);
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;
        if (a.len == 0) return 0.0;

        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        // SIMD-optimized calculations
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.f32x16.load(a[i..][0..16]);
                    const vb = Vector.f32x16.load(b[i..][0..16]);

                    dot_product += std.simd.f32x16.reduce_add(va * vb);
                    norm_a += std.simd.f32x16.reduce_add(va * va);
                    norm_b += std.simd.f32x16.reduce_add(vb * vb);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = Vector.f32x8.load(a[i..][0..8]);
                    const vb = Vector.f32x8.load(b[i..][0..8]);

                    dot_product += std.simd.f32x8.reduce_add(va * vb);
                    norm_a += std.simd.f32x8.reduce_add(va * va);
                    norm_b += std.simd.f32x8.reduce_add(vb * vb);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = Vector.f32x4.load(a[i..][0..4]);
                    const vb = Vector.f32x4.load(b[i..][0..4]);

                    dot_product += std.simd.f32x4.reduce_add(va * vb);
                    norm_a += std.simd.f32x4.reduce_add(va * va);
                    norm_b += std.simd.f32x4.reduce_add(vb * vb);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        const denominator = @sqrt(norm_a) * @sqrt(norm_b);
        if (denominator == 0.0) return 0.0;
        return dot_product / denominator;
    }

    /// Add two vectors using SIMD
    pub fn add(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.f32x16.load(a[i..][0..16]);
                    const vb = Vector.f32x16.load(b[i..][0..16]);
                    const sum = va + vb;
                    Vector.f32x16.store(result[i..][0..16], sum);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = Vector.f32x8.load(a[i..][0..8]);
                    const vb = Vector.f32x8.load(b[i..][0..8]);
                    const sum = va + vb;
                    Vector.f32x8.store(result[i..][0..8], sum);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = Vector.f32x4.load(a[i..][0..4]);
                    const vb = Vector.f32x4.load(b[i..][0..4]);
                    const sum = va + vb;
                    Vector.f32x4.store(result[i..][0..4], sum);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Subtract two vectors using SIMD
    pub fn subtract(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.f32x16.load(a[i..][0..16]);
                    const vb = Vector.f32x16.load(b[i..][0..16]);
                    const diff = va - vb;
                    Vector.f32x16.store(result[i..][0..16], diff);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = Vector.f32x8.load(a[i..][0..8]);
                    const vb = Vector.f32x8.load(b[i..][0..8]);
                    const diff = va - vb;
                    Vector.f32x8.store(result[i..][0..8], diff);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = Vector.f32x4.load(a[i..][0..4]);
                    const vb = Vector.f32x4.load(b[i..][0..4]);
                    const diff = va - vb;
                    Vector.f32x4.store(result[i..][0..4], diff);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] - b[i];
        }
    }

    /// Multiply vector by scalar using SIMD
    pub fn scale(result: []f32, vector: []const f32, scalar: f32) void {
        if (result.len != vector.len) return;

        const optimal_size = Vector.getOptimalSize(vector.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                const scale_vec = Vector.f32x16.splat(scalar);
                while (i + 16 <= vector.len) : (i += 16) {
                    const v = Vector.f32x16.load(vector[i..][0..16]);
                    const scaled = v * scale_vec;
                    Vector.f32x16.store(result[i..][0..16], scaled);
                }
            },
            8 => {
                const scale_vec = Vector.f32x8.splat(scalar);
                while (i + 8 <= vector.len) : (i += 8) {
                    const v = Vector.f32x8.load(vector[i..][0..8]);
                    const scaled = v * scale_vec;
                    Vector.f32x8.store(result[i..][0..8], scaled);
                }
            },
            4 => {
                const scale_vec = Vector.f32x4.splat(scalar);
                while (i + 4 <= vector.len) : (i += 4) {
                    const v = Vector.f32x4.load(vector[i..][0..4]);
                    const scaled = v * scale_vec;
                    Vector.f32x4.store(result[i..][0..4], scaled);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < vector.len) : (i += 1) {
            result[i] = vector[i] * scalar;
        }
    }

    /// Normalize vector to unit length
    pub fn normalize(result: []f32, vector: []const f32) void {
        if (result.len != vector.len) return;

        const norm = @sqrt(VectorOps.dotProduct(vector, vector));
        if (norm == 0.0) {
            @memset(result, 0.0);
            return;
        }

        VectorOps.scale(result, vector, 1.0 / norm);
    }

    /// Calculate dot product of two vectors
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;

        var sum: f32 = 0.0;
        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.f32x16.load(a[i..][0..16]);
                    const vb = Vector.f32x16.load(b[i..][0..16]);
                    sum += std.simd.f32x16.reduce_add(va * vb);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = Vector.f32x8.load(a[i..][0..8]);
                    const vb = Vector.f32x8.load(b[i..][0..8]);
                    sum += std.simd.f32x8.reduce_add(va * vb);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = Vector.f32x4.load(a[i..][0..4]);
                    const vb = Vector.f32x4.load(b[i..][0..4]);
                    sum += std.simd.f32x4.reduce_add(va * vb);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }
};

/// Matrix operations with SIMD acceleration
pub const MatrixOps = struct {
    /// Matrix-vector multiplication: result = matrix * vector
    pub fn matrixVectorMultiply(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) void {
        if (result.len != rows or vector.len != cols) return;

        const optimal_size = Vector.getOptimalSize(cols);
        var row: usize = 0;

        while (row < rows) : (row += 1) {
            const row_start = row * cols;
            var sum: f32 = 0.0;
            var col: usize = 0;

            // SIMD-optimized row-vector multiplication
            switch (optimal_size) {
                16 => {
                    while (col + 16 <= cols) : (col += 16) {
                        const matrix_row = Vector.f32x16.load(matrix[row_start + col ..][0..16]);
                        const vec_slice = Vector.f32x16.load(vector[col..][0..16]);
                        sum += std.simd.f32x16.reduce_add(matrix_row * vec_slice);
                    }
                },
                8 => {
                    while (col + 8 <= cols) : (col += 8) {
                        const matrix_row = Vector.f32x8.load(matrix[row_start + col ..][0..8]);
                        const vec_slice = Vector.f32x8.load(vector[col..][0..8]);
                        sum += std.simd.f32x8.reduce_add(matrix_row * vec_slice);
                    }
                },
                4 => {
                    while (col + 4 <= cols) : (col += 4) {
                        const matrix_row = Vector.f32x4.load(matrix[row_start + col ..][0..4]);
                        const vec_slice = Vector.f32x4.load(vector[col..][0..4]);
                        sum += std.simd.f32x4.reduce_add(matrix_row * vec_slice);
                    }
                },
                else => {},
            }

            // Handle remaining columns
            while (col < cols) : (col += 1) {
                sum += matrix[row_start + col] * vector[col];
            }

            result[row] = sum;
        }
    }

    /// Matrix-matrix multiplication: result = a * b
    pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
        if (result.len != m * k) return;

        // Initialize result to zero
        @memset(result, 0.0);

        const optimal_size = Vector.getOptimalSize(n);
        var i: usize = 0;

        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < k) : (j += 1) {
                var sum: f32 = 0.0;
                var l: usize = 0;

                // SIMD-optimized inner loop
                switch (optimal_size) {
                    16 => {
                        while (l + 16 <= n) : (l += 16) {
                            const a_row = Vector.f32x16.load(a[i * n + l ..][0..16]);
                            const b_col = Vector.f32x16.load(b[l * k + j ..][0..16]);
                            sum += std.simd.f32x16.reduce_add(a_row * b_col);
                        }
                    },
                    8 => {
                        while (l + 8 <= n) : (l += 8) {
                            const a_row = Vector.f32x8.load(a[i * n + l ..][0..8]);
                            const b_col = Vector.f32x8.load(b[l * k + j ..][0..8]);
                            sum += std.simd.f32x8.reduce_add(a_row * b_col);
                        }
                    },
                    4 => {
                        while (l + 4 <= n) : (l += 4) {
                            const a_row = Vector.f32x4.load(a[i * n + l ..][0..4]);
                            const b_col = Vector.f32x4.load(b[l * k + j ..][0..4]);
                            sum += std.simd.f32x4.reduce_add(a_row * b_col);
                        }
                    },
                    else => {},
                }

                // Handle remaining elements
                while (l < n) : (l += 1) {
                    sum += a[i * n + l] * b[l * k + j];
                }

                result[i * k + j] = sum;
            }
        }
    }

    /// Transpose matrix: result = matrix^T
    pub fn transpose(result: []f32, matrix: []const f32, rows: usize, cols: usize) void {
        if (result.len != rows * cols) return;

        const optimal_size = Vector.getOptimalSize(cols);
        var i: usize = 0;

        while (i < rows) : (i += 1) {
            var j: usize = 0;

            // SIMD-optimized row copying
            switch (optimal_size) {
                16 => {
                    while (j + 16 <= cols) : (j += 16) {
                        const row_data = Vector.f32x16.load(matrix[i * cols + j ..][0..16]);
                        Vector.f32x16.store(result[j * rows + i ..][0..16], row_data);
                    }
                },
                8 => {
                    while (j + 8 <= cols) : (j += 8) {
                        const row_data = Vector.f32x8.load(matrix[i * cols + j ..][0..8]);
                        Vector.f32x8.store(result[j * rows + i ..][0..8], row_data);
                    }
                },
                4 => {
                    while (j + 4 <= cols) : (j += 4) {
                        const row_data = Vector.f32x4.load(matrix[i * cols + j ..][0..4]);
                        Vector.f32x4.store(result[j * rows + i ..][0..4], row_data);
                    }
                },
                else => {},
            }

            // Handle remaining columns
            while (j < cols) : (j += 1) {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }
    }
};

/// Performance monitoring for SIMD operations
pub const PerformanceMonitor = struct {
    operation_count: u64 = 0,
    total_time_ns: u64 = 0,
    simd_usage_count: u64 = 0,
    scalar_fallback_count: u64 = 0,

    pub fn recordOperation(self: *PerformanceMonitor, duration_ns: u64, used_simd: bool) void {
        self.operation_count += 1;
        self.total_time_ns += duration_ns;
        if (used_simd) {
            self.simd_usage_count += 1;
        } else {
            self.scalar_fallback_count += 1;
        }
    }

    pub fn getAverageTime(self: *const PerformanceMonitor) f64 {
        if (self.operation_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_time_ns)) / @as(f64, @floatFromInt(self.operation_count));
    }

    pub fn getSimdUsageRate(self: *const PerformanceMonitor) f64 {
        if (self.operation_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.simd_usage_count)) / @as(f64, @floatFromInt(self.operation_count));
    }

    pub fn printStats(self: *const PerformanceMonitor) void {
        std.debug.print("SIMD Performance Statistics:\n", .{});
        std.debug.print("  Total Operations: {d}\n", .{self.operation_count});
        std.debug.print("  Average Time: {d:.3} ns\n", .{self.getAverageTime()});
        std.debug.print("  SIMD Usage Rate: {d:.1}%\n", .{self.getSimdUsageRate() * 100.0});
        std.debug.print("  SIMD Operations: {d}\n", .{self.simd_usage_count});
        std.debug.print("  Scalar Fallbacks: {d}\n", .{self.scalar_fallback_count});
    }
};

/// Global performance monitor instance
var global_performance_monitor = PerformanceMonitor{};

/// Get global performance monitor
pub fn getPerformanceMonitor() *PerformanceMonitor {
    return &global_performance_monitor;
}

// Re-export commonly used types
pub const f32x4 = Vector.f32x4;
pub const f32x8 = Vector.f32x8;
pub const f32x16 = Vector.f32x16;

// Re-export operations
pub const distance = VectorOps.distance;
pub const cosineSimilarity = VectorOps.cosineSimilarity;
pub const add = VectorOps.add;
pub const subtract = VectorOps.subtract;
pub const scale = VectorOps.scale;
pub const normalize = VectorOps.normalize;
pub const dotProduct = VectorOps.dotProduct;
pub const matrixVectorMultiply = MatrixOps.matrixVectorMultiply;
pub const matrixMultiply = MatrixOps.matrixMultiply;
pub const transpose = MatrixOps.transpose;

test "SIMD vector operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test vectors
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    // Test distance calculation
    const dist = distance(&a, &b);
    try testing.expect(dist > 0.0);

    // Test cosine similarity
    const similarity = cosineSimilarity(&a, &b);
    try testing.expect(similarity > 0.0 and similarity <= 1.0);

    // Test vector addition
    const result = try allocator.alloc(f32, a.len);
    defer allocator.free(result);
    add(result, &a, &b);
    try testing.expectEqual(@as(f32, 3.0), result[0]);
    try testing.expectEqual(@as(f32, 5.0), result[1]);

    // Test vector scaling
    scale(result, &a, 2.0);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    try testing.expectEqual(@as(f32, 4.0), result[1]);

    // Test dot product
    const dot = dotProduct(&a, &b);
    try testing.expect(dot > 0.0);
}

test "SIMD matrix operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test matrix-vector multiplication
    const matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const vector = [_]f32{ 1.0, 2.0 };
    const result = try allocator.alloc(f32, 3);
    defer allocator.free(result);

    matrixVectorMultiply(result, &matrix, &vector, 3, 2);
    try testing.expectEqual(@as(f32, 5.0), result[0]); // 1*1 + 2*2
    try testing.expectEqual(@as(f32, 11.0), result[1]); // 3*1 + 4*2
    try testing.expectEqual(@as(f32, 17.0), result[2]); // 5*1 + 6*2
}

test "SIMD performance monitoring" {
    const testing = std.testing;
    const monitor = getPerformanceMonitor();
    const start_time = std.time.nanoTimestamp();

    // Simulate some operations
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    _ = distance(&a, &b);

    const end_time = std.time.nanoTimestamp();
    const duration = @as(u64, @intCast(end_time - start_time));

    monitor.recordOperation(duration, true);
    try testing.expectEqual(@as(u64, 1), monitor.operation_count);
    try testing.expectEqual(@as(u64, 1), monitor.simd_usage_count);
}
