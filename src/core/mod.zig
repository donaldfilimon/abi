//! Core Module - Fundamental Types and Operations
//!
//! This module provides the foundational types, errors, and operations used across
//! the entire WDBX framework, including SIMD operations and error definitions.

const std = @import("std");

/// Framework-wide error set for consistent error handling
pub const FrameworkError = error{
    // Generic framework errors
    InvalidConfiguration,
    UnsupportedOperation,
    InvalidState,
    InvalidData,
    NotImplemented,
    ResourceExhausted,
    OperationFailed,

    // Memory-related errors
    OutOfMemory,
    InvalidAlignment,
    BufferTooSmall,

    // Data processing errors
    InvalidDimensions,
    TypeMismatch,
    ConversionFailed,

    // I/O errors
    FileNotFound,
    AccessDenied,
    NetworkError,

    // Computation errors
    NumericalInstability,
    ConvergenceFailure,
    DivisionByZero,
} || std.mem.Allocator.Error;

// SIMD functionality is now integrated directly into core

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;

// =============================================================================
// SIMD OPERATIONS
// =============================================================================

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
        var acc: f32 = 0.0;
        var i: usize = 0;

        // SIMD-optimized distance calculation
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = Vector.load(Vector.f32x16, a[i..][0..16]);
                    const vb = Vector.load(Vector.f32x16, b[i..][0..16]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    acc += @reduce(.Add, sq);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const diff = va - vb;
                    const sq = diff * diff;
                    acc += @reduce(.Add, sq);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const diff = va - vb;
                    const sq = diff * diff;
                    acc += @reduce(.Add, sq);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            acc += diff * diff;
        }

        return @sqrt(acc);
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
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);

                    dot_product += @reduce(.Add, va * vb);
                    norm_a += @reduce(.Add, va * va);
                    norm_b += @reduce(.Add, vb * vb);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);

                    dot_product += @reduce(.Add, va * vb);
                    norm_a += @reduce(.Add, va * va);
                    norm_b += @reduce(.Add, vb * vb);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);

                    dot_product += @reduce(.Add, va * vb);
                    norm_a += @reduce(.Add, va * va);
                    norm_b += @reduce(.Add, vb * vb);
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
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const s = va + vb;
                    @memcpy(result[i..][0..16], @as([16]f32, s)[0..]);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const s = va + vb;
                    @memcpy(result[i..][0..8], @as([8]f32, s)[0..]);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const s = va + vb;
                    @memcpy(result[i..][0..4], @as([4]f32, s)[0..]);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Calculate dot product of two vectors
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;

        var acc: f32 = 0.0;
        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    acc += @reduce(.Add, va * vb);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    acc += @reduce(.Add, va * vb);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    acc += @reduce(.Add, va * vb);
                }
            },
            else => {},
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            acc += a[i] * b[i];
        }

        return acc;
    }

    /// Multiply vector by scalar using SIMD
    pub fn scale(result: []f32, vector: []const f32, scalar: f32) void {
        if (result.len != vector.len) return;

        const optimal_size = Vector.getOptimalSize(vector.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                const scale_vec = @as(@Vector(16, f32), @splat(scalar));
                while (i + 16 <= vector.len) : (i += 16) {
                    const v = @as(@Vector(16, f32), vector[i..][0..16].*);
                    const scaled = v * scale_vec;
                    @memcpy(result[i..][0..16], @as([16]f32, scaled)[0..]);
                }
            },
            8 => {
                const scale_vec = @as(@Vector(8, f32), @splat(scalar));
                while (i + 8 <= vector.len) : (i += 8) {
                    const v = @as(@Vector(8, f32), vector[i..][0..8].*);
                    const scaled = v * scale_vec;
                    @memcpy(result[i..][0..8], @as([8]f32, scaled)[0..]);
                }
            },
            4 => {
                const scale_vec = @as(@Vector(4, f32), @splat(scalar));
                while (i + 4 <= vector.len) : (i += 4) {
                    const v = @as(@Vector(4, f32), vector[i..][0..4].*);
                    const scaled = v * scale_vec;
                    @memcpy(result[i..][0..4], @as([4]f32, scaled)[0..]);
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
            var acc_row: f32 = 0.0;
            var col: usize = 0;

            // SIMD-optimized row-vector multiplication
            switch (optimal_size) {
                16 => {
                    while (col + 16 <= cols) : (col += 16) {
                        const matrix_row = @as(@Vector(16, f32), matrix[row_start + col ..][0..16].*);
                        const vec_slice = @as(@Vector(16, f32), vector[col..][0..16].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                8 => {
                    while (col + 8 <= cols) : (col += 8) {
                        const matrix_row = @as(@Vector(8, f32), matrix[row_start + col ..][0..8].*);
                        const vec_slice = @as(@Vector(8, f32), vector[col..][0..8].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                4 => {
                    while (col + 4 <= cols) : (col += 4) {
                        const matrix_row = @as(@Vector(4, f32), matrix[row_start + col ..][0..4].*);
                        const vec_slice = @as(@Vector(4, f32), vector[col..][0..4].*);
                        acc_row += @reduce(.Add, matrix_row * vec_slice);
                    }
                },
                else => {},
            }

            // Handle remaining columns
            while (col < cols) : (col += 1) {
                acc_row += matrix[row_start + col] * vector[col];
            }

            result[row] = acc_row;
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

/// Compile-time feature detection
pub const Features = struct {
    pub const has_simd = @hasDecl(std.simd, "f32x4");
    pub const has_avx = @import("builtin").target.cpu.arch == .x86_64 and
        std.Target.x86.featureSetHas(@import("builtin").target.cpu.features, .avx);
    pub const has_neon = @import("builtin").target.cpu.arch == .aarch64 and
        std.Target.aarch64.featureSetHas(@import("builtin").target.cpu.features, .neon);
};

/// Common validation utilities
pub const Validation = struct {
    /// Validate that dimensions match
    pub fn validateDimensions(expected: usize, actual: usize) FrameworkError!void {
        if (expected != actual) {
            return FrameworkError.InvalidDimensions;
        }
    }

    /// Validate that slice is not empty
    pub fn validateNonEmpty(slice: anytype) FrameworkError!void {
        if (slice.len == 0) {
            return FrameworkError.InvalidData;
        }
    }

    /// Validate alignment requirements
    pub fn validateAlignment(ptr: anytype, alignment: usize) FrameworkError!void {
        if (@intFromPtr(ptr) % alignment != 0) {
            return FrameworkError.InvalidAlignment;
        }
    }
};

test "core framework error handling" {
    const testing = std.testing;

    // Test dimension validation
    try Validation.validateDimensions(128, 128);
    try testing.expectError(FrameworkError.InvalidDimensions, Validation.validateDimensions(128, 256));

    // Test empty slice validation
    const empty_slice: []const f32 = &.{};
    const valid_slice = &[_]f32{ 1.0, 2.0, 3.0 };

    try testing.expectError(FrameworkError.InvalidData, Validation.validateNonEmpty(empty_slice));
    try Validation.validateNonEmpty(valid_slice);
}

// =============================================================================
// SIMD CONVENIENCE RE-EXPORTS
// =============================================================================

// Re-export commonly used types
pub const f32x4 = Vector.f32x4;
pub const f32x8 = Vector.f32x8;
pub const f32x16 = Vector.f32x16;

// Re-export operations
pub const distance = VectorOps.distance;
pub const cosineSimilarity = VectorOps.cosineSimilarity;
pub const add = VectorOps.add;
pub const dotProduct = VectorOps.dotProduct;
pub const scale = VectorOps.scale;
pub const matrixVectorMultiply = MatrixOps.matrixVectorMultiply;

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
