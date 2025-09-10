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

    /// Subtract two vectors using SIMD
    pub fn subtract(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;

        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;

        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..16], @as([16]f32, diff)[0..]);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..8], @as([8]f32, diff)[0..]);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const diff = va - vb;
                    @memcpy(result[i..][0..4], @as([4]f32, diff)[0..]);
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

    /// Element-wise multiply: result = a * b
    pub fn multiply(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;
        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const prod = va * vb;
                    @memcpy(result[i..][0..16], @as([16]f32, prod)[0..]);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const prod = va * vb;
                    @memcpy(result[i..][0..8], @as([8]f32, prod)[0..]);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const prod = va * vb;
                    @memcpy(result[i..][0..4], @as([4]f32, prod)[0..]);
                }
            },
            else => {},
        }
        while (i < a.len) : (i += 1) {
            result[i] = a[i] * b[i];
        }
    }

    /// Element-wise divide: result = a / b (no special NaN handling)
    pub fn divide(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;
        const optimal_size = Vector.getOptimalSize(a.len);
        var i: usize = 0;
        switch (optimal_size) {
            16 => {
                while (i + 16 <= a.len) : (i += 16) {
                    const va = @as(@Vector(16, f32), a[i..][0..16].*);
                    const vb = @as(@Vector(16, f32), b[i..][0..16].*);
                    const quot = va / vb;
                    @memcpy(result[i..][0..16], @as([16]f32, quot)[0..]);
                }
            },
            8 => {
                while (i + 8 <= a.len) : (i += 8) {
                    const va = @as(@Vector(8, f32), a[i..][0..8].*);
                    const vb = @as(@Vector(8, f32), b[i..][0..8].*);
                    const quot = va / vb;
                    @memcpy(result[i..][0..8], @as([8]f32, quot)[0..]);
                }
            },
            4 => {
                while (i + 4 <= a.len) : (i += 4) {
                    const va = @as(@Vector(4, f32), a[i..][0..4].*);
                    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
                    const quot = va / vb;
                    @memcpy(result[i..][0..4], @as([4]f32, quot)[0..]);
                }
            },
            else => {},
        }
        while (i < a.len) : (i += 1) {
            result[i] = a[i] / b[i];
        }
    }

    /// Element-wise min: result[i] = min(a[i], b[i])
    pub fn min(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = @min(a[i], b[i]);
    }

    /// Element-wise max: result[i] = max(a[i], b[i])
    pub fn max(result: []f32, a: []const f32, b: []const f32) void {
        if (a.len != b.len or result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = @max(a[i], b[i]);
    }

    /// Element-wise absolute value
    pub fn abs(result: []f32, a: []const f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = @abs(a[i]);
    }

    /// Clamp elements to [lo, hi]
    pub fn clamp(result: []f32, a: []const f32, lo: f32, hi: f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            const x = a[i];
            result[i] = if (x < lo) lo else if (x > hi) hi else x;
        }
    }

    /// Element-wise square: result[i] = a[i]^2
    pub fn square(result: []f32, a: []const f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = a[i] * a[i];
    }

    /// Element-wise sqrt
    pub fn sqrt(result: []f32, a: []const f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = @sqrt(a[i]);
    }

    /// Element-wise exp
    pub fn exp(result: []f32, a: []const f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = std.math.exp(a[i]);
    }

    /// Element-wise natural log (ln)
    pub fn log(result: []f32, a: []const f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        while (i < a.len) : (i += 1) result[i] = std.math.log(a[i]);
    }

    /// Add scalar to vector: result = a + s
    pub fn addScalar(result: []f32, a: []const f32, s: f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        const vs4 = @as(@Vector(4, f32), @splat(s));
        while (i + 4 <= a.len) : (i += 4) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            const s4 = va + vs4;
            @memcpy(result[i..][0..4], @as([4]f32, s4)[0..]);
        }
        while (i < a.len) : (i += 1) result[i] = a[i] + s;
    }

    /// Subtract scalar from vector: result = a - s
    pub fn subScalar(result: []f32, a: []const f32, s: f32) void {
        if (result.len != a.len) return;
        var i: usize = 0;
        const vs4 = @as(@Vector(4, f32), @splat(s));
        while (i + 4 <= a.len) : (i += 4) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            const diff = va - vs4;
            @memcpy(result[i..][0..4], @as([4]f32, diff)[0..]);
        }
        while (i < a.len) : (i += 1) result[i] = a[i] - s;
    }

    /// L1 (Manhattan) distance
    pub fn l1Distance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        var acc: f32 = 0.0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            const vb = @as(@Vector(4, f32), b[i..][0..4].*);
            const diff = va - vb;
            const tmp: [4]f32 = @as([4]f32, diff);
            // absolute values
            var j: usize = 0;
            while (j < 4) : (j += 1) acc += @abs(tmp[j]);
        }
        while (i < a.len) : (i += 1) acc += @abs(a[i] - b[i]);
        return acc;
    }

    /// L-infinity (Chebyshev) distance
    pub fn linfDistance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        var maxd: f32 = 0.0;
        var i: usize = 0;
        while (i < a.len) : (i += 1) {
            const d = @abs(a[i] - b[i]);
            if (d > maxd) maxd = d;
        }
        return maxd;
    }

    /// Sum of elements
    pub fn sum(a: []const f32) f32 {
        var acc: f32 = 0.0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const va = @as(@Vector(4, f32), a[i..][0..4].*);
            acc += @reduce(.Add, va);
        }
        while (i < a.len) : (i += 1) acc += a[i];
        return acc;
    }

    /// Mean of elements
    pub fn mean(a: []const f32) f32 {
        if (a.len == 0) return 0.0;
        return VectorOps.sum(a) / @as(f32, @floatFromInt(a.len));
    }

    /// Variance (population)
    pub fn variance(a: []const f32) f32 {
        if (a.len == 0) return 0.0;
        const m = VectorOps.mean(a);
        var acc: f32 = 0.0;
        for (a) |x| {
            const d = x - m;
            acc += d * d;
        }
        return acc / @as(f32, @floatFromInt(a.len));
    }

    /// Standard deviation (population)
    pub fn stddev(a: []const f32) f32 {
        return @sqrt(VectorOps.variance(a));
    }

    /// AXPY operation: y = a*x + y
    pub fn axpy(y: []f32, a: f32, x: []const f32) void {
        if (y.len != x.len) return;
        var i: usize = 0;
        const av4 = @as(@Vector(4, f32), @splat(a));
        while (i + 4 <= y.len) : (i += 4) {
            const xv = @as(@Vector(4, f32), x[i..][0..4].*);
            const yv = @as(@Vector(4, f32), y[i..][0..4].*);
            const res = yv + (av4 * xv);
            @memcpy(y[i..][0..4], @as([4]f32, res)[0..]);
        }
        while (i < y.len) : (i += 1) y[i] = y[i] + a * x[i];
    }

    /// Fused multiply-add: result = x*y + z
    pub fn fma(result: []f32, x: []const f32, y: []const f32, z: []const f32) void {
        if (!(result.len == x.len and x.len == y.len and y.len == z.len)) return;
        var i: usize = 0;
        while (i + 4 <= result.len) : (i += 4) {
            const xv = @as(@Vector(4, f32), x[i..][0..4].*);
            const yv = @as(@Vector(4, f32), y[i..][0..4].*);
            const zv = @as(@Vector(4, f32), z[i..][0..4].*);
            const rv = (xv * yv) + zv;
            @memcpy(result[i..][0..4], @as([4]f32, rv)[0..]);
        }
        while (i < result.len) : (i += 1) result[i] = x[i] * y[i] + z[i];
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
                var acc_inner: f32 = 0.0;
                var l: usize = 0;

                // SIMD-optimized inner loop
                switch (optimal_size) {
                    16 => {
                        while (l + 16 <= n) : (l += 16) {
                            const a_row = Vector.load(Vector.f32x16, a[i * n + l ..][0..16]);
                            const b_col = Vector.load(Vector.f32x16, b[l * k + j ..][0..16]);
                            acc_inner += std.simd.f32x16.reduce_add(a_row * b_col);
                        }
                    },
                    8 => {
                        while (l + 8 <= n) : (l += 8) {
                            const a_row = Vector.load(Vector.f32x8, a[i * n + l ..][0..8]);
                            const b_col = Vector.load(Vector.f32x8, b[l * k + j ..][0..8]);
                            acc_inner += std.simd.f32x8.reduce_add(a_row * b_col);
                        }
                    },
                    4 => {
                        while (l + 4 <= n) : (l += 4) {
                            const a_row = Vector.load(Vector.f32x4, a[i * n + l ..][0..4]);
                            const b_col = Vector.load(Vector.f32x4, b[l * k + j ..][0..4]);
                            acc_inner += std.simd.f32x4.reduce_add(a_row * b_col);
                        }
                    },
                    else => {},
                }

                // Handle remaining elements
                while (l < n) : (l += 1) {
                    acc_inner += a[i * n + l] * b[l * k + j];
                }

                result[i * k + j] = acc_inner;
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
                        const row_data = Vector.load(Vector.f32x16, matrix[i * cols + j ..][0..16]);
                        Vector.store(result[j * rows + i ..][0..16], row_data);
                    }
                },
                8 => {
                    while (j + 8 <= cols) : (j += 8) {
                        const row_data = Vector.load(Vector.f32x8, matrix[i * cols + j ..][0..8]);
                        Vector.store(result[j * rows + i ..][0..8], row_data);
                    }
                },
                4 => {
                    while (j + 4 <= cols) : (j += 4) {
                        const row_data = Vector.load(Vector.f32x4, matrix[i * cols + j ..][0..4]);
                        Vector.store(result[j * rows + i ..][0..4], row_data);
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
pub const multiply = VectorOps.multiply;
pub const divide = VectorOps.divide;
pub const min = VectorOps.min;
pub const max = VectorOps.max;
pub const abs = VectorOps.abs;
pub const clamp = VectorOps.clamp;
pub const square = VectorOps.square;
pub const sqrt = VectorOps.sqrt;
pub const exp = VectorOps.exp;
pub const log = VectorOps.log;
pub const addScalar = VectorOps.addScalar;
pub const subScalar = VectorOps.subScalar;
pub const l1Distance = VectorOps.l1Distance;
pub const linfDistance = VectorOps.linfDistance;
pub const sum = VectorOps.sum;
pub const mean = VectorOps.mean;
pub const variance = VectorOps.variance;
pub const stddev = VectorOps.stddev;
pub const axpy = VectorOps.axpy;
pub const fma = VectorOps.fma;
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

    // Test element-wise multiply/divide
    multiply(result, &a, &b);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    divide(result, &b, &a);
    try testing.expectEqual(@as(f32, 1.5), result[1]);

    // Test min/max and L1 distance
    min(result, &a, &b);
    try testing.expectEqual(@as(f32, 1.0), result[0]);
    max(result, &a, &b);
    try testing.expectEqual(@as(f32, 2.0), result[0]);
    const l1 = l1Distance(&a, &b);
    try testing.expectEqual(@as(f32, 8.0), l1);
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
