//! SIMD Optimizations for Enhanced Graphics Performance
//!
//! This module provides SIMD (Single Instruction, Multiple Data) operations
//! using Zig's @Vector type for parallel processing in graphics computations.

const std = @import("std");
const gpu = @import("../mod.zig");

/// Vector types for common graphics operations
pub const VectorTypes = struct {
    /// 4-component float vector (RGBA, XYZW)
    pub const Vec4f = @Vector(4, f32);
    /// 3-component float vector (RGB, XYZ)
    pub const Vec3f = @Vector(3, f32);
    /// 2-component float vector (UV, XY)
    pub const Vec2f = @Vector(2, f32);

    /// 4-component integer vector
    pub const Vec4i = @Vector(4, i32);
    /// 3-component integer vector
    pub const Vec3i = @Vector(3, i32);
    /// 2-component integer vector
    pub const Vec2i = @Vector(2, i32);

    /// 4-component unsigned integer vector
    pub const Vec4u = @Vector(4, u32);
    /// 3-component unsigned integer vector
    pub const Vec3u = @Vector(3, u32);
    /// 2-component unsigned integer vector
    pub const Vec2u = @Vector(2, u32);

    /// 4x4 matrix as 4 vectors
    pub const Mat4x4f = [4]Vec4f;
    /// 3x3 matrix as 3 vectors
    pub const Mat3x3f = [3]Vec3f;
    /// 2x2 matrix as 2 vectors
    pub const Mat2x2f = [2]Vec2f;
};

/// SIMD math operations
pub const SIMDMath = struct {
    /// Vector addition
    pub fn add(a: anytype, b: anytype) @TypeOf(a) {
        return a + b;
    }

    /// Vector subtraction
    pub fn sub(a: anytype, b: anytype) @TypeOf(a) {
        return a - b;
    }

    /// Vector multiplication
    pub fn mul(a: anytype, b: anytype) @TypeOf(a) {
        return a * b;
    }

    /// Vector division
    pub fn div(a: anytype, b: anytype) @TypeOf(a) {
        return a / b;
    }

    /// Vector dot product
    pub fn dot(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f) f32 {
        const product = a * b;
        return product[0] + product[1] + product[2] + product[3];
    }

    /// Vector cross product (3D)
    pub fn cross(a: VectorTypes.Vec3f, b: VectorTypes.Vec3f) VectorTypes.Vec3f {
        return VectorTypes.Vec3f{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    }

    /// Vector length
    pub fn length(v: VectorTypes.Vec4f) f32 {
        return @sqrt(dot(v, v));
    }

    /// Vector normalization
    pub fn normalize(v: VectorTypes.Vec4f) VectorTypes.Vec4f {
        const len = length(v);
        if (len == 0.0) return VectorTypes.Vec4f{ 0, 0, 0, 0 };
        return v / @as(VectorTypes.Vec4f, @splat(len));
    }

    /// Vector linear interpolation
    pub fn lerp(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f, t: f32) VectorTypes.Vec4f {
        const t_vec = @as(VectorTypes.Vec4f, @splat(t));
        return a + (b - a) * t_vec;
    }

    /// Matrix-vector multiplication (4x4)
    pub fn mat4MulVec4(m: VectorTypes.Mat4x4f, v: VectorTypes.Vec4f) VectorTypes.Vec4f {
        const x = dot(m[0], v);
        const y = dot(m[1], v);
        const z = dot(m[2], v);
        const w = dot(m[3], v);
        return VectorTypes.Vec4f{ x, y, z, w };
    }

    /// Matrix multiplication (4x4)
    pub fn mat4MulMat4(a: VectorTypes.Mat4x4f, b: VectorTypes.Mat4x4f) VectorTypes.Mat4x4f {
        return VectorTypes.Mat4x4f{
            mat4MulVec4(a, b[0]),
            mat4MulVec4(a, b[1]),
            mat4MulVec4(a, b[2]),
            mat4MulVec4(a, b[3]),
        };
    }

    /// Advanced SIMD operations for AI/ML workloads
    /// SIMD fused multiply-add (FMA) operation
    pub fn fma(a: anytype, b: anytype, c: anytype) @TypeOf(a) {
        return a * b + c;
    }

    /// SIMD horizontal sum (sum all elements in vector)
    pub fn horizontalSum(v: anytype) std.meta.Child(@TypeOf(v)) {
        return @reduce(.Add, v);
    }

    /// SIMD horizontal maximum (find max element in vector)
    pub fn horizontalMax(v: anytype) std.meta.Child(@TypeOf(v)) {
        return @reduce(.Max, v);
    }

    /// SIMD horizontal minimum (find min element in vector)
    pub fn horizontalMin(v: anytype) std.meta.Child(@TypeOf(v)) {
        return @reduce(.Min, v);
    }

    /// SIMD element-wise maximum
    pub fn max(a: anytype, b: anytype) @TypeOf(a) {
        return @max(a, b);
    }

    /// SIMD element-wise minimum
    pub fn min(a: anytype, b: anytype) @TypeOf(a) {
        return @min(a, b);
    }

    /// SIMD element-wise absolute value
    pub fn abs(v: anytype) @TypeOf(v) {
        return @abs(v);
    }

    /// SIMD element-wise square root
    pub fn sqrt(v: anytype) @TypeOf(v) {
        return @sqrt(v);
    }

    /// SIMD element-wise reciprocal square root
    pub fn rsqrt(v: anytype) @TypeOf(v) {
        // Compute reciprocal square root as 1.0 / sqrt(v)
        const one = @as(@TypeOf(v), @splat(1.0));
        return one / @sqrt(v);
    }

    /// SIMD element-wise exponential
    pub fn exp(v: anytype) @TypeOf(v) {
        // Taylor series approximation for exp
        const x2 = v * v;
        const x3 = x2 * v;
        const x4 = x3 * v;
        const x5 = x4 * v;
        return @as(@TypeOf(v), @splat(1.0)) + v + x2 / @as(@TypeOf(v), @splat(2.0)) +
            x3 / @as(@TypeOf(v), @splat(6.0)) + x4 / @as(@TypeOf(v), @splat(24.0)) +
            x5 / @as(@TypeOf(v), @splat(120.0));
    }

    /// SIMD element-wise logarithm (natural log)
    pub fn log(v: anytype) @TypeOf(v) {
        // Approximation using log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
        const x = v - @as(@TypeOf(v), @splat(1.0));
        const x2 = x * x;
        const x3 = x2 * x;
        const x4 = x3 * x;
        return x - x2 / @as(@TypeOf(v), @splat(2.0)) + x3 / @as(@TypeOf(v), @splat(3.0)) -
            x4 / @as(@TypeOf(v), @splat(4.0));
    }

    /// SIMD element-wise sigmoid activation
    pub fn sigmoid(v: anytype) @TypeOf(v) {
        const neg_v = -v;
        const exp_neg_v = exp(neg_v);
        return @as(@TypeOf(v), @splat(1.0)) / (@as(@TypeOf(v), @splat(1.0)) + exp_neg_v);
    }

    /// SIMD element-wise tanh activation
    pub fn tanh(v: anytype) @TypeOf(v) {
        const exp_2v = exp(v * @as(@TypeOf(v), @splat(2.0)));
        return (exp_2v - @as(@TypeOf(v), @splat(1.0))) / (exp_2v + @as(@TypeOf(v), @splat(1.0)));
    }

    /// SIMD element-wise ReLU activation
    pub fn relu(v: anytype) @TypeOf(v) {
        return max(v, @as(@TypeOf(v), @splat(0.0)));
    }

    /// SIMD element-wise Leaky ReLU activation
    pub fn leakyRelu(v: anytype, alpha: std.meta.Child(@TypeOf(v))) @TypeOf(v) {
        const alpha_vec = @as(@TypeOf(v), @splat(alpha));
        return max(v, v * alpha_vec);
    }

    /// SIMD element-wise GELU activation (approximation)
    pub fn gelu(v: anytype) @TypeOf(v) {
        const sqrt_2_over_pi = @as(@TypeOf(v), @splat(@sqrt(2.0 / std.math.pi)));
        const coeff = v * sqrt_2_over_pi;
        return v * sigmoid(coeff * @as(@TypeOf(v), @splat(1.702)));
    }

    /// SIMD matrix multiplication (optimized for small matrices)
    pub fn matMulSIMD(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, p: usize) void {
        // Optimized matrix multiplication using SIMD
        const Vec4f = @Vector(4, f32);
        const Vec8f = @Vector(8, f32);
        _ = Vec8f; // autofix

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < p) : (j += 4) {
                if (j + 4 <= p) {
                    // Process 4 output elements at once
                    var sum = @as(Vec4f, @splat(0.0));
                    var k: usize = 0;
                    while (k < n) : (k += 1) {
                        const a_val = @as(Vec4f, @splat(a[i * n + k]));
                        const b_vals = Vec4f{
                            b[k * p + j],
                            b[k * p + j + 1],
                            b[k * p + j + 2],
                            b[k * p + j + 3],
                        };
                        sum += a_val * b_vals;
                    }
                    c[i * p + j] = sum[0];
                    c[i * p + j + 1] = sum[1];
                    c[i * p + j + 2] = sum[2];
                    c[i * p + j + 3] = sum[3];
                } else {
                    // Handle remaining elements
                    for (0..p - j) |jj| {
                        var sum: f32 = 0.0;
                        for (0..n) |k| {
                            sum += a[i * n + k] * b[k * p + j + jj];
                        }
                        c[i * p + j + jj] = sum;
                    }
                }
            }
        }
    }

    /// SIMD vectorized softmax (numerically stable)
    pub fn softmaxSIMD(input: []f32, output: []f32) void {
        // Find maximum for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        for (input) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exp(x - max) and sum
        var sum: f32 = 0.0;
        for (input, 0..) |val, i| {
            const exp_val = std.math.exp(val - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        const sum_vec = @as(@Vector(input.len, f32), @splat(sum));
        const output_vec = @as(@Vector(input.len, f32), output[0..].*) / sum_vec;
        @memcpy(output, std.mem.sliceAsBytes(output_vec[0..]));
    }

    /// SIMD batch normalization (vectorized)
    pub fn batchNormSIMD(input: []f32, output: []f32, gamma: f32, beta: f32, epsilon: f32) void {
        // Compute mean
        var mean: f32 = 0.0;
        for (input) |val| {
            mean += val;
        }
        mean /= @as(f32, @floatFromInt(input.len));

        // Compute variance
        var variance: f32 = 0.0;
        for (input) |val| {
            const diff = val - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(input.len));

        // Normalize
        const inv_std = 1.0 / @sqrt(variance + epsilon);
        for (input, 0..) |val, i| {
            output[i] = gamma * (val - mean) * inv_std + beta;
        }
    }
};

/// SIMD graphics operations
pub const SIMDGraphics = struct {
    /// Color blending (alpha blending)
    pub fn blendColors(src: VectorTypes.Vec4f, dst: VectorTypes.Vec4f) VectorTypes.Vec4f {
        const alpha = @as(VectorTypes.Vec4f, @splat(src[3]));
        const inv_alpha = @as(VectorTypes.Vec4f, @splat(1.0 - src[3]));
        return src * alpha + dst * inv_alpha;
    }

    /// Color space conversion (RGB to HSV)
    pub fn rgbToHsv(rgb: VectorTypes.Vec3f) VectorTypes.Vec3f {
        const r = rgb[0];
        const g = rgb[1];
        const b = rgb[2];

        const max_val = @max(r, @max(g, b));
        const min_val = @min(r, @min(g, b));
        const delta = max_val - min_val;

        var h: f32 = 0.0;
        if (delta != 0.0) {
            if (max_val == r) {
                h = 60.0 * ((g - b) / delta);
            } else if (max_val == g) {
                h = 60.0 * (2.0 + (b - r) / delta);
            } else {
                h = 60.0 * (4.0 + (r - g) / delta);
            }
            if (h < 0.0) h += 360.0;
        }

        const s = if (max_val == 0.0) 0.0 else delta / max_val;
        const v = max_val;

        return VectorTypes.Vec3f{ h, s, v };
    }

    /// Color space conversion (HSV to RGB)
    pub fn hsvToRgb(hsv: VectorTypes.Vec3f) VectorTypes.Vec3f {
        const h = hsv[0];
        const s = hsv[1];
        const v = hsv[2];

        const c = v * s;
        const x = c * (1.0 - @abs(@mod(h / 60.0, 2.0) - 1.0));
        const m = v - c;

        var r: f32 = 0.0;
        var g: f32 = 0.0;
        var b: f32 = 0.0;

        if (h < 60.0) {
            r = c;
            g = x;
            b = 0.0;
        } else if (h < 120.0) {
            r = x;
            g = c;
            b = 0.0;
        } else if (h < 180.0) {
            r = 0.0;
            g = c;
            b = x;
        } else if (h < 240.0) {
            r = 0.0;
            g = x;
            b = c;
        } else if (h < 300.0) {
            r = x;
            g = 0.0;
            b = c;
        } else {
            r = c;
            g = 0.0;
            b = x;
        }

        return VectorTypes.Vec3f{ r + m, g + m, b + m };
    }

    /// Gamma correction
    pub fn gammaCorrect(color: VectorTypes.Vec3f, gamma: f32) VectorTypes.Vec3f {
        const gamma_vec = @as(VectorTypes.Vec3f, @splat(gamma));
        return color * gamma_vec; // Simplified gamma correction
    }

    /// Tone mapping (Reinhard)
    pub fn toneMapReinhard(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
        return color / (@as(VectorTypes.Vec3f, @splat(1.0)) + color);
    }

    /// Tone mapping (ACES)
    pub fn toneMapACES(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
        const a = VectorTypes.Vec3f{ 2.51, 2.51, 2.51 };
        const b = VectorTypes.Vec3f{ 0.03, 0.03, 0.03 };
        const c = VectorTypes.Vec3f{ 2.43, 2.43, 2.43 };
        const d = VectorTypes.Vec3f{ 0.59, 0.59, 0.59 };
        const e = VectorTypes.Vec3f{ 0.14, 0.14, 0.14 };

        return (color * (a * color + b)) / (color * (c * color + d) + e);
    }
};

/// SIMD compute operations
pub const SIMDCompute = struct {
    /// Parallel array addition
    pub fn addArrays(a: []const f32, b: []const f32, result: []f32) void {
        const chunk_size = 4;
        var i: usize = 0;

        // Process in chunks of 4 using SIMD
        while (i + chunk_size <= a.len) : (i += chunk_size) {
            const a_vec = @as(VectorTypes.Vec4f, a[i .. i + chunk_size].*);
            const b_vec = @as(VectorTypes.Vec4f, b[i .. i + chunk_size].*);
            const result_vec = a_vec + b_vec;
            result[i .. i + chunk_size].* = result_vec;
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Parallel array multiplication
    pub fn mulArrays(a: []const f32, b: []const f32, result: []f32) void {
        const chunk_size = 4;
        var i: usize = 0;

        // Process in chunks of 4 using SIMD
        while (i + chunk_size <= a.len) : (i += chunk_size) {
            const a_vec = @as(VectorTypes.Vec4f, a[i .. i + chunk_size].*);
            const b_vec = @as(VectorTypes.Vec4f, b[i .. i + chunk_size].*);
            const result_vec = a_vec * b_vec;
            result[i .. i + chunk_size].* = result_vec;
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] * b[i];
        }
    }

    /// Parallel array scaling
    pub fn scaleArray(a: []const f32, scale: f32, result: []f32) void {
        const chunk_size = 4;
        const scale_vec = @as(VectorTypes.Vec4f, @splat(scale));
        var i: usize = 0;

        // Process in chunks of 4 using SIMD
        while (i + chunk_size <= a.len) : (i += chunk_size) {
            const a_vec = @as(VectorTypes.Vec4f, a[i .. i + chunk_size].*);
            const result_vec = a_vec * scale_vec;
            result[i .. i + chunk_size].* = result_vec;
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result[i] = a[i] * scale;
        }
    }

    /// Parallel array sum reduction
    pub fn sumArray(a: []const f32) f32 {
        const chunk_size = 4;
        var sum_vec = @as(VectorTypes.Vec4f, @splat(@as(f32, 0.0)));
        var i: usize = 0;

        // Process in chunks of 4 using SIMD
        while (i + chunk_size <= a.len) : (i += chunk_size) {
            const a_vec = @as(VectorTypes.Vec4f, a[i .. i + chunk_size].*);
            sum_vec += a_vec;
        }

        // Sum the vector components
        var sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i];
        }

        return sum;
    }

    /// Parallel array dot product
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        const chunk_size = 4;
        var dot_vec = @as(VectorTypes.Vec4f, @splat(@as(f32, 0.0)));
        var i: usize = 0;

        // Process in chunks of 4 using SIMD
        while (i + chunk_size <= a.len) : (i += chunk_size) {
            const a_vec = @as(VectorTypes.Vec4f, a[i .. i + chunk_size].*);
            const b_vec = @as(VectorTypes.Vec4f, b[i .. i + chunk_size].*);
            dot_vec += a_vec * b_vec;
        }

        // Sum the vector components
        var dot = dot_vec[0] + dot_vec[1] + dot_vec[2] + dot_vec[3];

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            dot += a[i] * b[i];
        }

        return dot;
    }
};

/// SIMD performance benchmarks
pub const SIMDBenchmarks = struct {
    /// Benchmark SIMD vs scalar operations
    pub fn benchmarkSIMDvsScalar(allocator: std.mem.Allocator, array_size: usize) !void {
        const a = try allocator.alloc(f32, array_size);
        const b = try allocator.alloc(f32, array_size);
        const result = try allocator.alloc(f32, array_size);
        defer allocator.free(a);
        defer allocator.free(b);
        defer allocator.free(result);

        // Initialize arrays with random data
        for (a, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 1000)) / 1000.0;
        }
        for (b, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt((i + 500) % 1000)) / 1000.0;
        }

        // Benchmark scalar addition
        const scalar_start = std.time.nanoTimestamp;
        for (a, b, 0..) |a_val, b_val, i| {
            result[i] = a_val + b_val;
        }
        const scalar_time = std.time.nanoTimestamp - scalar_start;

        // Benchmark SIMD addition
        const simd_start = std.time.nanoTimestamp;
        SIMDCompute.addArrays(a, b, result);
        const simd_time = std.time.nanoTimestamp - simd_start;

        std.log.info("📊 SIMD Performance Benchmark (Array Size: {})", .{array_size});
        std.log.info("  - Scalar Time: {} ns", .{scalar_time});
        std.log.info("  - SIMD Time: {} ns", .{simd_time});
        std.log.info("  - Speedup: {d:.2}x", .{@as(f64, @floatFromInt(scalar_time)) / @as(f64, @floatFromInt(simd_time))});
    }

    /// Benchmark matrix operations
    pub fn benchmarkMatrixOperations(allocator: std.mem.Allocator) !void {
        _ = allocator;

        const iterations = 1000000;

        // Benchmark matrix-vector multiplication
        const mat = VectorTypes.Mat4x4f{
            VectorTypes.Vec4f{ 1, 0, 0, 0 },
            VectorTypes.Vec4f{ 0, 1, 0, 0 },
            VectorTypes.Vec4f{ 0, 0, 1, 0 },
            VectorTypes.Vec4f{ 0, 0, 0, 1 },
        };
        const vec = VectorTypes.Vec4f{ 1, 2, 3, 4 };

        const start = std.time.nanoTimestamp;
        var i: u32 = 0;
        while (i < iterations) : (i += 1) {
            _ = SIMDMath.mat4MulVec4(mat, vec);
        }
        const time = std.time.nanoTimestamp - start;

        std.log.info("📊 Matrix-Vector Multiplication Benchmark", .{});
        std.log.info("  - Iterations: {}", .{iterations});
        std.log.info("  - Total Time: {} ns", .{time});
        std.log.info("  - Time per Operation: {} ns", .{@divTrunc(time, iterations)});
    }
};
