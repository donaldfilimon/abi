//! SIMD vector operations
//!
//! Provides high-performance vectorized operations using SIMD instructions
//! when available (AVX-512, NEON, WASM SIMD).

const std = @import("std");

/// Vector addition using SIMD when available
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and a.len == result.len);

    var i: usize = 0;
    const len = a.len;

    // Use SIMD for large vectors when available
    if (comptime std.simd.suggestVectorLength(f32)) |vec_len| {
        if (len >= vec_len) {
            const VecType = std.simd.Vector(vec_len, f32);

            while (i + vec_len <= len) : (i += vec_len) {
                const va: VecType = a[i..][0..vec_len].*;
                const vb: VecType = b[i..][0..vec_len].*;
                result[i..][0..vec_len].* = va + vb;
            }
        }
    }

    // Scalar fallback for remaining elements
    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Vector dot product using SIMD when available
pub fn vectorDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var result: f32 = 0.0;
    var i: usize = 0;
    const len = a.len;

    // Use SIMD for large vectors when available
    if (comptime std.simd.suggestVectorLength(f32)) |vec_len| {
        if (len >= vec_len) {
            const VecType = std.simd.Vector(vec_len, f32);
            var sum_vec = VecType{0} ** VecType{0}; // Zero vector

            while (i + vec_len <= len) : (i += vec_len) {
                const va: VecType = a[i..][0..vec_len].*;
                const vb: VecType = b[i..][0..vec_len].*;
                sum_vec += va * vb;
            }

            // Reduce vector sum to scalar
            for (sum_vec) |v| {
                result += v;
            }
        }
    }

    // Scalar fallback for remaining elements
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// Vector L2 norm using SIMD when available
pub fn vectorL2Norm(v: []const f32) f32 {
    var sum: f32 = 0.0;
    var i: usize = 0;
    const len = v.len;

    // Use SIMD for large vectors when available
    if (comptime std.simd.suggestVectorLength(f32)) |vec_len| {
        if (len >= vec_len) {
            const VecType = std.simd.Vector(vec_len, f32);
            var sum_vec = VecType{0} ** VecType{0}; // Zero vector

            while (i + vec_len <= len) : (i += vec_len) {
                const vv: VecType = v[i..][0..vec_len].*;
                sum_vec += vv * vv;
            }

            // Reduce vector sum to scalar
            for (sum_vec) |s| {
                sum += s;
            }
        }
    }

    // Scalar fallback for remaining elements
    while (i < len) : (i += 1) {
        const val = v[i];
        sum += val * val;
    }

    return @sqrt(sum);
}

/// Cosine similarity using SIMD operations
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dot_product = vectorDot(a, b);
    const norm_a = vectorL2Norm(a);
    const norm_b = vectorL2Norm(b);

    if (norm_a == 0.0 or norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (norm_a * norm_b);
}

/// Matrix multiplication with SIMD acceleration
pub fn matrixMultiply(
    a: []const f32,
    b: []const f32,
    result: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);
    std.debug.assert(result.len == m * n);

    // Initialize result to zero
    @memset(result, 0);

    // Simple implementation - could be optimized further with SIMD
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var sum: f32 = 0.0;
            var l: usize = 0;
            while (l < k) : (l += 1) {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

/// Vector reduction operations
pub fn vectorReduce(op: enum { sum, max, min }, v: []const f32) f32 {
    if (v.len == 0) return 0.0;

    var result = v[0];
    for (v[1..]) |val| {
        switch (op) {
            .sum => result += val,
            .max => result = @max(result, val),
            .min => result = @min(result, val),
        }
    }
    return result;
}

/// Check SIMD support at runtime
pub fn hasSimdSupport() bool {
    // Check for SIMD support - this is a basic check
    // In practice, you'd check CPU features
    return comptime std.simd.suggestVectorLength(f32) != null;
}

test "vector addition works" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    var result: [4]f32 = undefined;

    vectorAdd(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 1.5), result[0]);
    try std.testing.expectEqual(@as(f32, 3.5), result[1]);
    try std.testing.expectEqual(@as(f32, 5.5), result[2]);
    try std.testing.expectEqual(@as(f32, 7.5), result[3]);
}

test "vector dot product works" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    var b = [_]f32{ 4.0, 5.0, 6.0 };

    const result = vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 1e-6);
}

test "vector L2 norm works" {
    var v = [_]f32{ 3.0, 4.0 };

    const result = vectorL2Norm(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-6);
}

test "cosine similarity works" {
    var a = [_]f32{ 1.0, 0.0 };
    var b = [_]f32{ 0.0, 1.0 };

    const result = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-6);
}
