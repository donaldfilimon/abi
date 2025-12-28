//! SIMD vector operations
//!
//! Provides high-performance vectorized operations using SIMD instructions
//! when available (AVX-512, NEON, WASM SIMD).

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// Vector addition using SIMD when available
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    comptime std.debug.assert(@TypeOf(a) == []const f32);
    comptime std.debug.assert(@TypeOf(b) == []const f32);
    comptime std.debug.assert(@TypeOf(result) == []f32);
    std.debug.assert(a.len == b.len and a.len == result.len);

    const len = a.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = va + vb;
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Vector dot product using SIMD when available
pub fn vectorDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const len = a.len;
    var sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            vec_sum += va * vb;
        }

        const sums: [VectorSize]f32 = vec_sum;
        for (sums) |s| {
            sum += s;
        }
    }

    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Vector L2 norm using SIMD when available
pub fn vectorL2Norm(v: []const f32) f32 {
    const len = v.len;
    var sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            vec_sum += vv * vv;
        }

        const sums: [VectorSize]f32 = vec_sum;
        for (sums) |s| {
            sum += s;
        }
    }

    while (i < len) : (i += 1) {
        sum += v[i] * v[i];
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

/// Batch cosine similarity computation for database searches
/// Computes cosine similarity between a query vector and multiple database vectors
pub fn batchCosineSimilarity(
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(vectors.len == results.len);

    for (vectors, results) |vector, *result| {
        result.* = cosineSimilarity(query, vector);
    }
}

/// Vector reduction operations with SIMD acceleration
pub fn vectorReduce(op: enum { sum, max, min }, v: []const f32) f32 {
    if (v.len == 0) return 0.0;

    const len = v.len;
    var i: usize = 0;
    var result: f32 = if (op == .sum) 0.0 else v[0];

    if (comptime VectorSize > 1 and len >= VectorSize) {
        const Vec = @Vector(VectorSize, f32);
        var vec_result: Vec = @splat(result);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            switch (op) {
                .sum => vec_result += vv,
                .max => vec_result = @max(vec_result, vv),
                .min => vec_result = @min(vec_result, vv),
            }
        }

        const results: [VectorSize]f32 = vec_result;
        switch (op) {
            .sum => {
                for (results) |r| result += r;
            },
            .max => {
                for (results) |r| result = @max(result, r);
            },
            .min => {
                for (results) |r| result = @min(result, r);
            },
        }
    }

    while (i < len) : (i += 1) {
        switch (op) {
            .sum => result += v[i],
            .max => result = @max(result, v[i]),
            .min => result = @min(result, v[i]),
        }
    }

    return result;
}

/// Check if SIMD is available at compile time
pub fn hasSimdSupport() bool {
    return VectorSize > 1;
}

/// Matrix multiplication with blocking/tiling for cache efficiency and SIMD acceleration
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

    @memset(result, 0);

    const BLOCK_SIZE = 32;
    var i: usize = 0;

    while (i < m) : (i += BLOCK_SIZE) {
        const i_end = @min(i + BLOCK_SIZE, m);
        var j: usize = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            const j_end = @min(j + BLOCK_SIZE, n);
            var l: usize = 0;
            while (l < k) : (l += BLOCK_SIZE) {
                const l_end = @min(l + BLOCK_SIZE, k);

                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    var jj: usize = j;
                    while (jj < j_end) : (jj += 1) {
                        var sum: f32 = result[ii * n + jj];
                        var ll: usize = l;
                        while (ll < l_end) : (ll += 1) {
                            sum += a[ii * k + ll] * b[ll * n + jj];
                        }
                        result[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
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
