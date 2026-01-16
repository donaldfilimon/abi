//! SIMD vector operations
//!
//! Provides high-performance vectorized operations using SIMD instructions
//! when available (AVX-512, NEON, WASM SIMD).
//!
//! # Safety Requirements
//! All functions require input vectors to be properly sized:
//! - All input vectors must have matching lengths where applicable
//! - Empty vectors are not allowed (undefined behavior)
//! - Result buffers must be pre-allocated with correct size
//!
//! # Performance Notes
//! - Functions use @Vector for SIMD acceleration when VectorSize > 1
//! - Debug builds include additional bounds checking via std.debug.assert
//! - Release builds rely on loop bounds for safety (no debug.assert overhead)

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// Vector addition using SIMD when available
/// @param a First input vector
/// @param b Second input vector (must have same length as a)
/// @param result Output buffer (must have same length as inputs, caller-owned)
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
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
/// @param a First input vector
/// @param b Second input vector (must have same length as a)
/// @return Scalar dot product
pub fn vectorDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
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
/// @param v Input vector (must have len > 0)
/// @return Euclidean norm of the vector
pub fn vectorL2Norm(v: []const f32) f32 {
    std.debug.assert(v.len > 0);
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
/// @param a First input vector (must not be empty)
/// @param b Second input vector (must not be empty and same length as a)
/// @return Cosine similarity in range [-1, 1], or 0.0 for zero-length vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len == 0 or b.len == 0) return 0.0;
    if (a.len != b.len) return 0.0;

    const dot_product = vectorDot(a, b);
    const norm_a = vectorL2Norm(a);
    const norm_b = vectorL2Norm(b);

    if (norm_a == 0.0 or norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (norm_a * norm_b);
}

/// Batch cosine similarity computation with pre-computed query norm
/// Fast version that avoids redundant query norm computation
/// @param query Query vector (must not be empty)
/// @param query_norm Pre-computed L2 norm of query (must be > 0)
/// @param vectors Array of database vectors
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarityFast(
    query: []const f32,
    query_norm: f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(query_norm > 0.0);
    std.debug.assert(vectors.len == results.len);

    for (vectors, results) |vector, *result| {
        const dot = vectorDot(query, vector);
        const vec_norm = vectorL2Norm(vector);
        result.* = if (query_norm > 0 and vec_norm > 0)
            dot / (query_norm * vec_norm)
        else
            0;
    }
}

/// Batch cosine similarity computation for database searches
/// Computes cosine similarity between a query vector and multiple database vectors
/// @param query Query vector (must not be empty)
/// @param vectors Array of database vectors
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarity(
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(vectors.len == results.len);

    const query_norm = vectorL2Norm(query);
    batchCosineSimilarityFast(query, query_norm, vectors, results);
}

/// Vector reduction operations with SIMD acceleration
/// @param op Reduction operation: sum, max, or min
/// @param v Input vector
/// @return Reduced value (0.0 for sum on empty, undefined for min/max on empty)
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

/// SIMD instruction set capabilities detected at compile time
pub const SimdCapabilities = struct {
    vector_size: usize,
    has_simd: bool,
    arch: Arch,

    pub const Arch = enum {
        x86_64,
        aarch64,
        wasm,
        generic,
    };
};

/// Get SIMD capabilities for current platform
pub fn getSimdCapabilities() SimdCapabilities {
    const builtin = @import("builtin");
    const arch: SimdCapabilities.Arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        .wasm32, .wasm64 => .wasm,
        else => .generic,
    };

    return .{
        .vector_size = VectorSize,
        .has_simd = VectorSize > 1,
        .arch = arch,
    };
}

/// Matrix multiplication with blocking/tiling for cache efficiency and SIMD acceleration
/// Computes result[m][n] = a[m][k] * b[k][n]
/// @param a Matrix A (size m x k, row-major order)
/// @param b Matrix B (size k x n, row-major order)
/// @param result Output matrix (size m x n, caller-owned, must be pre-zeroed or will be overwritten)
/// @param m Number of rows in A and result
/// @param n Number of columns in B and result
/// @param k Number of columns in A, rows in B (must match)
pub fn matrixMultiply(
    a: []const f32,
    b: []const f32,
    result: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(m > 0 and n > 0 and k > 0);
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
                    // SIMD-vectorized inner loop: process VectorSize columns at a time
                    var jj: usize = j;
                    if (comptime VectorSize > 1) {
                        const Vec = @Vector(VectorSize, f32);
                        while (jj + VectorSize <= j_end) : (jj += VectorSize) {
                            var vec_sum: Vec = result[ii * n + jj ..][0..VectorSize].*;
                            var ll: usize = l;
                            while (ll < l_end) : (ll += 1) {
                                const a_val: Vec = @splat(a[ii * k + ll]);
                                const b_vec: Vec = b[ll * n + jj ..][0..VectorSize].*;
                                vec_sum += a_val * b_vec;
                            }
                            result[ii * n + jj ..][0..VectorSize].* = vec_sum;
                        }
                    }
                    // Scalar fallback for remaining columns
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

test "matrix multiplication works" {
    // 2x3 * 3x2 = 2x2
    // A = [1 2 3]   B = [7  8 ]   Result = [58  64 ]
    //     [4 5 6]       [9  10]            [139 154]
    //                   [11 12]
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var result: [4]f32 = undefined;

    matrixMultiply(&a, &b, &result, 2, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), result[3], 1e-5);
}

test "matrix multiplication larger" {
    // Test with a matrix size that exercises SIMD paths (8x8)
    const size = 8;
    var a: [size * size]f32 = undefined;
    var b: [size * size]f32 = undefined;
    var result: [size * size]f32 = undefined;

    // Initialize with simple values for verification
    for (0..size) |i| {
        for (0..size) |j| {
            a[i * size + j] = @floatFromInt(i + j);
            b[i * size + j] = @floatFromInt(i * j);
        }
    }

    matrixMultiply(&a, &b, &result, size, size, size);

    // Verify result[0][0] = sum(a[0][k] * b[k][0]) for k=0..7
    // = 0*0 + 1*0 + 2*0 + 3*0 + 4*0 + 5*0 + 6*0 + 7*0 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-4);

    // Verify result[1][1] = sum(a[1][k] * b[k][1]) for k=0..7
    // = 1*0 + 2*1 + 3*2 + 4*3 + 5*4 + 6*5 + 7*6 + 8*7 = 0 + 2 + 6 + 12 + 20 + 30 + 42 + 56 = 168
    try std.testing.expectApproxEqAbs(@as(f32, 168.0), result[size + 1], 1e-4);
}
