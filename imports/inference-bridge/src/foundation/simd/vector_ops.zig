//! Basic SIMD vector operations
//!
//! Provides vectorized add, dot product, L2 norm, cosine similarity,
//! batch operations, and generic reduction.

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

/// Element-wise vector multiplication using SIMD when available
pub fn vectorMul(a: []const f32, b: []const f32, result: []f32) void {
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
            result[i..][0..VectorSize].* = va * vb;
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] * b[i];
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
    var dot_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            vec_sum += va * vb;
        }

        // Use @reduce for efficient horizontal sum (Zig 0.16+)
        dot_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        dot_sum += a[i] * b[i];
    }

    return dot_sum;
}

/// Vector L2 norm using SIMD when available
/// @param v Input vector (must have len > 0)
/// @return Euclidean norm of the vector
pub fn vectorL2Norm(v: []const f32) f32 {
    std.debug.assert(v.len > 0);
    const len = v.len;
    var norm_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            vec_sum += vv * vv;
        }

        // Use @reduce for efficient horizontal sum (Zig 0.16+)
        norm_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        norm_sum += v[i] * v[i];
    }

    return @sqrt(norm_sum);
}

/// Cosine similarity using fused single-pass SIMD
/// Computes dot product and both norms simultaneously, eliminating 2 extra
/// passes over the data compared to separate dot + norm_a + norm_b calls.
/// @param a First input vector (must not be empty)
/// @param b Second input vector (must not be empty and same length as a)
/// @return Cosine similarity in range [-1, 1], or 0.0 for zero-length vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len == 0 or b.len == 0) return 0.0;
    if (a.len != b.len) return 0.0;

    const len = a.len;
    var i: usize = 0;
    var dot_sum: f32 = 0.0;
    var norm_a_sq: f32 = 0.0;
    var norm_b_sq: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var v_dot: Vec = @splat(0.0);
        var v_na: Vec = @splat(0.0);
        var v_nb: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            v_dot += va * vb;
            v_na += va * va;
            v_nb += vb * vb;
        }

        dot_sum = @reduce(.Add, v_dot);
        norm_a_sq = @reduce(.Add, v_na);
        norm_b_sq = @reduce(.Add, v_nb);
    }

    // Scalar tail for remaining elements
    while (i < len) : (i += 1) {
        dot_sum += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    const denom = @sqrt(norm_a_sq) * @sqrt(norm_b_sq);
    return if (denom == 0.0) 0.0 else dot_sum / denom;
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

/// Batch cosine similarity with pre-computed norms for maximum performance
/// Use this when database vector norms are pre-computed and stored
/// @param query Query vector (must not be empty)
/// @param query_norm Pre-computed L2 norm of query (must be > 0)
/// @param vectors Array of database vectors
/// @param vector_norms Pre-computed L2 norms of database vectors (same length as vectors)
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarityPrecomputed(
    query: []const f32,
    query_norm: f32,
    vectors: []const []const f32,
    vector_norms: []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(query_norm > 0.0);
    std.debug.assert(vectors.len == results.len);
    std.debug.assert(vectors.len == vector_norms.len);

    for (vectors, vector_norms, results) |vector, vec_norm, *result| {
        const dot = vectorDot(query, vector);
        result.* = if (query_norm > 0 and vec_norm > 0)
            dot / (query_norm * vec_norm)
        else
            0;
    }
}

/// Batch dot-product computation.
/// Computes the dot product of a single `query` vector against each vector in `vectors`.
/// Results are stored in the `results` slice, which must have the same length as `vectors`.
pub fn batchDotProduct(
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(vectors.len == results.len);
    for (vectors, results) |vec, *res| {
        res.* = vectorDot(query, vec);
    }
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

    if (comptime VectorSize > 1) {
        if (len >= VectorSize) {
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

            // Use @reduce for efficient horizontal reduction (Zig 0.16+)
            switch (op) {
                .sum => result += @reduce(.Add, vec_result),
                .max => result = @max(result, @reduce(.Max, vec_result)),
                .min => result = @min(result, @reduce(.Min, vec_result)),
            }
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

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

test "vectorAdd basic" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var result: [5]f32 = undefined;
    vectorAdd(&a, &b, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 55.0), result[4], 0.001);
}

test "vectorDot product" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    const dot = vectorDot(&a, &b);
    // 1*4 + 2*5 + 3*6 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot, 0.001);
}

test "vectorDot self is squared norm" {
    const a = [_]f32{ 3.0, 4.0 };
    const dot = vectorDot(&a, &a);
    // 9 + 16 = 25
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), dot, 0.001);
}

test "vectorL2Norm" {
    const v = [_]f32{ 3.0, 4.0 };
    const norm = vectorL2Norm(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 0.001);
}

test "cosineSimilarity identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const sim = cosineSimilarity(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.001);
}

test "cosineSimilarity orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sim, 0.001);
}

test "cosineSimilarity opposite vectors" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ -1.0, 0.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), sim, 0.001);
}

test "cosineSimilarity empty returns zero" {
    const empty: []const f32 = &.{};
    try std.testing.expectEqual(@as(f32, 0.0), cosineSimilarity(empty, empty));
}

test "batchCosineSimilarity" {
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const v1 = [_]f32{ 1.0, 0.0, 0.0 }; // identical
    const v2 = [_]f32{ 0.0, 1.0, 0.0 }; // orthogonal
    const vectors = [_][]const f32{ &v1, &v2 };
    var results: [2]f32 = undefined;
    batchCosineSimilarity(&query, &vectors, &results);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), results[1], 0.001);
}

test "vectorReduce sum" {
    const v = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const s = vectorReduce(.sum, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), s, 0.001);
}

test "vectorReduce max and min" {
    const v = [_]f32{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), vectorReduce(.max, &v), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vectorReduce(.min, &v), 0.001);
}

test "vectorReduce empty returns zero" {
    const empty: []const f32 = &.{};
    try std.testing.expectEqual(@as(f32, 0.0), vectorReduce(.sum, empty));
}

test "vectorAdd large array exercises SIMD path" {
    // Use array > VectorSize to guarantee SIMD loop runs
    var a: [33]f32 = undefined;
    var b: [33]f32 = undefined;
    var result: [33]f32 = undefined;
    for (&a, &b, 0..) |*av, *bv, i| {
        av.* = @floatFromInt(i);
        bv.* = @floatFromInt(i * 2);
    }
    vectorAdd(&a, &b, &result);
    // Spot-check first and last
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 96.0), result[32], 0.001); // 32 + 64
}

test "batchCosineSimilarityPrecomputed" {
    const query = [_]f32{ 1.0, 0.0 };
    const v1 = [_]f32{ 1.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0 };
    const vectors = [_][]const f32{ &v1, &v2 };
    const norms = [_]f32{ 1.0, 1.0 };
    var results: [2]f32 = undefined;
    batchCosineSimilarityPrecomputed(&query, 1.0, &vectors, &norms, &results);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), results[1], 0.001);
}

test "batchDotProduct" {
    const query = [_]f32{ 1.0, 2.0, 3.0 };
    const v1 = [_]f32{ 1.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0 };
    const vectors = [_][]const f32{ &v1, &v2 };
    var results: [2]f32 = undefined;
    batchDotProduct(&query, &vectors, &results);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), results[1], 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
