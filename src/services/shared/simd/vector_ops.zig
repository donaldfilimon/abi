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

        // Use @reduce for efficient horizontal reduction (Zig 0.16+)
        switch (op) {
            .sum => result += @reduce(.Add, vec_result),
            .max => result = @max(result, @reduce(.Max, vec_result)),
            .min => result = @min(result, @reduce(.Min, vec_result)),
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
