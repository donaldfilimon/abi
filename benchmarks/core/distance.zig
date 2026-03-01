//! Unified Distance Metrics for Benchmarks
//!
//! Provides consistent distance computation across all benchmark suites
//! with both scalar and SIMD implementations.
//!
//! ## Supported Metrics
//!
//! - **euclidean**: L2 distance (sqrt of sum of squared differences)
//! - **euclidean_sq**: Squared L2 distance (faster, preserves ordering)
//! - **cosine**: Cosine distance (1 - cosine similarity)
//! - **dot_product**: Negative dot product (for maximum inner product search)
//! - **manhattan**: L1 distance (sum of absolute differences)

const std = @import("std");

/// Distance metric types
pub const Metric = enum {
    /// Euclidean (L2) distance
    euclidean,
    /// Squared Euclidean distance (faster, preserves ordering)
    euclidean_sq,
    /// Cosine distance (1 - cosine_similarity)
    cosine,
    /// Negative dot product (for MIPS)
    dot_product,
    /// Manhattan (L1) distance
    manhattan,
};

/// Compute distance between two vectors using the specified metric
pub fn compute(comptime metric: Metric, a: []const f32, b: []const f32) f32 {
    return switch (metric) {
        .euclidean => euclidean(a, b),
        .euclidean_sq => euclideanSq(a, b),
        .cosine => cosine(a, b),
        .dot_product => dotProduct(a, b),
        .manhattan => manhattan(a, b),
    };
}

/// Compute distance using SIMD acceleration
pub fn computeSimd(comptime metric: Metric, comptime N: usize, a: []const f32, b: []const f32) f32 {
    return switch (metric) {
        .euclidean => simdEuclidean(N, a, b),
        .euclidean_sq => simdEuclideanSq(N, a, b),
        .cosine => simdCosine(N, a, b),
        .dot_product => simdDotProduct(N, a, b),
        .manhattan => simdManhattan(N, a, b),
    };
}

// ============================================================================
// Scalar Implementations
// ============================================================================

/// Euclidean (L2) distance
pub fn euclidean(a: []const f32, b: []const f32) f32 {
    return @sqrt(euclideanSq(a, b));
}

/// Squared Euclidean distance (faster, preserves ordering for k-NN)
pub fn euclideanSq(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| {
        const diff = x - y;
        sum += diff * diff;
    }
    return sum;
}

/// Cosine distance (1 - cosine similarity)
pub fn cosine(a: []const f32, b: []const f32) f32 {
    var dot_sum: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |x, y| {
        dot_sum += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 1.0;
    return 1.0 - (dot_sum / denom);
}

/// Cosine similarity (not distance)
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return 1.0 - cosine(a, b);
}

/// Negative dot product (for maximum inner product search)
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    var dot_sum: f32 = 0;
    for (a, b) |x, y| {
        dot_sum += x * y;
    }
    return -dot_sum; // Negative so smaller = better (like other distances)
}

/// Raw dot product (positive)
pub fn dot(a: []const f32, b: []const f32) f32 {
    var result: f32 = 0;
    for (a, b) |x, y| {
        result += x * y;
    }
    return result;
}

/// Manhattan (L1) distance
pub fn manhattan(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| {
        sum += @abs(x - y);
    }
    return sum;
}

// ============================================================================
// SIMD Implementations
// ============================================================================

/// SIMD Euclidean distance
fn simdEuclidean(comptime N: usize, a: []const f32, b: []const f32) f32 {
    return @sqrt(simdEuclideanSq(N, a, b));
}

/// SIMD Squared Euclidean distance
fn simdEuclideanSq(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: Vec = @splat(0);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        const diff = va - vb;
        sum += diff * diff;
    }

    // Horizontal sum
    var result: f32 = @reduce(.Add, sum);

    // Handle remainder
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

/// SIMD Cosine distance
fn simdCosine(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var dot_sum: Vec = @splat(0);
    var norm_a_sum: Vec = @splat(0);
    var norm_b_sum: Vec = @splat(0);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        dot_sum += va * vb;
        norm_a_sum += va * va;
        norm_b_sum += vb * vb;
    }

    var dot_result: f32 = @reduce(.Add, dot_sum);
    var norm_a: f32 = @reduce(.Add, norm_a_sum);
    var norm_b: f32 = @reduce(.Add, norm_b_sum);

    // Handle remainder
    while (i < a.len) : (i += 1) {
        dot_result += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 1.0;
    return 1.0 - (dot_result / denom);
}

/// SIMD Negative dot product
fn simdDotProduct(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: Vec = @splat(0);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        sum += va * vb;
    }

    var result: f32 = @reduce(.Add, sum);

    // Handle remainder
    while (i < a.len) : (i += 1) {
        result += a[i] * b[i];
    }

    return -result;
}

/// SIMD Manhattan distance
fn simdManhattan(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: Vec = @splat(0);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        const diff = va - vb;
        sum += @abs(diff);
    }

    var result: f32 = @reduce(.Add, sum);

    // Handle remainder
    while (i < a.len) : (i += 1) {
        result += @abs(a[i] - b[i]);
    }

    return result;
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Compute distances from a query to multiple vectors
pub fn computeBatch(
    comptime metric: Metric,
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    for (vectors, results) |vec, *dist| {
        dist.* = compute(metric, query, vec);
    }
}

/// Compute distances using SIMD for batch operations
pub fn computeBatchSimd(
    comptime metric: Metric,
    comptime N: usize,
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    for (vectors, results) |vec, *dist| {
        dist.* = computeSimd(metric, N, query, vec);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "euclidean distance" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };

    // Euclidean distance should be sqrt(2)
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(2.0)), euclidean(&a, &b), 0.001);

    // Squared should be 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), euclideanSq(&a, &b), 0.001);

    // Same vector should have 0 distance
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), euclideanSq(&a, &a), 0.001);
}

test "cosine distance" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    const c = [_]f32{ 1, 0, 0, 0 };

    // Orthogonal vectors should have cosine distance of 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosine(&a, &b), 0.001);

    // Same direction should have cosine distance of 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosine(&a, &c), 0.001);
}

test "dot product" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 1, 1, 1, 1 };

    // Dot product should be 1+2+3+4 = 10
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), dot(&a, &b), 0.001);

    // Distance version is negative
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), dotProduct(&a, &b), 0.001);
}

test "manhattan distance" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 0, 0, 0, 0 };

    // Manhattan distance should be |1|+|2|+|3|+|4| = 10
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), manhattan(&a, &b), 0.001);
}

test "simd euclidean" {
    const a = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0, 0, 1, 0, 0 };

    const scalar = euclideanSq(&a, &b);
    const simd = simdEuclideanSq(4, &a, &b);

    try std.testing.expectApproxEqAbs(scalar, simd, 0.001);
}

test "compute generic" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };

    const euc = compute(.euclidean_sq, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), euc, 0.001);

    const cos = compute(.cosine, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos, 0.001);
}
