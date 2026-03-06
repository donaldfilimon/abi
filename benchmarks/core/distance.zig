//! Unified Distance Metrics for Benchmarks
//!
//! Provides consistent distance computation across all benchmark suites
//! by wrapping the framework's canonical SIMD implementations.

const std = @import("std");
const abi = @import("abi");
const Distance = abi.wdbx.Distance;

pub const Metric = enum {
    euclidean_sq,
    cosine,
    manhattan,
    dot_product,
};

/// Compute distance between two vectors using the specified metric.
/// Automatically uses SIMD optimizations from the framework.
pub fn compute(metric: Metric, a: []const f32, b: []const f32) f32 {
    return switch (metric) {
        .euclidean_sq => Distance.euclideanDistance(a, b),
        .cosine => Distance.cosineSimilarity(a, b),
        .manhattan => Distance.manhattanDistance(a, b),
        .dot_product => Distance.dotProduct(a, b),
    };
}

// For compatibility with existing benchmark code
pub const euclideanSq = Distance.euclideanDistance;
pub const cosine = Distance.cosineSimilarity;
pub const dot = Distance.dotProduct;
pub const manhattan = Distance.manhattanDistance;

test "distance metrics wrap canonical implementations" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 0, 1, 0 };

    // Euclidean squared distance between [1,0,0] and [0,1,0] is 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), compute(.euclidean_sq, &a, &b), 0.001);
}
