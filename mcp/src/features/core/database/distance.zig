//! WDBX Distance & Similarity Metric Implementations
//!
//! All functions are pure (no allocations) and delegate inner-loop
//! arithmetic to the SIMD core for hardware acceleration.
//!
//! Metric semantics:
//! - **Cosine similarity**: range [-1, 1]; higher is more similar.
//! - **Euclidean distance**: range [0, ∞); lower is closer.
//! - **Dot product**: unbounded; higher is more similar (for normalised vectors).
//! - **Manhattan distance**: range [0, ∞); lower is closer.

const std = @import("std");
const simd = @import("simd.zig");

pub const Distance = struct {
    // ─────────────────────────────────────────────────────────────
    // Primary metrics
    // ─────────────────────────────────────────────────────────────

    /// Cosine similarity ∈ [-1, 1].
    ///
    /// Returns 0.0 when either vector has zero magnitude.
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        const dot = simd.dotProduct(f32, a, b);
        const na = simd.norm(f32, a);
        const nb = simd.norm(f32, b);
        return if (na > 0 and nb > 0) dot / (na * nb) else 0;
    }

    /// Euclidean (L2) distance ∈ [0, ∞).
    pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        return @sqrt(euclideanDistanceSq(a, b));
    }

    /// Squared Euclidean distance — avoids `sqrt` when only ordering matters.
    pub fn euclideanDistanceSq(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        // Build a temporary difference vector and compute its squared norm.
        var sum: f32 = 0;
        for (a, b) |av, bv| {
            const diff = av - bv;
            sum += diff * diff;
        }
        return sum;
    }

    /// Dot product (maximum inner product, MIP).
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        return simd.dotProduct(f32, a, b);
    }

    /// Manhattan (L1) distance ∈ [0, ∞).
    pub fn manhattanDistance(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var sum: f32 = 0;
        for (a, b) |av, bv| {
            sum += @abs(av - bv);
        }
        return sum;
    }

    // ─────────────────────────────────────────────────────────────
    // Chebyshev (L∞) — useful for grid / chessboard spaces
    // ─────────────────────────────────────────────────────────────

    /// Chebyshev (L∞) distance: max |aᵢ − bᵢ|.
    pub fn chebyshevDistance(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        var max_diff: f32 = 0;
        for (a, b) |av, bv| {
            const diff = @abs(av - bv);
            if (diff > max_diff) max_diff = diff;
        }
        return max_diff;
    }

    // ─────────────────────────────────────────────────────────────
    // Convenience: cosine *distance* (1 − similarity), ∈ [0, 2]
    // ─────────────────────────────────────────────────────────────

    /// Cosine distance = 1 − cosine_similarity. Always ∈ [0, 2].
    pub fn cosineDistance(a: []const f32, b: []const f32) f32 {
        return 1.0 - cosineSimilarity(a, b);
    }
};

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

test "Distance.cosineSimilarity identical vectors" {
    const a = [_]f32{ 1, 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), Distance.cosineSimilarity(&a, &a), 0.001);
}

test "Distance.cosineSimilarity orthogonal vectors" {
    const a = [_]f32{ 1, 0 };
    const b = [_]f32{ 0, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), Distance.cosineSimilarity(&a, &b), 0.001);
}

test "Distance.cosineSimilarity zero vector returns 0" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(f32, 0.0), Distance.cosineSimilarity(&a, &b));
}

test "Distance.euclideanDistance 3-4-5 triple" {
    const a = [_]f32{ 0, 0 };
    const b = [_]f32{ 3, 4 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), Distance.euclideanDistance(&a, &b), 0.001);
}

test "Distance.euclideanDistanceSq avoids sqrt" {
    const a = [_]f32{ 0, 0 };
    const b = [_]f32{ 3, 4 };
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), Distance.euclideanDistanceSq(&a, &b), 0.001);
}

test "Distance.dotProduct known result" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), Distance.dotProduct(&a, &b), 0.001);
}

test "Distance.manhattanDistance known result" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 1, 2, 3 };
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), Distance.manhattanDistance(&a, &b), 0.001);
}

test "Distance.chebyshevDistance returns max abs diff" {
    const a = [_]f32{ 0, 0, 0 };
    const b = [_]f32{ 1, 5, 3 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), Distance.chebyshevDistance(&a, &b), 0.001);
}

test "Distance.cosineDistance identical is 0" {
    const a = [_]f32{ 1, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), Distance.cosineDistance(&a, &a), 0.001);
}
