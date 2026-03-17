//! Integration Tests — Cross-module verification of the ABI framework
//!
//! These tests verify that key subsystems are reachable and internally
//! consistent through the public `abi` package interface.

const std = @import("std");
const abi = @import("abi");

// ── Database subsystem ──────────────────────────────────────────────────────

test "database types are accessible" {
    // Compile-time check that core database types resolve through the public API.
    comptime {
        std.debug.assert(@hasDecl(abi.database, "DatabaseConfig"));
        std.debug.assert(@hasDecl(abi.database, "DatabaseHandle"));
        std.debug.assert(@hasDecl(abi.database, "SearchResult"));
    }
}

// ── Inference subsystem ─────────────────────────────────────────────────────

test "inference engine types are accessible" {
    // Compile-time check that inference types resolve through the public API.
    comptime {
        std.debug.assert(@hasDecl(abi.inference, "Engine"));
        std.debug.assert(@hasDecl(abi.inference, "EngineConfig"));
        std.debug.assert(@hasDecl(abi.inference, "FinishReason"));
    }
}

test "inference finish reason has expected variants" {
    // Verify the FinishReason enum contains the stop variant.
    const reason: abi.inference.FinishReason = .stop;
    try std.testing.expect(reason == .stop);
}

// ── Distance functions ──────────────────────────────────────────────────────

test "simd distance functions are consistent" {
    const Distance = abi.database.distance.Distance;

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    // Cosine similarity of a vector with itself should be 1.
    const self_sim = Distance.cosineSimilarity(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), self_sim, 1e-5);

    // Euclidean distance of a vector with itself should be 0.
    const self_dist = Distance.euclideanDistanceSq(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), self_dist, 1e-5);

    // Dot product should be commutative.
    const dp_ab = Distance.dotProduct(&a, &b);
    const dp_ba = Distance.dotProduct(&b, &a);
    try std.testing.expectApproxEqAbs(dp_ab, dp_ba, 1e-5);
}

// ── Framework metadata ──────────────────────────────────────────────────────

test "framework version is available" {
    const ver = abi.meta.version();
    try std.testing.expect(ver.len > 0);
}

test "feature catalog enumerates features" {
    const features = abi.feature_catalog.all_features;
    try std.testing.expect(features.len > 0);
}
