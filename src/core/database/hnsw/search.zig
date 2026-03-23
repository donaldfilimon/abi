//! HNSW Batch Search Helpers
//!
//! Types and free functions supporting parallel batch search operations.

const std = @import("std");
const index_mod = @import("../index.zig");

/// Configuration for adaptive search that auto-tunes ef for target recall.
pub const AdaptiveSearchConfig = struct {
    initial_ef: u32 = 50,
    max_ef: u32 = 500,
    ef_step: u32 = 50,
    target_recall: f32 = 0.95,
    max_iterations: u32 = 5,

    /// Compute an ef value adapted to dataset size.
    /// Uses heuristic: ef = min(50 + 2 * sqrt(n), 500)
    /// Larger datasets need higher ef for comparable recall.
    pub fn adaptiveEfForSize(self: AdaptiveSearchConfig, dataset_size: usize) u32 {
        const n_f: f64 = @floatFromInt(dataset_size);
        const sqrt_n: usize = @intFromFloat(@sqrt(n_f));
        const ef = @min(50 + 2 * sqrt_n, self.max_ef);
        return @intCast(@max(ef, self.initial_ef));
    }
};

/// Statistics about the HNSW index structure.
pub const IndexStats = struct {
    num_vectors: usize,
    num_layers: u32,
    avg_degree: f32,
    memory_bytes: usize,
    entry_point_id: ?u32,
};

/// Result type for batch search operations
pub const BatchSearchResult = struct {
    /// Query index in the original batch
    query_index: usize,
    /// Search results for this query
    results: []index_mod.IndexResult,
};

test "AdaptiveSearchConfig defaults" {
    const config = AdaptiveSearchConfig{};
    try std.testing.expectEqual(@as(u32, 50), config.initial_ef);
    try std.testing.expectEqual(@as(u32, 500), config.max_ef);
    try std.testing.expectEqual(@as(u32, 50), config.ef_step);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), config.target_recall, 0.001);
    try std.testing.expectEqual(@as(u32, 5), config.max_iterations);
}

test "AdaptiveSearchConfig adaptiveEfForSize" {
    const config = AdaptiveSearchConfig{};

    // Empty dataset: ef should be initial_ef (50)
    try std.testing.expectEqual(@as(u32, 50), config.adaptiveEfForSize(0));

    // Tiny dataset (1): ef = max(50 + 2*1, 50) = 52
    try std.testing.expectEqual(@as(u32, 52), config.adaptiveEfForSize(1));

    // Medium dataset (10000): ef = min(50 + 2*100, 500) = 250
    try std.testing.expectEqual(@as(u32, 250), config.adaptiveEfForSize(10_000));

    // Large dataset (62500): ef = min(50 + 2*250, 500) = 500 (capped)
    try std.testing.expectEqual(@as(u32, 500), config.adaptiveEfForSize(62_500));

    // Very large dataset: ef should be capped at max_ef
    try std.testing.expectEqual(@as(u32, 500), config.adaptiveEfForSize(1_000_000));
}
