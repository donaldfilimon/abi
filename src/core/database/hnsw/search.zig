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
