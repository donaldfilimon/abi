//! Hierarchical Navigable Small World (HNSW) vector index implementation.
//! Provides efficient approximate nearest neighbor search in high-dimensional spaces.
//!
//! Performance optimizations:
//! - SearchStatePool: Pre-allocated search states to avoid allocation per query
//! - DistanceCache: LRU cache for frequently computed distances
//! - Prefetching: Memory prefetch hints for graph traversal
//! - Vectorized distance computation via SIMD
//! - GPU-accelerated batch distance computation for large neighbor sets

const std = @import("std");
const build_options = @import("build_options");
const simd = @import("../../../foundation/mod.zig").simd;
const index_mod = @import("../index.zig");
const gpu_accel = @import("../gpu_accel.zig");

// Re-export extracted sub-modules
pub const search_state = @import("../search_state.zig");
pub const distance_cache = @import("../distance_cache.zig");
pub const search_types = @import("search.zig");
pub const persistence = @import("persistence.zig");
pub const insert_impl = @import("insert.zig");
pub const search_impl = @import("search_impl.zig");

pub const SearchState = search_state.SearchState;
pub const SearchStatePool = search_state.SearchStatePool;
pub const DistanceCache = distance_cache.DistanceCache;

// Re-export types from search sub-module
pub const AdaptiveSearchConfig = search_types.AdaptiveSearchConfig;
pub const IndexStats = search_types.IndexStats;

pub const index_impl = @import("index.zig");
pub const HnswIndex = index_impl.HnswIndex;

// Test discovery: pull in extracted test file and sub-modules
comptime {
    if (@import("builtin").is_test) {
        _ = @import("../hnsw_test.zig");
        _ = index_impl;
        _ = search_state;
        _ = distance_cache;
        _ = search_types;
        _ = persistence;
        _ = insert_impl;
        _ = search_impl;
    }
}

test {
    std.testing.refAllDecls(@This());
}
