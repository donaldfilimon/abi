//! HNSW search implementation helpers.
//!
//! Standalone functions for search, batch search, and distance computation
//! that are called by HnswIndex methods.

const std = @import("std");
const index_mod = @import("../index.zig");
const simd = @import("../../../foundation/mod.zig").simd;
const search_state = @import("../search_state.zig");
const gpu_accel = @import("../gpu_accel.zig");
const search_types = @import("search.zig");

pub const SearchState = search_state.SearchState;
pub const BatchSearchResult = search_types.BatchSearchResult;

/// Sequential distance computation using SIMD with prefetching.
pub fn computeBatchDistancesSequential(
    query: []const f32,
    query_norm: f32,
    records: []const index_mod.VectorRecordView,
    neighbor_ids: []const u32,
    distances: []f32,
) void {
    // Prefetch first few vectors to warm the cache
    const prefetch_ahead: usize = 4;
    for (0..@min(prefetch_ahead, neighbor_ids.len)) |ni| {
        const id = neighbor_ids[ni];
        if (id < records.len) {
            @prefetch(records[id].vector.ptr, .{
                .locality = 2,
                .rw = .read,
                .cache = .data,
            });
        }
    }

    for (neighbor_ids, 0..) |id, ni| {
        // Prefetch ahead for upcoming iterations
        if (ni + prefetch_ahead < neighbor_ids.len) {
            const future_id = neighbor_ids[ni + prefetch_ahead];
            if (future_id < records.len) {
                @prefetch(records[future_id].vector.ptr, .{
                    .locality = 2,
                    .rw = .read,
                    .cache = .data,
                });
            }
        }

        if (id < records.len) {
            const vec = records[id].vector;
            const vec_norm = simd.vectorL2Norm(vec);
            if (vec_norm > 0 and query_norm > 0) {
                const dot = simd.vectorDot(query, vec);
                distances[ni] = 1.0 - (dot / (query_norm * vec_norm));
            } else {
                distances[ni] = 1.0;
            }
        } else {
            distances[ni] = 1.0;
        }
    }
}

/// Process neighbors sequentially using SIMD (fallback when GPU batch path fails).
pub fn searchNeighborsSequential(
    allocator: std.mem.Allocator,
    neighbors: []const u32,
    records: []const index_mod.VectorRecordView,
    query: []const f32,
    query_norm: f32,
    state: *SearchState,
) !void {
    for (neighbors) |v| {
        if (!state.visited.contains(v) and v < records.len) {
            @prefetch(records[v].vector.ptr, .{ .locality = 2, .rw = .read });
        }
    }
    for (neighbors) |v| {
        if (!state.visited.contains(v)) {
            try state.visited.put(allocator, v, {});
            const vec_norm = simd.vectorL2Norm(records[v].vector);
            const d = if (vec_norm == 0.0) 1.0 else blk: {
                const dot = simd.vectorDot(query, records[v].vector);
                break :blk 1.0 - (dot / (query_norm * vec_norm));
            };
            try state.candidates.put(allocator, v, d);
            try state.queue.append(allocator, v);
        }
    }
}

/// Compute cosine distance between query and vector using SIMD.
pub inline fn computeDistance(query: []const f32, query_norm: f32, vector: []const f32) f32 {
    const vec_norm = simd.vectorL2Norm(vector);
    if (vec_norm == 0.0) return 1.0;

    const dot = simd.vectorDot(query, vector);
    return 1.0 - (dot / (query_norm * vec_norm));
}
