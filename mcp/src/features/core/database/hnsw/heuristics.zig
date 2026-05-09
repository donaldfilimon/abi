//! Shared HNSW heuristics and distance functions.
const std = @import("std");
const index_mod = @import("../index.zig");
const simd = @import("../../../../foundation/mod.zig").simd;

/// Select neighbors using heuristic pruning that considers both distance and diversity.
pub fn selectNeighborsHeuristic(
    norms: []const f32,
    allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    records: []const index_mod.VectorRecordView,
    node_id: u32,
    candidates: *std.AutoHashMapUnmanaged(u32, f32),
    m_val: usize,
) ![]u32 {
    const CandidatePair = struct { id: u32, dist: f32 };
    var sorted = std.ArrayListUnmanaged(CandidatePair).empty;
    defer sorted.deinit(temp_allocator);

    var it = candidates.iterator();
    while (it.next()) |entry_item| {
        if (entry_item.key_ptr.* != node_id) {
            try sorted.append(temp_allocator, .{ .id = entry_item.key_ptr.*, .dist = entry_item.value_ptr.* });
        }
    }

    std.sort.heap(CandidatePair, sorted.items, {}, struct {
        fn lessThan(_: void, a: CandidatePair, b: CandidatePair) bool {
            return a.dist < b.dist;
        }
    }.lessThan);

    var selected = std.ArrayListUnmanaged(u32).empty;
    errdefer selected.deinit(allocator);

    for (sorted.items, 0..) |candidate, idx| {
        if (selected.items.len >= m_val) break;

        if (idx + 1 < sorted.items.len) {
            const next_id = sorted.items[idx + 1].id;
            if (next_id < records.len) {
                @prefetch(records[next_id].vector.ptr, .{ .locality = 3, .rw = .read });
            }
        }

        var should_add = true;
        for (selected.items) |existing| {
            const dist_to_existing = computeDistance(norms, records, candidate.id, existing);
            if (dist_to_existing < candidate.dist) {
                should_add = false;
                break;
            }
        }

        if (should_add) {
            try selected.append(allocator, candidate.id);
        }
    }

    if (selected.items.len < m_val) {
        for (sorted.items) |candidate| {
            if (selected.items.len >= m_val) break;

            var already_added = false;
            for (selected.items) |existing| {
                if (existing == candidate.id) {
                    already_added = true;
                    break;
                }
            }

            if (!already_added) {
                try selected.append(allocator, candidate.id);
            }
        }
    }

    return selected.toOwnedSlice(allocator);
}

/// Compute distance between two nodes.
pub fn computeDistance(
    norms: []const f32,
    records: []const index_mod.VectorRecordView,
    a: u32,
    b: u32,
) f32 {
    if (norms.len > a and norms.len > b) {
        const na = norms[a];
        const nb = norms[b];
        if (na > 0.0 and nb > 0.0) {
            const dot = simd.vectorDot(records[a].vector, records[b].vector);
            return 1.0 - dot / (na * nb);
        }
    }
    return 1.0 - simd.cosineSimilarity(records[a].vector, records[b].vector);
}
