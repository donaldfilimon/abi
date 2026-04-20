//! HNSW insert algorithms: node insertion, neighbor connection, and heuristic pruning.
//!
//! These are standalone helper functions called by HnswIndex methods.
//! They receive individual struct fields rather than `*HnswIndex` so that
//! the module can be compiled independently without circular imports.

const std = @import("std");
const index_mod = @import("../index.zig");
const simd = @import("../../../../foundation/mod.zig").simd;
const distance_cache_mod = @import("../distance_cache.zig");

pub const DistanceCache = distance_cache_mod.DistanceCache;

/// NodeLayers mirrors the HnswIndex.NodeLayers type (avoids circular import with mod.zig).
pub const NodeLayers = struct {
    layers: []index_mod.NeighborList,
};

/// Parameters shared across insert helper functions (avoids passing many args repeatedly).
pub const InsertContext = struct {
    nodes: []NodeLayers,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    norms: []const f32,
    distance_cache: ?*DistanceCache,
};

/// Insert a new node at a specific position in the HNSW graph.
///
/// Implements the standard HNSW insertion algorithm:
/// 1. Determine target layer using exponential distribution
/// 2. Greedy descent from max_layer to target_layer + 1
/// 3. Layer-by-layer neighbor connection from target_layer to 0
/// 4. Update entry point if new node is at a higher layer
pub fn insertAt(
    ctx: InsertContext,
    entry_point: *?u32,
    max_layer: *i32,
    allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    records: []const index_mod.VectorRecordView,
    node_id: u32,
    random: std.Random,
    m_l: f32,
) !void {
    const target_layer = @as(i32, @intFromFloat(@floor(-@log(random.float(f32)) * m_l)));
    ctx.nodes[node_id].layers = try allocator.alloc(index_mod.NeighborList, @intCast(target_layer + 1));
    for (ctx.nodes[node_id].layers) |*list| {
        list.nodes = &.{};
    }

    if (entry_point.* == null) {
        entry_point.* = node_id;
        max_layer.* = target_layer;
        return;
    }

    var curr_node = entry_point.*.?;
    var curr_dist = computeNodeDistance(ctx, records, node_id, curr_node);

    // 1. Greedy search down to layer above target_layer
    var lc: i32 = max_layer.*;
    while (lc > target_layer) : (lc -= 1) {
        var changed = true;
        while (changed) {
            changed = false;
            const neighbors = ctx.nodes[curr_node].layers[@intCast(lc)].nodes;

            for (neighbors) |neighbor| {
                if (neighbor < records.len) {
                    @prefetch(records[neighbor].vector.ptr, .{ .locality = 3, .rw = .read });
                }
            }

            for (neighbors) |neighbor| {
                const d = computeNodeDistance(ctx, records, node_id, neighbor);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr_node = neighbor;
                    changed = true;
                }
            }
        }
    }

    // 2. Perform layered insertion from target_layer down to 0
    lc = @min(target_layer, max_layer.*);
    while (lc >= 0) : (lc -= 1) {
        try connectNeighbors(ctx, allocator, temp_allocator, records, node_id, curr_node, @intCast(lc));
    }

    // 3. Update global entry point if new node is at a higher layer
    if (target_layer > max_layer.*) {
        max_layer.* = target_layer;
        entry_point.* = node_id;
    }
}

/// Connect a node to its neighbors at a specific layer using heuristic pruning.
fn connectNeighbors(
    ctx: InsertContext,
    allocator: std.mem.Allocator,
    temp_allocator: std.mem.Allocator,
    records: []const index_mod.VectorRecordView,
    node_id: u32,
    entry: u32,
    layer: usize,
) !void {
    const m_val = if (layer == 0) ctx.m_max0 else ctx.m_max;

    // Build candidate list using ef_construction expansion
    var candidates = std.AutoHashMapUnmanaged(u32, f32).empty;
    defer candidates.deinit(temp_allocator);

    var visited = std.AutoHashMapUnmanaged(u32, void).empty;
    defer visited.deinit(temp_allocator);

    const entry_dist = computeNodeDistance(ctx, records, node_id, entry);
    try candidates.put(temp_allocator, entry, entry_dist);
    try visited.put(temp_allocator, entry, {});

    // BFS expansion to find candidates
    var queue = std.ArrayListUnmanaged(u32).empty;
    defer queue.deinit(temp_allocator);
    try queue.append(temp_allocator, entry);

    var head: usize = 0;
    while (head < queue.items.len and candidates.count() < ctx.ef_construction) : (head += 1) {
        const curr = queue.items[head];
        if (layer < ctx.nodes[curr].layers.len) {
            const neighbors = ctx.nodes[curr].layers[layer].nodes;

            for (neighbors) |neighbor| {
                if (!visited.contains(neighbor) and neighbor < records.len) {
                    @prefetch(records[neighbor].vector.ptr, .{ .locality = 2, .rw = .read });
                }
            }

            for (neighbors) |neighbor| {
                if (!visited.contains(neighbor)) {
                    try visited.put(temp_allocator, neighbor, {});
                    const dist = computeNodeDistance(ctx, records, node_id, neighbor);
                    try candidates.put(temp_allocator, neighbor, dist);
                    try queue.append(temp_allocator, neighbor);
                }
            }
        }
    }

    // Select best neighbors using heuristic pruning
    const selected = try selectNeighborsHeuristic(ctx.norms, allocator, temp_allocator, records, node_id, &candidates, m_val);
    ctx.nodes[node_id].layers[layer].nodes = selected;

    // Update bidirectional links with proper pruning
    const node_neighbors = ctx.nodes[node_id].layers[layer].nodes;
    for (node_neighbors, 0..) |neighbor, neighbor_idx| {
        if (layer >= ctx.nodes[neighbor].layers.len) continue;

        if (neighbor_idx + 1 < node_neighbors.len) {
            const next_neighbor = node_neighbors[neighbor_idx + 1];
            if (next_neighbor < ctx.nodes.len and ctx.nodes[next_neighbor].layers.len > layer) {
                @prefetch(ctx.nodes[next_neighbor].layers[layer].nodes.ptr, .{
                    .locality = 2,
                    .rw = .read,
                    .cache = .data,
                });
            }
        }

        var neighbor_links = std.AutoHashMapUnmanaged(u32, f32).empty;
        defer neighbor_links.deinit(temp_allocator);

        const existing_neighbors = ctx.nodes[neighbor].layers[layer].nodes;

        for (existing_neighbors) |existing| {
            if (existing < records.len) {
                @prefetch(records[existing].vector.ptr, .{
                    .locality = 2,
                    .rw = .read,
                    .cache = .data,
                });
            }
        }

        for (existing_neighbors) |existing| {
            const dist = computeNodeDistance(ctx, records, neighbor, existing);
            try neighbor_links.put(temp_allocator, existing, dist);
        }

        if (!neighbor_links.contains(node_id)) {
            const dist = computeNodeDistance(ctx, records, neighbor, node_id);
            try neighbor_links.put(temp_allocator, node_id, dist);
        }

        if (neighbor_links.count() > m_val) {
            const pruned = try selectNeighborsHeuristic(ctx.norms, allocator, temp_allocator, records, neighbor, &neighbor_links, m_val);
            allocator.free(ctx.nodes[neighbor].layers[layer].nodes);
            ctx.nodes[neighbor].layers[layer].nodes = pruned;
        } else {
            var new_links = std.ArrayListUnmanaged(u32).empty;
            errdefer new_links.deinit(allocator);

            var it = neighbor_links.keyIterator();
            while (it.next()) |key| {
                try new_links.append(allocator, key.*);
            }

            allocator.free(ctx.nodes[neighbor].layers[layer].nodes);
            ctx.nodes[neighbor].layers[layer].nodes = try new_links.toOwnedSlice(allocator);
        }
    }
}

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
                @prefetch(records[next_id].vector.ptr, .{
                    .locality = 3,
                    .rw = .read,
                    .cache = .data,
                });
            }
        }

        var should_add = true;
        for (selected.items) |existing| {
            const dist_to_existing = if (norms.len > candidate.id and norms.len > existing) blk: {
                const na = norms[candidate.id];
                const nb = norms[existing];
                if (na > 0.0 and nb > 0.0) {
                    const dot = simd.vectorDot(
                        records[candidate.id].vector,
                        records[existing].vector,
                    );
                    break :blk 1.0 - dot / (na * nb);
                }
                break :blk 1.0;
            } else 1.0 - simd.cosineSimilarity(
                records[candidate.id].vector,
                records[existing].vector,
            );
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

/// Compute node-to-node distance with optional caching.
pub fn computeNodeDistance(
    ctx: InsertContext,
    records: []const index_mod.VectorRecordView,
    a: u32,
    b: u32,
) f32 {
    // Check cache first
    if (ctx.distance_cache) |cache| {
        if (cache.get(a, b)) |cached| {
            return cached;
        }
    }

    // Compute cosine distance using pre-computed norms when available
    const dist = if (ctx.norms.len > a and ctx.norms.len > b) blk: {
        const na = ctx.norms[a];
        const nb = ctx.norms[b];
        if (na > 0.0 and nb > 0.0) {
            const dot = simd.vectorDot(records[a].vector, records[b].vector);
            break :blk 1.0 - dot / (na * nb);
        }
        break :blk 1.0;
    } else 1.0 - simd.cosineSimilarity(records[a].vector, records[b].vector);

    // Store in cache
    if (ctx.distance_cache) |cache| {
        cache.put(a, b, dist);
    }

    return dist;
}

test {
    std.testing.refAllDecls(@This());
}
