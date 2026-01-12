//! Hierarchical Navigable Small World (HNSW) vector index implementation.
//! Provides efficient approximate nearest neighbor search in high-dimensional spaces.

const std = @import("std");
const simd = @import("../../shared/simd.zig");
const index_mod = @import("index.zig");

/// HNSW index structure supporting layered graph traversal.
pub const HnswIndex = struct {
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    entry_point: ?u32,
    max_layer: i32,
    nodes: []NodeLayers,

    pub const NodeLayers = struct {
        layers: []index_mod.NeighborList,
    };

    /// Build a new HNSW index from a set of records.
    /// @param allocator Memory allocator for graph structures
    /// @param records Source vector records
    /// @param m Max number of connections per node
    /// @param ef_construction Size of the dynamic candidate list during construction
    /// @return Initialized HnswIndex
    pub fn build(
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        m: usize,
        ef_construction: usize,
    ) !HnswIndex {
        if (records.len == 0) return index_mod.IndexError.EmptyIndex;

        var self = HnswIndex{
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .ef_construction = ef_construction,
            .entry_point = null,
            .max_layer = -1,
            .nodes = try allocator.alloc(NodeLayers, records.len),
        };
        errdefer allocator.free(self.nodes);

        for (self.nodes) |*node| {
            node.layers = &.{};
        }

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const m_l = 1.0 / @as(f32, @log(@as(f32, @floatFromInt(m))));

        for (records, 0..) |_, i| {
            try self.insert(allocator, records, @intCast(i), random, m_l);
        }

        return self;
    }

    /// Insert a new node into the HNSW graph.
    fn insert(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        random: std.Random,
        m_l: f32,
    ) !void {
        const target_layer = @as(i32, @intFromFloat(@floor(-@log(random.float(f32)) * m_l)));
        self.nodes[node_id].layers = try allocator.alloc(index_mod.NeighborList, @intCast(target_layer + 1));
        for (self.nodes[node_id].layers) |*list| {
            list.nodes = &.{};
        }

        if (self.entry_point == null) {
            self.entry_point = node_id;
            self.max_layer = target_layer;
            return;
        }

        var curr_node = self.entry_point.?;
        var curr_dist = 1.0 - simd.cosineSimilarity(records[node_id].vector, records[curr_node].vector);

        // 1. Greedy search down to layer above target_layer
        var lc: i32 = self.max_layer;
        while (lc > target_layer) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                for (self.nodes[curr_node].layers[@intCast(lc)].nodes) |neighbor| {
                    const d = 1.0 - simd.cosineSimilarity(records[node_id].vector, records[neighbor].vector);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_node = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Perform layered insertion from target_layer down to 0
        lc = @min(target_layer, self.max_layer);
        while (lc >= 0) : (lc -= 1) {
            try self.connectNeighbors(allocator, records, node_id, curr_node, @intCast(lc));
        }

        // 3. Update global entry point if new node is at a higher layer
        if (target_layer > self.max_layer) {
            self.max_layer = target_layer;
            self.entry_point = node_id;
        }
    }

    /// Connect a node to its neighbors at a specific layer using proper HNSW neighbor selection.
    fn connectNeighbors(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        entry: u32,
        layer: usize,
    ) !void {
        const m_val = if (layer == 0) self.m_max0 else self.m_max;

        // Build candidate list using ef_construction expansion
        var candidates = std.AutoHashMapUnmanaged(u32, f32){};
        defer candidates.deinit(allocator);

        var visited = std.AutoHashMapUnmanaged(u32, void){};
        defer visited.deinit(allocator);

        // Start with entry point
        const entry_dist = 1.0 - simd.cosineSimilarity(records[node_id].vector, records[entry].vector);
        try candidates.put(allocator, entry, entry_dist);
        try visited.put(allocator, entry, {});

        // BFS expansion to find candidates
        var queue = std.ArrayListUnmanaged(u32){};
        defer queue.deinit(allocator);
        try queue.append(allocator, entry);

        var head: usize = 0;
        while (head < queue.items.len and candidates.count() < self.ef_construction) : (head += 1) {
            const curr = queue.items[head];
            if (layer < self.nodes[curr].layers.len) {
                for (self.nodes[curr].layers[layer].nodes) |neighbor| {
                    if (!visited.contains(neighbor)) {
                        try visited.put(allocator, neighbor, {});
                        const dist = 1.0 - simd.cosineSimilarity(records[node_id].vector, records[neighbor].vector);
                        try candidates.put(allocator, neighbor, dist);
                        try queue.append(allocator, neighbor);
                    }
                }
            }
        }

        // Select best neighbors using heuristic pruning
        const selected = try self.selectNeighborsHeuristic(allocator, records, node_id, &candidates, m_val);
        self.nodes[node_id].layers[layer].nodes = selected;

        // Update bidirectional links with proper pruning
        for (self.nodes[node_id].layers[layer].nodes) |neighbor| {
            if (layer >= self.nodes[neighbor].layers.len) continue;

            var neighbor_links = std.AutoHashMapUnmanaged(u32, f32){};
            defer neighbor_links.deinit(allocator);

            // Collect existing neighbors
            for (self.nodes[neighbor].layers[layer].nodes) |existing| {
                const dist = 1.0 - simd.cosineSimilarity(records[neighbor].vector, records[existing].vector);
                try neighbor_links.put(allocator, existing, dist);
            }

            // Add new link if not exists
            if (!neighbor_links.contains(node_id)) {
                const dist = 1.0 - simd.cosineSimilarity(records[neighbor].vector, records[node_id].vector);
                try neighbor_links.put(allocator, node_id, dist);
            }

            // Prune if needed
            if (neighbor_links.count() > m_val) {
                const pruned = try self.selectNeighborsHeuristic(allocator, records, neighbor, &neighbor_links, m_val);
                allocator.free(self.nodes[neighbor].layers[layer].nodes);
                self.nodes[neighbor].layers[layer].nodes = pruned;
            } else {
                // Just update with new links
                var new_links = std.ArrayListUnmanaged(u32){};
                errdefer new_links.deinit(allocator);

                var it = neighbor_links.keyIterator();
                while (it.next()) |key| {
                    try new_links.append(allocator, key.*);
                }

                allocator.free(self.nodes[neighbor].layers[layer].nodes);
                self.nodes[neighbor].layers[layer].nodes = try new_links.toOwnedSlice(allocator);
            }
        }
    }

    /// Select neighbors using heuristic pruning that considers both distance and diversity.
    fn selectNeighborsHeuristic(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        node_id: u32,
        candidates: *std.AutoHashMapUnmanaged(u32, f32),
        m_val: usize,
    ) ![]u32 {
        _ = self;

        // Sort candidates by distance (ascending)
        const CandidatePair = struct { id: u32, dist: f32 };
        var sorted = std.ArrayListUnmanaged(CandidatePair){};
        defer sorted.deinit(allocator);

        var it = candidates.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.* != node_id) { // Don't include self
                try sorted.append(allocator, .{ .id = entry.key_ptr.*, .dist = entry.value_ptr.* });
            }
        }

        // Sort by distance (closest first)
        std.sort.heap(CandidatePair, sorted.items, {}, struct {
            fn lessThan(_: void, a: CandidatePair, b: CandidatePair) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Select using heuristic: prefer diverse neighbors over purely closest
        var selected = std.ArrayListUnmanaged(u32){};
        errdefer selected.deinit(allocator);

        for (sorted.items) |candidate| {
            if (selected.items.len >= m_val) break;

            // Check if this candidate is closer to node than to any selected neighbor
            var should_add = true;
            for (selected.items) |existing| {
                const dist_to_existing = 1.0 - simd.cosineSimilarity(
                    records[candidate.id].vector,
                    records[existing].vector,
                );
                // If candidate is closer to an existing neighbor than to the node,
                // skip it to maintain diversity
                if (dist_to_existing < candidate.dist) {
                    should_add = false;
                    break;
                }
            }

            if (should_add) {
                try selected.append(allocator, candidate.id);
            }
        }

        // If we don't have enough neighbors due to heuristic, fill with closest
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

    /// Search the HNSW graph for the nearest neighbors of a query vector.
    /// @param allocator Memory allocator for search results
    /// @param records Source vector records
    /// @param query Query vector
    /// @param top_k Number of results to return
    /// @return Slice of IndexResult sorted by similarity
    pub fn search(
        self: *const HnswIndex,
        allocator: std.mem.Allocator,
        records: []const index_mod.VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]index_mod.IndexResult {
        if (self.entry_point == null or records.len == 0) {
            return allocator.alloc(index_mod.IndexResult, 0);
        }

        var curr_node = self.entry_point.?;
        var curr_dist = 1.0 - simd.cosineSimilarity(query, records[curr_node].vector);

        // 1. Zoom in through layers
        var lc: i32 = self.max_layer;
        while (lc > 0) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                for (self.nodes[curr_node].layers[@intCast(lc)].nodes) |neighbor| {
                    const d = 1.0 - simd.cosineSimilarity(query, records[neighbor].vector);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_node = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Local search on layer 0 with candidate accumulation
        var candidates = std.AutoHashMapUnmanaged(u32, f32){};
        defer candidates.deinit(allocator);
        try candidates.put(allocator, curr_node, curr_dist);

        var queue = std.ArrayListUnmanaged(u32){};
        defer queue.deinit(allocator);
        try queue.append(allocator, curr_node);

        var head: usize = 0;
        // Limit search expansion for performance
        const max_candidates = @max(top_k * 2, self.ef_construction / 2);

        while (head < queue.items.len and queue.items.len < max_candidates) : (head += 1) {
            const u = queue.items[head];
            for (self.nodes[u].layers[0].nodes) |v| {
                if (!candidates.contains(v)) {
                    const d = 1.0 - simd.cosineSimilarity(query, records[v].vector);
                    try candidates.put(allocator, v, d);
                    try queue.append(allocator, v);
                }
            }
        }

        // 3. Extract and sort top-k results
        var results = try allocator.alloc(index_mod.IndexResult, candidates.count());
        var it = candidates.iterator();
        var i: usize = 0;
        while (it.next()) |entry| {
            results[i] = .{
                .id = records[entry.key_ptr.*].id,
                .score = 1.0 - entry.value_ptr.*, // Convert distance back to similarity
            };
            i += 1;
        }

        std.sort.heap(index_mod.IndexResult, results, {}, struct {
            fn lessThan(_: void, a: index_mod.IndexResult, b: index_mod.IndexResult) bool {
                return a.score > b.score;
            }
        }.lessThan);

        if (results.len > top_k) {
            const final = try allocator.dupe(index_mod.IndexResult, results[0..top_k]);
            allocator.free(results);
            return final;
        }
        return results;
    }

    /// Free resources associated with the index.
    pub fn deinit(self: *HnswIndex, allocator: std.mem.Allocator) void {
        for (self.nodes) |node| {
            for (node.layers) |list| {
                allocator.free(list.nodes);
            }
            allocator.free(node.layers);
        }
        allocator.free(self.nodes);
        self.* = undefined;
    }

    /// Save HNSW structure to a binary writer.
    pub fn save(self: HnswIndex, writer: anytype) !void {
        try writer.writeInt(u32, @intCast(self.nodes.len), .little);
        try writer.writeInt(u32, @intCast(self.m), .little);
        try writer.writeInt(u32, if (self.entry_point) |ep| ep else 0, .little);
        try writer.writeInt(i32, self.max_layer, .little);
        try writer.writeInt(u32, @intCast(self.ef_construction), .little);

        for (self.nodes) |node| {
            try writer.writeInt(u32, @intCast(node.layers.len), .little);
            for (node.layers) |list| {
                try writer.writeInt(u32, @intCast(list.nodes.len), .little);
                for (list.nodes) |neighbor| {
                    try writer.writeInt(u32, neighbor, .little);
                }
            }
        }
    }

    /// Load HNSW structure from a binary reader.
    pub fn load(allocator: std.mem.Allocator, reader: anytype) !HnswIndex {
        const node_count = try reader.readInt(u32, .little);
        const m = try reader.readInt(u32, .little);
        const entry_point = try reader.readInt(u32, .little);
        const max_layer = try reader.readInt(i32, .little);
        const ef_construction = try reader.readInt(u32, .little);

        var self = HnswIndex{
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .ef_construction = ef_construction,
            .entry_point = if (node_count > 0) entry_point else null,
            .max_layer = max_layer,
            .nodes = try allocator.alloc(NodeLayers, node_count),
        };
        errdefer allocator.free(self.nodes);

        for (self.nodes) |*node| {
            const layer_count = try reader.readInt(u32, .little);
            node.layers = try allocator.alloc(index_mod.NeighborList, layer_count);
            for (node.layers) |*list| {
                const neighbor_count = try reader.readInt(u32, .little);
                list.nodes = try allocator.alloc(u32, neighbor_count);
                for (list.nodes) |*neighbor| {
                    neighbor.* = try reader.readInt(u32, .little);
                }
            }
        }

        return self;
    }
};

test "hnsw structure basic lifecycle" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.7, 0.7 } },
    };

    var index = try HnswIndex.build(allocator, &records, 16, 100);
    defer index.deinit(allocator);

    try std.testing.expect(index.nodes.len == 3);
    try std.testing.expect(index.entry_point != null);

    const query = [_]f32{ 0.8, 0.6 };
    const results = try index.search(allocator, &records, &query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len <= 2);
    if (results.len > 0) {
        // Result 3 should be top since similarity is high
        try std.testing.expect(results[0].id == 3);
    }
}
