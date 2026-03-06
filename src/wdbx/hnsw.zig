//! Hierarchical Navigable Small World (HNSW) Index
//!
//! Multi-layer proximity graph for approximate nearest neighbor search.
//! Each node exists on layers 0..max_level. Layer 0 contains all nodes;
//! higher layers are progressively sparser, enabling logarithmic search.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (2018).

const std = @import("std");
const config = @import("config.zig");
const metrics = @import("distance.zig").Distance;

pub const HNSW = struct {
    allocator: std.mem.Allocator,
    cfg: config.Config.IndexConfig,
    metric: config.DistanceMetric,

    /// All stored vectors (index = node ID).
    vectors: std.ArrayListUnmanaged([]const f32) = .empty,

    /// Per-node neighbor lists, one per layer the node exists on.
    /// neighbors[node_id][layer] = slice of neighbor node IDs.
    neighbors: std.ArrayListUnmanaged([][]u32) = .empty,

    /// The level each node was assigned to (0-based max).
    node_levels: std.ArrayListUnmanaged(u32) = .empty,

    /// Entry point node ID (the node with the highest level).
    entry_point: ?u32 = null,

    /// Current maximum level in the graph.
    max_level: u32 = 0,

    /// Precomputed mL = 1/ln(M) for random level generation.
    m_l: f64,

    /// PRNG for level assignment.
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, cfg: config.Config, engine_metric: config.DistanceMetric) !HNSW {
        const m: f64 = @floatFromInt(@max(cfg.index.hnsw_m, 2));
        return .{
            .allocator = allocator,
            .cfg = cfg.index,
            .metric = engine_metric,
            .m_l = 1.0 / @log(m),
            .rng = std.Random.DefaultPrng.init(42),
        };
    }

    pub fn deinit(self: *HNSW) void {
        for (self.vectors.items) |vec| {
            self.allocator.free(vec);
        }
        self.vectors.deinit(self.allocator);

        for (self.neighbors.items) |layers| {
            for (layers) |layer_neighbors| {
                self.allocator.free(layer_neighbors);
            }
            self.allocator.free(layers);
        }
        self.neighbors.deinit(self.allocator);
        self.node_levels.deinit(self.allocator);
    }

    // ─── Random level assignment ───────────────────────────────────────

    fn randomLevel(self: *HNSW) u32 {
        const random = self.rng.random();
        const r = random.float(f64);
        // Level = floor(−ln(uniform) × mL), capped at a sane maximum.
        const raw: f64 = -@log(@max(r, 1e-18)) * self.m_l;
        const level: u32 = @intFromFloat(@min(raw, 31.0));
        return level;
    }

    // ─── Distance helper ───────────────────────────────────────────────

    fn distance(self: *const HNSW, a: []const f32, b: []const f32) f32 {
        return switch (self.metric) {
            .cosine => 1.0 - metrics.cosineSimilarity(a, b),
            .euclidean => metrics.euclideanDistance(a, b),
            .manhattan => metrics.manhattanDistance(a, b),
            .dot_product => -metrics.dotProduct(a, b),
        };
    }

    // ─── Insert ────────────────────────────────────────────────────────

    pub fn insert(self: *HNSW, vector: []const f32) !usize {
        const cloned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(cloned);

        const node_id: u32 = @intCast(self.vectors.items.len);
        const node_level = self.randomLevel();

        // Allocate per-layer neighbor lists (initially empty).
        const num_layers = node_level + 1;
        const layer_neighbors = try self.allocator.alloc([]u32, num_layers);
        for (layer_neighbors) |*ln| {
            ln.* = &.{};
        }
        errdefer self.allocator.free(layer_neighbors);

        try self.vectors.append(self.allocator, cloned);
        errdefer _ = self.vectors.pop();

        try self.neighbors.append(self.allocator, layer_neighbors);
        errdefer _ = self.neighbors.pop();

        try self.node_levels.append(self.allocator, node_level);
        errdefer _ = self.node_levels.pop();

        if (self.entry_point == null) {
            // First node — just set as entry point.
            self.entry_point = node_id;
            self.max_level = node_level;
            return node_id;
        }

        const ep = self.entry_point.?;

        // Phase 1: Greedy descent from top → node_level+1 (find closest entry).
        var current_ep = ep;
        {
            var lc: u32 = self.max_level;
            while (lc > node_level) : (lc -= 1) {
                current_ep = self.greedyClosest(cloned, current_ep, lc);
                if (lc == 0) break;
            }
        }

        // Phase 2: Insert at layers min(node_level, max_level) down to 0.
        const insert_from = @min(node_level, self.max_level);
        {
            var lc: u32 = insert_from;
            while (true) : (lc -= 1) {
                // Find ef_construction nearest neighbors at this layer.
                const ef_c = self.cfg.hnsw_ef_construction;
                const candidates = try self.searchLayer(cloned, current_ep, ef_c, lc);
                defer self.allocator.free(candidates);

                // Select M nearest from candidates.
                const m: usize = if (lc == 0) @as(usize, self.cfg.hnsw_m) * 2 else self.cfg.hnsw_m;
                const selected = candidates[0..@min(candidates.len, m)];

                // Connect node → selected neighbors.
                try self.setNeighbors(node_id, lc, selected);

                // Connect each neighbor back → node (bidirectional).
                for (selected) |neighbor_id| {
                    try self.addNeighborPruned(neighbor_id, node_id, lc, m);
                }

                // Use closest candidate as entry for next layer down.
                if (candidates.len > 0) {
                    current_ep = candidates[0];
                }

                if (lc == 0) break;
            }
        }

        // Update entry point if new node is higher level.
        if (node_level > self.max_level) {
            self.entry_point = node_id;
            self.max_level = node_level;
        }

        return node_id;
    }

    // ─── Search ────────────────────────────────────────────────────────

    /// Returns node IDs sorted by proximity (closest first).
    pub fn search(self: *HNSW, query: []const f32, k: usize, ef: u16) ![]usize {
        if (self.entry_point == null or self.vectors.items.len == 0) {
            return try self.allocator.alloc(usize, 0);
        }

        const ep = self.entry_point.?;

        // Phase 1: Greedy descent from top layer → layer 1.
        var current_ep = ep;
        if (self.max_level > 0) {
            var lc: u32 = self.max_level;
            while (lc > 0) : (lc -= 1) {
                current_ep = self.greedyClosest(query, current_ep, lc);
            }
        }

        // Phase 2: Search layer 0 with ef expansion.
        const effective_ef: u16 = @intCast(@max(@as(usize, ef), k));
        const candidates = try self.searchLayer(query, current_ep, effective_ef, 0);
        defer self.allocator.free(candidates);

        // Return top-k.
        const take = @min(k, candidates.len);
        const results = try self.allocator.alloc(usize, take);
        for (results, 0..) |*res, i| {
            res.* = candidates[i];
        }
        return results;
    }

    // ─── Internal: greedy closest at a single layer ────────────────────

    fn greedyClosest(self: *const HNSW, query: []const f32, ep: u32, layer: u32) u32 {
        var current = ep;
        var current_dist = self.distance(query, self.vectors.items[current]);

        var changed = true;
        while (changed) {
            changed = false;
            const layer_neighbors = self.getNeighbors(current, layer);
            for (layer_neighbors) |neighbor| {
                const d = self.distance(query, self.vectors.items[neighbor]);
                if (d < current_dist) {
                    current = neighbor;
                    current_dist = d;
                    changed = true;
                }
            }
        }
        return current;
    }

    // ─── Internal: ef-bounded search at a single layer ─────────────────

    const Candidate = struct {
        id: u32,
        dist: f32,
    };

    fn searchLayer(self: *HNSW, query: []const f32, ep: u32, ef: u16, layer: u32) ![]u32 {
        const ep_dist = self.distance(query, self.vectors.items[ep]);

        // Candidates: sorted closest-first (min-heap semantics via sorted list).
        var candidates = std.ArrayListUnmanaged(Candidate){};
        defer candidates.deinit(self.allocator);
        try candidates.append(self.allocator, .{ .id = ep, .dist = ep_dist });

        // Visited set.
        var visited = std.AutoHashMapUnmanaged(u32, void){};
        defer visited.deinit(self.allocator);
        try visited.put(self.allocator, ep, {});

        // Best results (keep up to ef).
        var results = std.ArrayListUnmanaged(Candidate){};
        defer results.deinit(self.allocator);
        try results.append(self.allocator, .{ .id = ep, .dist = ep_dist });

        while (candidates.items.len > 0) {
            // Pop closest candidate.
            var best_idx: usize = 0;
            for (candidates.items, 0..) |c, i| {
                if (c.dist < candidates.items[best_idx].dist) best_idx = i;
            }
            const current = candidates.items[best_idx];
            _ = candidates.swapRemove(best_idx);

            // Furthest result distance.
            var worst_result_dist: f32 = 0;
            for (results.items) |r| {
                if (r.dist > worst_result_dist) worst_result_dist = r.dist;
            }

            // If closest candidate is further than worst result, done.
            if (current.dist > worst_result_dist and results.items.len >= ef) {
                break;
            }

            // Explore neighbors.
            const layer_neighbors = self.getNeighbors(current.id, layer);
            for (layer_neighbors) |neighbor| {
                const gop = try visited.getOrPut(self.allocator, neighbor);
                if (gop.found_existing) continue;

                const d = self.distance(query, self.vectors.items[neighbor]);

                // Recompute worst result distance.
                var current_worst: f32 = 0;
                for (results.items) |r| {
                    if (r.dist > current_worst) current_worst = r.dist;
                }

                if (results.items.len < ef or d < current_worst) {
                    try candidates.append(self.allocator, .{ .id = neighbor, .dist = d });
                    try results.append(self.allocator, .{ .id = neighbor, .dist = d });

                    // Prune results to ef size.
                    if (results.items.len > ef) {
                        // Remove worst.
                        var worst_idx: usize = 0;
                        for (results.items, 0..) |r, i| {
                            if (r.dist > results.items[worst_idx].dist) worst_idx = i;
                        }
                        _ = results.swapRemove(worst_idx);
                    }
                }
            }
        }

        // Sort results by distance (closest first) and return IDs.
        std.mem.sort(Candidate, results.items, {}, struct {
            fn cmp(_: void, a: Candidate, b: Candidate) bool {
                return a.dist < b.dist;
            }
        }.cmp);

        const out = try self.allocator.alloc(u32, results.items.len);
        for (out, 0..) |*o, i| {
            o.* = results.items[i].id;
        }
        return out;
    }

    // ─── Internal: neighbor management ─────────────────────────────────

    fn getNeighbors(self: *const HNSW, node: u32, layer: u32) []const u32 {
        const layers = self.neighbors.items[node];
        if (layer >= layers.len) return &.{};
        return layers[layer];
    }

    fn setNeighbors(self: *HNSW, node: u32, layer: u32, neighbor_ids: []const u32) !void {
        const layers = self.neighbors.items[node];
        if (layer >= layers.len) return;

        // Free old.
        if (layers[layer].len > 0) {
            self.allocator.free(layers[layer]);
        }

        // Clone new.
        layers[layer] = try self.allocator.dupe(u32, neighbor_ids);
    }

    fn addNeighborPruned(self: *HNSW, node: u32, new_neighbor: u32, layer: u32, max_m: usize) !void {
        const layers = self.neighbors.items[node];
        if (layer >= layers.len) return;

        const current = layers[layer];

        // Check if already connected.
        for (current) |n| {
            if (n == new_neighbor) return;
        }

        if (current.len < max_m) {
            // Room for more — just append.
            const new = try self.allocator.alloc(u32, current.len + 1);
            @memcpy(new[0..current.len], current);
            new[current.len] = new_neighbor;
            if (current.len > 0) self.allocator.free(current);
            layers[layer] = new;
        } else if (self.cfg.enable_pruning) {
            // Pruning: replace worst neighbor if new one is closer.
            const node_vec = self.vectors.items[node];
            const new_dist = self.distance(node_vec, self.vectors.items[new_neighbor]);

            var worst_idx: usize = 0;
            var worst_dist: f32 = 0;
            for (current, 0..) |n, i| {
                const d = self.distance(node_vec, self.vectors.items[n]);
                if (d > worst_dist) {
                    worst_dist = d;
                    worst_idx = i;
                }
            }

            if (new_dist < worst_dist) {
                // Mutable slice write: this is safe because we own the allocation.
                const mutable: []u32 = @constCast(current);
                mutable[worst_idx] = new_neighbor;
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "HNSW insert and search basic" {
    const default_cfg: config.Config = .{};
    var idx = try HNSW.init(std.testing.allocator, default_cfg, .cosine);
    defer idx.deinit();

    _ = try idx.insert(&[_]f32{ 1.0, 0.0, 0.0 });
    _ = try idx.insert(&[_]f32{ 0.0, 1.0, 0.0 });
    _ = try idx.insert(&[_]f32{ 0.0, 0.0, 1.0 });

    const query = [_]f32{ 0.9, 0.1, 0.0 };
    const res = try idx.search(&query, 1, 64);
    defer std.testing.allocator.free(res);

    try std.testing.expectEqual(@as(usize, 1), res.len);
    try std.testing.expectEqual(@as(usize, 0), res[0]); // Closest to [1,0,0]
}

test "HNSW manhattan metric" {
    const cfg: config.Config = .{ .metric = .manhattan };
    var idx = try HNSW.init(std.testing.allocator, cfg, .manhattan);
    defer idx.deinit();

    _ = try idx.insert(&[_]f32{ 1.0, 1.0, 1.0 });
    _ = try idx.insert(&[_]f32{ 5.0, 5.0, 5.0 });
    _ = try idx.insert(&[_]f32{ 2.0, 2.0, 2.0 });

    const query = [_]f32{ 1.1, 1.2, 1.0 };
    const res = try idx.search(&query, 1, 32);
    defer std.testing.allocator.free(res);

    try std.testing.expectEqual(@as(usize, 1), res.len);
    try std.testing.expectEqual(@as(usize, 0), res[0]);
}

test "HNSW multi-node search returns top-k" {
    const cfg: config.Config = .{ .metric = .euclidean };
    var idx = try HNSW.init(std.testing.allocator, cfg, .euclidean);
    defer idx.deinit();

    // Insert 10 vectors along a line.
    for (0..10) |i| {
        const val: f32 = @floatFromInt(i);
        _ = try idx.insert(&[_]f32{ val, 0, 0 });
    }

    const query = [_]f32{ 4.5, 0, 0 };
    const res = try idx.search(&query, 3, 64);
    defer std.testing.allocator.free(res);

    try std.testing.expect(res.len >= 1);
    // The closest should be 4 or 5.
    try std.testing.expect(res[0] == 4 or res[0] == 5);
}

test "HNSW empty search returns empty" {
    const cfg: config.Config = .{};
    var idx = try HNSW.init(std.testing.allocator, cfg, .cosine);
    defer idx.deinit();

    const res = try idx.search(&[_]f32{ 1, 0, 0 }, 5, 64);
    defer std.testing.allocator.free(res);

    try std.testing.expectEqual(@as(usize, 0), res.len);
}

test "HNSW recall vs brute force" {
    const cfg: config.Config = .{ .metric = .euclidean };
    var idx = try HNSW.init(std.testing.allocator, cfg, .euclidean);
    defer idx.deinit();

    // Insert 100 pseudo-random 8-dim vectors.
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    const n: usize = 100;
    const dims: usize = 8;

    var all_vecs: [n][dims]f32 = undefined;
    for (0..n) |i| {
        for (0..dims) |d| {
            all_vecs[i][d] = random.float(f32) * 2.0 - 1.0;
        }
        _ = try idx.insert(&all_vecs[i]);
    }

    // Query
    var query: [dims]f32 = undefined;
    for (0..dims) |d| {
        query[d] = random.float(f32) * 2.0 - 1.0;
    }

    const k: usize = 5;
    const hnsw_results = try idx.search(&query, k, 128);
    defer std.testing.allocator.free(hnsw_results);

    // Brute-force ground truth.
    const BFCandidate = struct { id: usize, dist: f32 };
    var bf: [n]BFCandidate = undefined;
    for (0..n) |i| {
        bf[i] = .{ .id = i, .dist = metrics.euclideanDistance(&query, &all_vecs[i]) };
    }
    std.mem.sort(BFCandidate, &bf, {}, struct {
        fn cmp(_: void, a: BFCandidate, b: BFCandidate) bool {
            return a.dist < b.dist;
        }
    }.cmp);

    // Count overlap (recall).
    var hits: usize = 0;
    for (hnsw_results) |hid| {
        for (0..k) |j| {
            if (bf[j].id == hid) {
                hits += 1;
                break;
            }
        }
    }

    // With 100 vectors and ef=128, recall should be very high.
    try std.testing.expect(hits >= 3); // At least 60% recall.
}
