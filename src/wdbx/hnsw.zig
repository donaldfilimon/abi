//! Hierarchical Navigable Small World (HNSW) Index
//!
//! Multi-layer proximity graph for O(log N) approximate nearest-neighbor search.
//! Each node exists on layers 0..max_level; higher layers are progressively sparser.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (2018).

const std = @import("std");
const Allocator = std.mem.Allocator;
const simd = @import("simd.zig");

pub const Config = struct {
    M: u32 = 16,
    M0: u32 = 32,
    ef_construction: u32 = 200,
    ef_search: u32 = 50,
    ml: f64 = 1.0 / @log(2.0),
    metric: Metric = .cosine,
    dimension: u32 = 768,

    pub const Metric = enum { cosine, l2, inner_product };
};

pub const SearchResult = struct {
    id: u64,
    distance: f32,
    index: u32,
};

pub const HnswIndex = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,

    /// Stored vectors (index = node id).
    vectors: std.ArrayListUnmanaged([]f32) = .empty,
    /// Per-node, per-layer neighbor lists: neighbors[node][layer] = []u32.
    neighbors: std.ArrayListUnmanaged([][]u32) = .empty,
    /// Level each node was assigned.
    node_levels: std.ArrayListUnmanaged(u32) = .empty,
    /// Entry point (highest-level node).
    entry_point: ?u32 = null,
    /// Current max level in the graph.
    max_level: u32 = 0,
    /// PRNG for random level generation.
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: Allocator, config: Config) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .rng = std.Random.DefaultPrng.init(42),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.vectors.items) |vec| self.allocator.free(vec);
        self.vectors.deinit(self.allocator);

        for (self.neighbors.items) |layers| {
            for (layers) |nbrs| self.allocator.free(nbrs);
            self.allocator.free(layers);
        }
        self.neighbors.deinit(self.allocator);
        self.node_levels.deinit(self.allocator);
    }

    pub fn len(self: *const Self) usize {
        return self.vectors.items.len;
    }

    // ── Distance helper ──────────────────────────────────────────────

    fn dist(self: *const Self, a: []const f32, b: []const f32) f32 {
        return switch (self.config.metric) {
            .cosine => 1.0 - simd.Distance.cosine(a, b),
            .l2 => simd.Distance.l2Squared(a, b),
            .inner_product => -simd.Distance.innerProduct(a, b),
        };
    }

    // ── Random level ─────────────────────────────────────────────────

    fn randomLevel(self: *Self) u32 {
        const r = self.rng.random().float(f64);
        const raw: f64 = -@log(@max(r, 1e-18)) * self.config.ml;
        return @intFromFloat(@min(raw, 31.0));
    }

    // ── Insert ───────────────────────────────────────────────────────

    pub fn insert(self: *Self, id: u64, vector: []const f32) !void {
        _ = id; // id is implicit = vectors.items.len

        const cloned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(cloned);

        const node_id: u32 = @intCast(self.vectors.items.len);
        const node_level = self.randomLevel();

        // Allocate per-layer neighbor lists.
        const num_layers = node_level + 1;
        const layer_nbrs = try self.allocator.alloc([]u32, num_layers);
        for (layer_nbrs) |*ln| ln.* = &.{};
        errdefer self.allocator.free(layer_nbrs);

        try self.vectors.append(self.allocator, cloned);
        errdefer _ = self.vectors.pop();
        try self.neighbors.append(self.allocator, layer_nbrs);
        errdefer _ = self.neighbors.pop();
        try self.node_levels.append(self.allocator, node_level);
        errdefer _ = self.node_levels.pop();

        if (self.entry_point == null) {
            self.entry_point = node_id;
            self.max_level = node_level;
            return;
        }

        var ep = self.entry_point.?;

        // Phase 1: Greedy descent from top to node_level + 1.
        var level: u32 = self.max_level;
        while (level > node_level) : (level -= 1) {
            ep = self.greedyClosest(cloned, ep, level);
            if (level == 0) break;
        }

        // Phase 2: Insert at each layer from min(node_level, max_level) down to 0.
        const start_level = @min(node_level, self.max_level);
        level = start_level;
        while (true) {
            const max_neighbors: u32 = if (level == 0) self.config.M0 else self.config.M;
            const candidates = try self.searchLayer(cloned, ep, self.config.ef_construction, level);
            defer self.allocator.free(candidates);

            // Select up to max_neighbors nearest from candidates.
            const select_count = @min(max_neighbors, @as(u32, @intCast(candidates.len)));
            const selected = candidates[0..select_count];

            // Set neighbors for new node at this layer.
            const new_nbrs = try self.allocator.alloc(u32, select_count);
            for (selected, 0..) |cand, i| new_nbrs[i] = cand.index;
            self.allocator.free(layer_nbrs[level]);
            layer_nbrs[level] = new_nbrs;

            // Add back-links and prune.
            for (selected) |cand| {
                try self.addNeighbor(cand.index, node_id, level, max_neighbors);
            }

            if (candidates.len > 0) ep = candidates[0].index;
            if (level == 0) break;
            level -= 1;
        }

        if (node_level > self.max_level) {
            self.max_level = node_level;
            self.entry_point = node_id;
        }
    }

    // ── Search ───────────────────────────────────────────────────────

    pub fn search(self: *Self, query: []const f32, k: u32) ![]SearchResult {
        if (self.entry_point == null) return &.{};

        var ep = self.entry_point.?;

        // Greedy descent through upper layers.
        var level: u32 = self.max_level;
        while (level > 0) : (level -= 1) {
            ep = self.greedyClosest(query, ep, level);
        }

        // Search layer 0 with ef_search.
        const candidates = try self.searchLayer(query, ep, self.config.ef_search, 0);
        defer self.allocator.free(candidates);

        const result_count = @min(k, @as(u32, @intCast(candidates.len)));
        const results = try self.allocator.alloc(SearchResult, result_count);
        for (candidates[0..result_count], 0..) |c, i| {
            results[i] = .{
                .id = c.index,
                .distance = c.distance,
                .index = c.index,
            };
        }
        return results;
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn greedyClosest(self: *const Self, query: []const f32, start: u32, level: u32) u32 {
        var current = start;
        var current_dist = self.dist(query, self.vectors.items[current]);

        var changed = true;
        while (changed) {
            changed = false;
            if (level < self.neighbors.items[current].len) {
                for (self.neighbors.items[current][level]) |nbr| {
                    const d = self.dist(query, self.vectors.items[nbr]);
                    if (d < current_dist) {
                        current = nbr;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
        }
        return current;
    }

    const Candidate = struct {
        index: u32,
        distance: f32,

        fn lessThan(_: void, a: Candidate, b: Candidate) bool {
            return a.distance < b.distance;
        }
    };

    fn searchLayer(self: *Self, query: []const f32, ep: u32, ef: u32, level: u32) ![]Candidate {
        var visited = std.AutoHashMap(u32, void).init(self.allocator);
        defer visited.deinit();

        var candidates = std.ArrayList(Candidate).init(self.allocator);
        defer candidates.deinit();

        const ep_dist = self.dist(query, self.vectors.items[ep]);
        try candidates.append(.{ .index = ep, .distance = ep_dist });
        try visited.put(ep, {});

        var i: usize = 0;
        while (i < candidates.items.len) : (i += 1) {
            const current = candidates.items[i];
            if (level < self.neighbors.items[current.index].len) {
                for (self.neighbors.items[current.index][level]) |nbr| {
                    if (visited.contains(nbr)) continue;
                    try visited.put(nbr, {});

                    const d = self.dist(query, self.vectors.items[nbr]);

                    // Add if within ef or better than worst.
                    if (candidates.items.len < ef or d < candidates.items[candidates.items.len - 1].distance) {
                        try candidates.append(.{ .index = nbr, .distance = d });
                        std.mem.sort(Candidate, candidates.items, {}, Candidate.lessThan);
                        if (candidates.items.len > ef) {
                            candidates.items.len = ef;
                        }
                    }
                }
            }
        }

        // Return owned slice sorted by distance.
        const result = try self.allocator.dupe(Candidate, candidates.items);
        return result;
    }

    fn addNeighbor(self: *Self, node: u32, new_nbr: u32, level: u32, max_neighbors: u32) !void {
        if (level >= self.neighbors.items[node].len) return;

        const old = self.neighbors.items[node][level];

        // Check if already connected.
        for (old) |n| {
            if (n == new_nbr) return;
        }

        if (old.len < max_neighbors) {
            // Room to add.
            const new_list = try self.allocator.alloc(u32, old.len + 1);
            @memcpy(new_list[0..old.len], old);
            new_list[old.len] = new_nbr;
            self.allocator.free(old);
            self.neighbors.items[node][level] = new_list;
        } else {
            // Prune: find the furthest neighbor and replace if new one is closer.
            const node_vec = self.vectors.items[node];
            var worst_idx: usize = 0;
            var worst_dist: f32 = 0.0;
            for (old, 0..) |nbr, idx| {
                const d = self.dist(node_vec, self.vectors.items[nbr]);
                if (d > worst_dist) {
                    worst_dist = d;
                    worst_idx = idx;
                }
            }
            const new_dist = self.dist(node_vec, self.vectors.items[new_nbr]);
            if (new_dist < worst_dist) {
                // Cast away const for the mutable neighbor list.
                const mutable: []u32 = @constCast(old);
                mutable[worst_idx] = new_nbr;
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "hnsw basic insert and search" {
    const allocator = std.testing.allocator;

    var index = HnswIndex.init(allocator, .{
        .dimension = 4,
        .M = 4,
        .M0 = 8,
        .ef_construction = 16,
        .ef_search = 16,
    });
    defer index.deinit();

    // Insert some vectors.
    const vecs = [_][4]f32{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.9, 0.1, 0.0, 0.0 },
        .{ 0.1, 0.9, 0.0, 0.0 },
    };

    for (vecs, 0..) |*v, i| {
        try index.insert(@intCast(i), v);
    }
    try std.testing.expectEqual(@as(usize, 5), index.len());

    // Search for vector close to first one.
    const query = [_]f32{ 0.95, 0.05, 0.0, 0.0 };
    const results = try index.search(&query, 3);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expect(results.len <= 3);
}

test "hnsw empty search" {
    const allocator = std.testing.allocator;
    var index = HnswIndex.init(allocator, .{ .dimension = 4 });
    defer index.deinit();

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try index.search(&query, 5);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}
