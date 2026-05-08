//! In-memory Vamana graph index for approximate nearest neighbor search.
//!
//! Implements the Vamana algorithm from the DiskANN paper: greedy neighbor
//! selection with alpha-based RNG pruning. This is a pure in-memory index
//! operating on raw `[]const f32` vectors with L2 distance.

const std = @import("std");
const types = @import("types.zig");
const SearchCandidate = types.SearchCandidate;
const computeL2DistanceSquared = types.computeL2DistanceSquared;

/// Vamana graph configuration
pub const VamanaConfig = struct {
    max_degree: u32 = 64,
    alpha: f32 = 1.2,
    build_list_size: u32 = 128,
    search_list_size: u32 = 64,
    beam_width: u32 = 4,
};

/// Search result from VamanaIndex
pub const VamanaSearchResult = struct {
    id: u32,
    distance: f32,

    fn lessThan(_: void, a: VamanaSearchResult, b: VamanaSearchResult) bool {
        return a.distance < b.distance;
    }
};

/// In-memory Vamana graph index for approximate nearest neighbor search.
///
/// Implements the Vamana algorithm from the DiskANN paper: greedy neighbor
/// selection with alpha-based RNG pruning. This is a pure in-memory index
/// operating on raw `[]const f32` vectors with L2 distance.
pub const VamanaIndex = struct {
    allocator: std.mem.Allocator,
    config: VamanaConfig,
    dim: u32,
    /// Adjacency lists: graph.items[i] contains neighbor IDs for node i
    graph: std.ArrayListUnmanaged(std.ArrayListUnmanaged(u32)),
    /// Stored vectors (owned copies)
    vectors: std.ArrayListUnmanaged([]f32),
    entry_point: u32 = 0,
    num_vectors: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, dim: u32, config: VamanaConfig) VamanaIndex {
        return .{
            .allocator = allocator,
            .config = config,
            .dim = dim,
            .graph = .empty,
            .vectors = .empty,
        };
    }

    pub fn deinit(self: *VamanaIndex) void {
        for (self.graph.items) |*adj| {
            adj.deinit(self.allocator);
        }
        self.graph.deinit(self.allocator);
        for (self.vectors.items) |vec| {
            self.allocator.free(vec);
        }
        self.vectors.deinit(self.allocator);
    }

    /// Build the Vamana graph from a flat array of vectors.
    /// `data` must contain `n * dim` floats where n = data.len / dim.
    pub fn build(self: *VamanaIndex, data: []const f32) !void {
        const dim = self.dim;
        const n: u32 = @intCast(data.len / dim);
        if (n == 0) return error.EmptyDataset;

        // Store vectors
        try self.vectors.ensureTotalCapacity(self.allocator, n);
        try self.graph.ensureTotalCapacity(self.allocator, n);
        for (0..n) |i| {
            const src = data[i * dim ..][0..dim];
            const copy = try self.allocator.alloc(f32, dim);
            @memcpy(copy, src);
            self.vectors.appendAssumeCapacity(copy);
            self.graph.appendAssumeCapacity(.empty);
        }
        self.num_vectors = n;

        // Select medoid as entry point
        self.entry_point = self.findMedoid();

        // Initialize random graph: connect each node to max_degree random neighbors
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const max_deg = self.config.max_degree;

        for (0..n) |i| {
            var count: u32 = 0;
            const target = @min(max_deg, n - 1);
            var attempts: u32 = 0;
            while (count < target and attempts < target * 3) : (attempts += 1) {
                const j = random.intRangeAtMost(u32, 0, n - 1);
                if (j == @as(u32, @intCast(i))) continue;
                // Check for duplicate
                var dup = false;
                for (self.graph.items[i].items) |existing| {
                    if (existing == j) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    try self.graph.items[i].append(self.allocator, j);
                    count += 1;
                }
            }
        }

        // Vamana iteration: for each vector in random order, improve its neighborhood
        var order = try self.allocator.alloc(u32, n);
        defer self.allocator.free(order);
        for (0..n) |i| {
            order[i] = @intCast(i);
        }
        // Fisher-Yates shuffle
        for (0..n) |i| {
            const j = random.intRangeAtMost(usize, i, n - 1);
            const tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }

        for (order) |node_id| {
            // Greedy search from entry point to find candidates
            const candidates = try self.greedySearchInternal(self.vectors.items[node_id], self.config.build_list_size);
            defer self.allocator.free(candidates);

            // Robust prune: select neighbors using alpha-RNG criterion
            const new_neighbors = try self.robustPruneInternal(node_id, candidates);
            defer self.allocator.free(new_neighbors);

            // Replace forward edges
            self.graph.items[node_id].clearRetainingCapacity();
            for (new_neighbors) |nbr| {
                try self.graph.items[node_id].append(self.allocator, nbr);
            }

            // Add reverse edges (maintain max_degree)
            for (new_neighbors) |nbr| {
                if (self.graph.items[nbr].items.len < max_deg) {
                    // Check duplicate
                    var found = false;
                    for (self.graph.items[nbr].items) |existing| {
                        if (existing == node_id) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        try self.graph.items[nbr].append(self.allocator, node_id);
                    }
                } else {
                    // Neighbor list full -- prune to make room if beneficial
                    // Collect current neighbors + new candidate as SearchCandidates
                    var nbr_candidates = try self.allocator.alloc(SearchCandidate, self.graph.items[nbr].items.len + 1);
                    defer self.allocator.free(nbr_candidates);
                    for (self.graph.items[nbr].items, 0..) |existing, ci| {
                        nbr_candidates[ci] = .{
                            .id = existing,
                            .distance = computeL2DistanceSquared(self.vectors.items[nbr], self.vectors.items[existing]),
                        };
                    }
                    nbr_candidates[self.graph.items[nbr].items.len] = .{
                        .id = node_id,
                        .distance = computeL2DistanceSquared(self.vectors.items[nbr], self.vectors.items[node_id]),
                    };
                    // Sort by distance
                    std.mem.sort(SearchCandidate, nbr_candidates, {}, struct {
                        fn lt(_: void, a: SearchCandidate, b: SearchCandidate) bool {
                            return std.math.order(a.distance, b.distance) == .lt;
                        }
                    }.lt);
                    const pruned = try self.robustPruneInternal(nbr, nbr_candidates);
                    defer self.allocator.free(pruned);
                    self.graph.items[nbr].clearRetainingCapacity();
                    for (pruned) |p| {
                        try self.graph.items[nbr].append(self.allocator, p);
                    }
                }
            }
        }
    }

    /// Search for the k nearest neighbors of `query`.
    pub fn search(self: *const VamanaIndex, query: []const f32, k: u32) ![]VamanaSearchResult {
        if (self.num_vectors == 0) return try self.allocator.alloc(VamanaSearchResult, 0);

        const candidates = try self.greedySearchInternal(query, @max(self.config.search_list_size, k));
        defer self.allocator.free(candidates);

        const result_count = @min(k, @as(u32, @intCast(candidates.len)));
        const results = try self.allocator.alloc(VamanaSearchResult, result_count);
        for (0..result_count) |i| {
            results[i] = .{
                .id = candidates[i].id,
                .distance = @sqrt(candidates[i].distance),
            };
        }
        return results;
    }

    fn findMedoid(self: *const VamanaIndex) u32 {
        const dim = self.dim;
        const n = self.num_vectors;
        const sample = @min(n, 1000);

        // Compute centroid from sample
        var centroid_buf: [4096]f32 = undefined;
        const centroid = centroid_buf[0..dim];
        @memset(centroid, 0);

        for (0..sample) |i| {
            for (0..dim) |d| {
                centroid[d] += self.vectors.items[i][d];
            }
        }
        const sample_f: f32 = @floatFromInt(sample);
        for (centroid) |*c| {
            c.* /= sample_f;
        }

        var best: u32 = 0;
        var best_dist: f32 = std.math.inf(f32);
        for (0..n) |i| {
            const d = computeL2DistanceSquared(self.vectors.items[i], centroid);
            if (d < best_dist) {
                best_dist = d;
                best = @intCast(i);
            }
        }
        return best;
    }

    fn greedySearchInternal(self: *const VamanaIndex, query: []const f32, list_size: u32) ![]SearchCandidate {
        var visited: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer visited.deinit(self.allocator);

        // Min-heap for frontier expansion
        var frontier = std.PriorityQueue(SearchCandidate, void, SearchCandidate.lessThan).initContext({});
        try frontier.ensureTotalCapacity(self.allocator, list_size);
        defer frontier.deinit(self.allocator);

        // Result list kept sorted by distance
        var result = std.ArrayListUnmanaged(SearchCandidate).empty;
        defer result.deinit(self.allocator);

        const entry_dist = computeL2DistanceSquared(query, self.vectors.items[self.entry_point]);
        try frontier.push(self.allocator, .{ .id = self.entry_point, .distance = entry_dist });
        try visited.put(self.allocator, self.entry_point, {});

        while (frontier.count() > 0) {
            const current = frontier.pop() orelse break;

            try result.append(self.allocator, current);

            // Sort result by distance and trim to list_size
            std.mem.sort(SearchCandidate, result.items, {}, struct {
                fn lt(_: void, a: SearchCandidate, b: SearchCandidate) bool {
                    return std.math.order(a.distance, b.distance) == .lt;
                }
            }.lt);
            if (result.items.len > list_size) {
                result.shrinkRetainingCapacity(list_size);
            }

            // Early termination: if current is worse than the worst in result and result is full
            if (result.items.len >= list_size and current.distance > result.items[list_size - 1].distance) {
                break;
            }

            // Expand neighbors
            if (current.id < self.graph.items.len) {
                for (self.graph.items[current.id].items) |nbr_id| {
                    if (visited.contains(nbr_id)) continue;
                    try visited.put(self.allocator, nbr_id, {});
                    const nbr_dist = computeL2DistanceSquared(query, self.vectors.items[nbr_id]);
                    try frontier.push(self.allocator, .{ .id = nbr_id, .distance = nbr_dist });
                }
            }
        }

        // Final sort
        std.mem.sort(SearchCandidate, result.items, {}, struct {
            fn lt(_: void, a: SearchCandidate, b: SearchCandidate) bool {
                return std.math.order(a.distance, b.distance) == .lt;
            }
        }.lt);

        return try result.toOwnedSlice(self.allocator);
    }

    fn robustPruneInternal(self: *const VamanaIndex, node_id: u32, candidates: []const SearchCandidate) ![]u32 {
        var pruned = std.ArrayListUnmanaged(u32).empty;
        const alpha = self.config.alpha;
        const max_deg = self.config.max_degree;

        for (candidates) |candidate| {
            if (candidate.id == node_id) continue;
            if (pruned.items.len >= max_deg) break;

            var dominated = false;
            for (pruned.items) |existing_id| {
                const dist_existing_to_cand = computeL2DistanceSquared(
                    self.vectors.items[existing_id],
                    self.vectors.items[candidate.id],
                );
                if (dist_existing_to_cand * alpha < candidate.distance) {
                    dominated = true;
                    break;
                }
            }

            if (!dominated) {
                try pruned.append(self.allocator, candidate.id);
            }
        }

        return try pruned.toOwnedSlice(self.allocator);
    }

    /// Return the maximum out-degree in the graph (for constraint verification).
    pub fn maxOutDegree(self: *const VamanaIndex) u32 {
        var max: u32 = 0;
        for (self.graph.items) |adj| {
            const deg: u32 = @intCast(adj.items.len);
            if (deg > max) max = deg;
        }
        return max;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "vamana build 100 random vectors and search returns correct nearest" {
    const allocator = std.testing.allocator;
    const dim: u32 = 8;
    const n: u32 = 100;

    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    // Generate random vectors
    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32) * 2.0 - 1.0;
    }

    var idx = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 1.2,
        .build_list_size = 32,
        .search_list_size = 32,
    });
    defer idx.deinit();
    try idx.build(&data);

    try std.testing.expectEqual(@as(u32, n), idx.num_vectors);

    // Use the first vector as query; it should be its own nearest neighbor
    const query = data[0..dim];
    const results = try idx.search(query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    // The exact vector should be returned with distance ~0
    try std.testing.expect(results[0].distance < 0.01);

    // Brute-force find true nearest neighbor
    var bf_best_id: u32 = 0;
    var bf_best_dist: f32 = std.math.inf(f32);
    for (0..n) |i| {
        const d = computeL2DistanceSquared(query, data[i * dim ..][0..dim]);
        if (d < bf_best_dist) {
            bf_best_dist = d;
            bf_best_id = @intCast(i);
        }
    }
    try std.testing.expectEqual(bf_best_id, results[0].id);
}

test "vamana max_degree constraint is maintained" {
    const allocator = std.testing.allocator;
    const dim: u32 = 4;
    const n: u32 = 50;
    const max_deg: u32 = 8;

    var prng = std.Random.DefaultPrng.init(7777);
    const random = prng.random();

    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32);
    }

    var idx = VamanaIndex.init(allocator, dim, .{
        .max_degree = max_deg,
        .alpha = 1.2,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer idx.deinit();
    try idx.build(&data);

    try std.testing.expect(idx.maxOutDegree() <= max_deg);
}

test "vamana different alpha values affect graph density" {
    const allocator = std.testing.allocator;
    const dim: u32 = 4;
    const n: u32 = 40;

    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32);
    }

    // Low alpha (strict pruning) -> fewer edges
    var index_low = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 1.0,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer index_low.deinit();
    try index_low.build(&data);

    // High alpha (relaxed pruning) -> more edges
    var index_high = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 2.0,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer index_high.deinit();
    try index_high.build(&data);

    // Count total edges
    var edges_low: u64 = 0;
    for (index_low.graph.items) |adj| {
        edges_low += adj.items.len;
    }
    var edges_high: u64 = 0;
    for (index_high.graph.items) |adj| {
        edges_high += adj.items.len;
    }

    // Higher alpha should produce at least as many edges (more lenient pruning)
    try std.testing.expect(edges_high >= edges_low);

    // Both should still respect max_degree
    try std.testing.expect(index_low.maxOutDegree() <= 16);
    try std.testing.expect(index_high.maxOutDegree() <= 16);
}
