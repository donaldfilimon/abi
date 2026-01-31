//! HNSW Index Benchmarks
//!
//! Benchmarks for Hierarchical Navigable Small World (HNSW) index operations:
//! - Index construction
//! - k-NN search
//! - Parameter sensitivity analysis

const std = @import("std");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

/// Search result entry
pub const SearchResult = struct {
    id: u64,
    dist: f32,
};

/// Simplified HNSW index for benchmarking
pub fn HNSWIndex(comptime distance_fn: fn ([]const f32, []const f32) f32) type {
    return struct {
        const Self = @This();

        const Node = struct {
            vector: []const f32,
            id: u64,
            neighbors: []std.ArrayListUnmanaged(u32),
            level: u8,
        };

        allocator: std.mem.Allocator,
        nodes: std.ArrayListUnmanaged(Node),
        entry_point: ?u32,
        max_level: u8,
        m: usize,
        m_max: usize,
        ef_construction: usize,
        ml: f64,

        pub fn init(allocator: std.mem.Allocator, m: usize, ef_construction: usize) Self {
            return .{
                .allocator = allocator,
                .nodes = .{},
                .entry_point = null,
                .max_level = 0,
                .m = m,
                .m_max = m * 2,
                .ef_construction = ef_construction,
                .ml = 1.0 / @log(@as(f64, @floatFromInt(m))),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.nodes.items) |*node| {
                for (node.neighbors) |*layer| {
                    layer.deinit(self.allocator);
                }
                self.allocator.free(node.neighbors);
            }
            self.nodes.deinit(self.allocator);
        }

        fn getRandomLevel(self: *Self, rand: std.Random) u8 {
            const r = rand.float(f64);
            const level = @as(u8, @intFromFloat(-@log(r) * self.ml));
            return @min(level, 15);
        }

        pub fn insert(self: *Self, vector: []const f32, id: u64, rand: std.Random) !void {
            const level = self.getRandomLevel(rand);

            const neighbors = try self.allocator.alloc(std.ArrayListUnmanaged(u32), level + 1);
            for (neighbors) |*layer| {
                layer.* = .{};
            }

            const node = Node{
                .vector = vector,
                .id = id,
                .neighbors = neighbors,
                .level = level,
            };

            const new_idx: u32 = @intCast(self.nodes.items.len);
            try self.nodes.append(self.allocator, node);

            if (self.entry_point == null) {
                self.entry_point = new_idx;
                self.max_level = level;
                return;
            }

            var curr_node = self.entry_point.?;
            var curr_dist = distance_fn(vector, self.nodes.items[curr_node].vector);

            var l: i16 = @as(i16, self.max_level);
            while (l > @as(i16, level)) : (l -= 1) {
                var changed = true;
                while (changed) {
                    changed = false;
                    const curr = &self.nodes.items[curr_node];
                    if (l < curr.neighbors.len) {
                        for (curr.neighbors[@intCast(l)].items) |neighbor_idx| {
                            const d = distance_fn(vector, self.nodes.items[neighbor_idx].vector);
                            if (d < curr_dist) {
                                curr_node = neighbor_idx;
                                curr_dist = d;
                                changed = true;
                            }
                        }
                    }
                }
            }

            while (l >= 0) : (l -= 1) {
                const ul: usize = @intCast(l);
                const max_neighbors: usize = if (l == 0) self.m_max else self.m;

                var candidates = std.ArrayListUnmanaged(struct { idx: u32, dist: f32 }){};
                defer candidates.deinit(self.allocator);

                try candidates.append(self.allocator, .{ .idx = curr_node, .dist = curr_dist });

                if (candidates.items.len > 0) {
                    const neighbor_count = @min(candidates.items.len, max_neighbors);
                    for (candidates.items[0..neighbor_count]) |cand| {
                        try self.nodes.items[new_idx].neighbors[ul].append(self.allocator, cand.idx);

                        if (ul < self.nodes.items[cand.idx].neighbors.len) {
                            const their_neighbors = &self.nodes.items[cand.idx].neighbors[ul];
                            if (their_neighbors.items.len < max_neighbors) {
                                try their_neighbors.append(self.allocator, new_idx);
                            }
                        }
                    }
                }
            }

            if (level > self.max_level) {
                self.max_level = level;
                self.entry_point = new_idx;
            }
        }

        pub fn search(self: *Self, query: []const f32, k: usize, ef: usize) ![]SearchResult {
            if (self.entry_point == null) return &[_]SearchResult{};

            var curr_node = self.entry_point.?;
            var curr_dist = distance_fn(query, self.nodes.items[curr_node].vector);

            var l: i16 = @as(i16, self.max_level);
            while (l > 0) : (l -= 1) {
                var changed = true;
                while (changed) {
                    changed = false;
                    const curr = &self.nodes.items[curr_node];
                    if (@as(usize, @intCast(l)) < curr.neighbors.len) {
                        for (curr.neighbors[@intCast(l)].items) |neighbor_idx| {
                            const d = distance_fn(query, self.nodes.items[neighbor_idx].vector);
                            if (d < curr_dist) {
                                curr_node = neighbor_idx;
                                curr_dist = d;
                                changed = true;
                            }
                        }
                    }
                }
            }

            var candidates = std.ArrayListUnmanaged(SearchResult){};
            errdefer candidates.deinit(self.allocator);

            var visited = std.AutoHashMapUnmanaged(u32, void){};
            defer visited.deinit(self.allocator);

            try visited.put(self.allocator, curr_node, {});
            try candidates.append(self.allocator, .{
                .id = self.nodes.items[curr_node].id,
                .dist = curr_dist,
            });

            var i: usize = 0;
            while (i < candidates.items.len and i < ef) : (i += 1) {
                for (self.nodes.items, 0..) |node, idx| {
                    if (visited.contains(@intCast(idx))) continue;
                    const d = distance_fn(query, node.vector);
                    if (candidates.items.len < ef or d < candidates.items[candidates.items.len - 1].dist) {
                        try visited.put(self.allocator, @intCast(idx), {});
                        try candidates.append(self.allocator, .{ .id = node.id, .dist = d });
                    }
                }
            }

            std.mem.sort(SearchResult, candidates.items, {}, struct {
                fn cmp(_: void, a: SearchResult, b: SearchResult) bool {
                    return a.dist < b.dist;
                }
            }.cmp);

            const result_count = @min(k, candidates.items.len);
            const result = try self.allocator.alloc(SearchResult, result_count);
            @memcpy(result, candidates.items[0..result_count]);

            return result;
        }

        pub fn count(self: *const Self) usize {
            return self.nodes.items.len;
        }

        pub fn estimateMemoryUsage(self: *const Self) u64 {
            var total: u64 = 0;
            for (self.nodes.items) |node| {
                total += node.vector.len * @sizeOf(f32);
                for (node.neighbors) |layer| {
                    total += layer.items.len * @sizeOf(u32);
                }
            }
            return total;
        }
    };
}

/// Default HNSW index using squared Euclidean distance
pub const EuclideanHNSW = HNSWIndex(core.distance.euclideanSq);

/// HNSW index using cosine distance
pub const CosineHNSW = HNSWIndex(core.distance.cosine);

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark HNSW index construction
pub fn benchBuildTime(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    m: usize,
    ef_construction: usize,
) !struct { build_time_ns: u64, index: *EuclideanHNSW } {
    var index = try allocator.create(EuclideanHNSW);
    index.* = EuclideanHNSW.init(allocator, m, ef_construction);

    var prng = std.Random.DefaultPrng.init(12345);

    var timer = std.time.Timer.start() catch return error.TimerFailed;

    for (vectors, 0..) |v, i| {
        try index.insert(v, @intCast(i), prng.random());
    }

    return .{
        .build_time_ns = timer.read(),
        .index = index,
    };
}

/// Benchmark HNSW search
pub fn benchSearch(
    allocator: std.mem.Allocator,
    index: *EuclideanHNSW,
    queries: [][]f32,
    k: usize,
    ef: usize,
) !struct { total_time_ns: u64, results_count: u64 } {
    var timer = std.time.Timer.start() catch return error.TimerFailed;
    var total_results: u64 = 0;

    for (queries) |q| {
        const results = try index.search(q, k, ef);
        defer allocator.free(results);
        total_results += results.len;
    }

    return .{
        .total_time_ns = timer.read(),
        .results_count = total_results,
    };
}

/// Run comprehensive HNSW benchmarks
pub fn runHnswBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n[HNSW Index Benchmarks]\n", .{});

    for (config.dataset_sizes[0..@min(3, config.dataset_sizes.len)]) |size| {
        for (config.dimensions[0..@min(3, config.dimensions.len)]) |dim| {
            const vectors = try core.vectors.generateNormalized(allocator, size, dim, config.seed);
            defer core.vectors.free(allocator, vectors);

            const queries = try core.vectors.generateNormalized(allocator, @min(100, config.query_iterations), dim, config.seed +% 1000);
            defer core.vectors.free(allocator, queries);

            const query_limit: usize = if (size >= 10000) @min(3, queries.len) else queries.len;
            const query_slice = queries[0..query_limit];
            const ef_search: usize = if (size >= 10000) 32 else 64;

            // Build benchmark
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "hnsw_build_{d}x{d}", .{ size, dim }) catch "hnsw_build";

                const result = try runner.run(
                    .{
                        .name = name,
                        .category = "database/hnsw",
                        .warmup_iterations = 1,
                        .min_time_ns = config.min_time_ns / 10,
                        .max_iterations = 5,
                    },
                    struct {
                        fn bench(a: std.mem.Allocator, vecs: [][]f32, m: usize, ef_c: usize) !u64 {
                            const build_result = try benchBuildTime(a, vecs, m, ef_c);
                            defer {
                                build_result.index.deinit();
                                a.destroy(build_result.index);
                            }
                            return @intCast(vecs.len);
                        }
                    }.bench,
                    .{ allocator, vectors, config.hnsw_m, config.hnsw_ef_construction },
                );

                std.debug.print("  {s}: {d:.0} vectors/sec\n", .{
                    name,
                    result.stats.opsPerSecond() * @as(f64, @floatFromInt(size)),
                });
            }

            // Search benchmark
            {
                const build_result = try benchBuildTime(allocator, vectors, config.hnsw_m, config.hnsw_ef_construction);
                defer {
                    build_result.index.deinit();
                    allocator.destroy(build_result.index);
                }

                for (config.k_values[0..@min(3, config.k_values.len)]) |k| {
                    var name_buf: [64]u8 = undefined;
                    const name = std.fmt.bufPrint(&name_buf, "hnsw_search_{d}x{d}_k{d}", .{ size, dim, k }) catch "hnsw_search";

                    const search_result = try runner.run(
                        .{
                            .name = name,
                            .category = "database/hnsw",
                            .warmup_iterations = 10,
                            .min_time_ns = config.min_time_ns / 2,
                            .max_iterations = 100,
                        },
                        struct {
                            fn bench(
                                a: std.mem.Allocator,
                                idx: *EuclideanHNSW,
                                qs: [][]f32,
                                kval: usize,
                                ef: usize,
                            ) !u64 {
                                const res = try benchSearch(a, idx, qs, kval, ef);
                                return res.results_count;
                            }
                        }.bench,
                        .{ allocator, build_result.index, query_slice, k, ef_search },
                    );

                    std.debug.print("  {s}: {d:.0} queries/sec\n", .{
                        name,
                        search_result.stats.opsPerSecond() * @as(f64, @floatFromInt(query_slice.len)),
                    });
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "hnsw basic operations" {
    const allocator = std.testing.allocator;

    var index = EuclideanHNSW.init(allocator, 16, 100);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(42);

    // Insert some vectors
    const vec1 = [_]f32{ 1, 0, 0, 0 };
    const vec2 = [_]f32{ 0, 1, 0, 0 };
    const vec3 = [_]f32{ 0.9, 0.1, 0, 0 };

    try index.insert(&vec1, 1, prng.random());
    try index.insert(&vec2, 2, prng.random());
    try index.insert(&vec3, 3, prng.random());

    try std.testing.expectEqual(@as(usize, 3), index.count());

    // Search
    const query = [_]f32{ 1, 0, 0, 0 };
    const results = try index.search(&query, 2, 10);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expectEqual(@as(u64, 1), results[0].id); // Exact match should be first
}
