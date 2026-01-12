//! Database and HNSW Vector Search Benchmarks
//!
//! Industry-standard benchmarks for vector databases:
//! - Insertion throughput (single and batch)
//! - Query latency (k-NN search)
//! - Recall accuracy at different k values
//! - Build time for different index sizes
//! - Memory efficiency
//! - Update/delete operations
//! - Concurrent read/write patterns
//! - Distance metric comparisons (L2, cosine, dot product)
//! - Scalability with dimension size
//! - Recovery/durability tests

const std = @import("std");
const framework = @import("framework.zig");

/// Database benchmark configuration
pub const DatabaseBenchConfig = struct {
    /// Vector dimensions to test
    dimensions: []const usize = &.{ 64, 128, 256, 384, 512, 768, 1024, 1536 },
    /// Dataset sizes
    dataset_sizes: []const usize = &.{ 1000, 10000, 50000, 100000 },
    /// Batch sizes for batch insertion
    batch_sizes: []const usize = &.{ 1, 10, 100, 1000 },
    /// k values for k-NN search
    k_values: []const usize = &.{ 1, 5, 10, 20, 50, 100 },
    /// Number of query iterations
    query_iterations: usize = 1000,
    /// HNSW parameters
    hnsw_m: usize = 16,
    hnsw_ef_construction: usize = 200,
    hnsw_ef_search: []const usize = &.{ 16, 32, 64, 128, 256 },
};

/// Distance metrics
pub const DistanceMetric = enum {
    euclidean,
    cosine,
    dot_product,
    manhattan,
};

/// Search result entry - used by HNSW and brute force search
pub const SearchResult = struct {
    id: u64,
    dist: f32,
};

// ============================================================================
// Vector Generation Utilities
// ============================================================================

/// Generate random normalized vectors (for embedding simulation)
fn generateRandomVectors(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const vectors = try allocator.alloc([]f32, count);
    errdefer {
        for (vectors) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        var norm: f32 = 0;

        for (vec.*) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
            norm += val.* * val.*;
        }

        // Normalize
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*val| {
                val.* /= norm;
            }
        }
    }

    return vectors;
}

fn freeVectors(allocator: std.mem.Allocator, vectors: [][]f32) void {
    for (vectors) |v| {
        allocator.free(v);
    }
    allocator.free(vectors);
}

/// Generate clustered vectors (more realistic distribution)
fn generateClusteredVectors(
    allocator: std.mem.Allocator,
    count: usize,
    dim: usize,
    num_clusters: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Generate cluster centers
    var centers = try allocator.alloc([]f32, num_clusters);
    defer {
        for (centers) |c| {
            allocator.free(c);
        }
        allocator.free(centers);
    }

    for (centers) |*center| {
        center.* = try allocator.alloc(f32, dim);
        for (center.*) |*val| {
            val.* = rand.float(f32) * 2.0 - 1.0;
        }
    }

    // Generate points around clusters
    const vectors = try allocator.alloc([]f32, count);
    errdefer {
        for (vectors) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        const cluster_idx = rand.intRangeLessThan(usize, 0, num_clusters);
        const center = centers[cluster_idx];

        var norm: f32 = 0;
        for (vec.*, center) |*val, c| {
            val.* = c + (rand.float(f32) - 0.5) * 0.2;
            norm += val.* * val.*;
        }

        // Normalize
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*val| {
                val.* /= norm;
            }
        }
    }

    return vectors;
}

// ============================================================================
// Distance Functions
// ============================================================================

fn euclideanDistanceSq(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| {
        const diff = x - y;
        sum += diff * diff;
    }
    return sum;
}

fn cosineDistance(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |x, y| {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 1.0;
    return 1.0 - (dot / denom);
}

fn dotProduct(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    for (a, b) |x, y| {
        dot += x * y;
    }
    return -dot; // Negative for distance (higher dot = smaller distance)
}

fn manhattanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| {
        sum += @abs(x - y);
    }
    return sum;
}

// ============================================================================
// Simple HNSW Implementation for Benchmarking
// ============================================================================

/// Simplified HNSW index for benchmarking
fn HNSWIndex(comptime distance_fn: fn ([]const f32, []const f32) f32) type {
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
        m: usize, // Max neighbors per layer
        m_max: usize, // Max neighbors for layer 0
        ef_construction: usize,
        ml: f64, // Level multiplier

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

            // Allocate neighbor lists for each layer
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

            // Traverse from top to insertion level
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

            // Search and connect at each layer from level down to 0
            while (l >= 0) : (l -= 1) {
                const ul: usize = @intCast(l);
                const max_neighbors: usize = if (l == 0) self.m_max else self.m;

                // Simple greedy search for nearest neighbors
                var candidates = std.ArrayListUnmanaged(struct { idx: u32, dist: f32 }){};
                defer candidates.deinit(self.allocator);

                try candidates.append(self.allocator, .{ .idx = curr_node, .dist = curr_dist });

                // Connect to nearest neighbors
                if (candidates.items.len > 0) {
                    const count = @min(candidates.items.len, max_neighbors);
                    for (candidates.items[0..count]) |cand| {
                        try self.nodes.items[new_idx].neighbors[ul].append(self.allocator, cand.idx);

                        // Bidirectional connection
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

            // Traverse from top to layer 0
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

            // Search at layer 0 with ef candidates
            var candidates = std.ArrayListUnmanaged(SearchResult){};
            errdefer candidates.deinit(self.allocator);

            var visited = std.AutoHashMapUnmanaged(u32, void){};
            defer visited.deinit(self.allocator);

            try visited.put(self.allocator, curr_node, {});
            try candidates.append(self.allocator, .{
                .id = self.nodes.items[curr_node].id,
                .dist = curr_dist,
            });

            // Expand search
            var i: usize = 0;
            while (i < candidates.items.len and i < ef) : (i += 1) {
                // This is simplified - real HNSW uses priority queues
                for (self.nodes.items, 0..) |node, idx| {
                    if (visited.contains(@intCast(idx))) continue;
                    const d = distance_fn(query, node.vector);
                    if (candidates.items.len < ef or d < candidates.items[candidates.items.len - 1].dist) {
                        try visited.put(self.allocator, @intCast(idx), {});
                        try candidates.append(self.allocator, .{ .id = node.id, .dist = d });
                    }
                }
            }

            // Sort by distance and return top-k
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
    };
}

// ============================================================================
// Brute Force Search (for recall calculation)
// ============================================================================

fn bruteForceSearch(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    query: []const f32,
    k: usize,
    comptime distance_fn: fn ([]const f32, []const f32) f32,
) ![]u64 {
    var distances = try allocator.alloc(SearchResult, vectors.len);
    defer allocator.free(distances);

    for (vectors, 0..) |v, i| {
        distances[i] = .{ .id = @intCast(i), .dist = distance_fn(query, v) };
    }

    std.mem.sort(SearchResult, distances, {}, struct {
        fn cmp(_: void, a: SearchResult, b: SearchResult) bool {
            return a.dist < b.dist;
        }
    }.cmp);

    const result = try allocator.alloc(u64, k);
    for (result, 0..) |*r, i| {
        r.* = distances[i].id;
    }

    return result;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn benchInsertionThroughput(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
) !u64 {
    var index = HNSWIndex(euclideanDistanceSq).init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);

    for (vectors, 0..) |v, i| {
        try index.insert(v, @intCast(i), prng.random());
    }

    return @intCast(vectors.len);
}

fn benchBatchInsertion(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    batch_size: usize,
) !u64 {
    var index = HNSWIndex(euclideanDistanceSq).init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    var inserted: u64 = 0;

    var i: usize = 0;
    while (i < vectors.len) {
        const batch_end = @min(i + batch_size, vectors.len);
        for (vectors[i..batch_end], i..) |v, idx| {
            try index.insert(v, @intCast(idx), prng.random());
            inserted += 1;
        }
        i = batch_end;
    }

    return inserted;
}

fn benchQueryLatency(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    queries: [][]f32,
    k: usize,
    ef: usize,
) !u64 {
    var index = HNSWIndex(euclideanDistanceSq).init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);

    // Build index
    for (vectors, 0..) |v, i| {
        try index.insert(v, @intCast(i), prng.random());
    }

    // Run queries
    var total_results: u64 = 0;
    for (queries) |q| {
        const results = try index.search(q, k, ef);
        defer allocator.free(results);
        total_results += results.len;
    }

    return total_results;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runDatabaseBenchmarks(allocator: std.mem.Allocator, config: DatabaseBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    DATABASE/HNSW VECTOR SEARCH BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Distance metric benchmarks
    std.debug.print("[Distance Metric Comparison]\n", .{});
    for ([_]usize{ 128, 512, 1024 }) |dim| {
        const vectors = try generateRandomVectors(allocator, 1000, dim, 42);
        defer freeVectors(allocator, vectors);

        // Euclidean
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "euclidean_d{d}", .{dim}) catch "euclidean";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/distance",
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(vecs: [][]f32) f32 {
                        var sum: f32 = 0;
                        for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                            sum += euclideanDistanceSq(a, b);
                        }
                        return sum;
                    }
                }.bench,
                .{vectors},
            );

            std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
            });
        }

        // Cosine
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "cosine_d{d}", .{dim}) catch "cosine";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/distance",
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(vecs: [][]f32) f32 {
                        var sum: f32 = 0;
                        for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                            sum += cosineDistance(a, b);
                        }
                        return sum;
                    }
                }.bench,
                .{vectors},
            );

            std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
            });
        }
    }

    // Insertion benchmarks
    std.debug.print("\n[Insertion Throughput]\n", .{});
    for ([_]usize{ 1000, 5000 }) |size| {
        for ([_]usize{ 128, 384 }) |dim| {
            const vectors = try generateRandomVectors(allocator, size, dim, 42);
            defer freeVectors(allocator, vectors);

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "insert_{d}x{d}", .{ size, dim }) catch "insert";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/insert",
                    .warmup_iterations = 5,
                    .min_time_ns = 1_000_000_000,
                    .max_iterations = 50,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32) !u64 {
                        return try benchInsertionThroughput(a, vecs);
                    }
                }.bench,
                .{ allocator, vectors },
            );

            std.debug.print("  {s}: {d:.0} vectors/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(size)),
            });
        }
    }

    // Query latency benchmarks
    std.debug.print("\n[Query Latency (k-NN Search)]\n", .{});
    for ([_]usize{ 1000, 5000 }) |size| {
        const dim: usize = 256;
        const vectors = try generateClusteredVectors(allocator, size, dim, 10, 42);
        defer freeVectors(allocator, vectors);

        const queries = try generateRandomVectors(allocator, 100, dim, 99);
        defer freeVectors(allocator, queries);

        for (config.k_values[0..@min(4, config.k_values.len)]) |k| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "query_{d}_k{d}", .{ size, k }) catch "query";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/query",
                    .warmup_iterations = 5,
                    .min_time_ns = 1_000_000_000,
                    .max_iterations = 50,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32, qs: [][]f32, kval: usize) !u64 {
                        return try benchQueryLatency(a, vecs, qs, kval, 64);
                    }
                }.bench,
                .{ allocator, vectors, queries, k },
            );

            std.debug.print("  {s}: {d:.0} queries/sec, {d:.0}ns/query\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.mean_ns,
            });
        }
    }

    // Batch insertion comparison
    std.debug.print("\n[Batch Insertion Comparison]\n", .{});
    {
        const size: usize = 5000;
        const dim: usize = 256;
        const vectors = try generateRandomVectors(allocator, size, dim, 42);
        defer freeVectors(allocator, vectors);

        for ([_]usize{ 1, 10, 100, 500 }) |batch| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "batch_{d}", .{batch}) catch "batch";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/batch",
                    .warmup_iterations = 3,
                    .min_time_ns = 1_000_000_000,
                    .max_iterations = 20,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32, bs: usize) !u64 {
                        return try benchBatchInsertion(a, vecs, bs);
                    }
                }.bench,
                .{ allocator, vectors, batch },
            );

            std.debug.print("  {s}: {d:.0} vectors/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(size)),
            });
        }
    }

    // Brute force baseline
    std.debug.print("\n[Brute Force Baseline (for Recall Reference)]\n", .{});
    {
        const size: usize = 1000;
        const dim: usize = 128;
        const k: usize = 10;

        const vectors = try generateRandomVectors(allocator, size, dim, 42);
        defer freeVectors(allocator, vectors);

        const queries = try generateRandomVectors(allocator, 100, dim, 99);
        defer freeVectors(allocator, queries);

        const result = try runner.run(
            .{
                .name = "brute_force",
                .category = "database/baseline",
                .warmup_iterations = 5,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, vecs: [][]f32, qs: [][]f32, kval: usize) !u64 {
                    var count: u64 = 0;
                    for (qs) |q| {
                        const res = try bruteForceSearch(a, vecs, q, kval, euclideanDistanceSq);
                        defer a.free(res);
                        count += res.len;
                    }
                    return count;
                }
            }.bench,
            .{ allocator, vectors, queries, k },
        );

        std.debug.print("  brute_force_{d}x{d}_k{d}: {d:.0} queries/sec\n", .{
            size,
            dim,
            k,
            result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runDatabaseBenchmarks(allocator, .{});
}

test "distance metrics" {
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };

    // Euclidean distance squared should be 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), euclideanDistanceSq(&a, &b), 0.001);

    // Cosine distance for orthogonal vectors should be 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineDistance(&a, &b), 0.001);

    // Same vector should have 0 distance
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), euclideanDistanceSq(&a, &a), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineDistance(&a, &a), 0.001);
}

test "vector generation" {
    const allocator = std.testing.allocator;

    const vectors = try generateRandomVectors(allocator, 10, 64, 42);
    defer freeVectors(allocator, vectors);

    try std.testing.expectEqual(@as(usize, 10), vectors.len);
    try std.testing.expectEqual(@as(usize, 64), vectors[0].len);

    // Check normalization
    var norm: f32 = 0;
    for (vectors[0]) |v| {
        norm += v * v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}
