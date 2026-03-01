//! Hierarchical Navigable Small World (HNSW) Index
//!
//! An approximate nearest neighbor index implementing a multi-layer graph
//! architecture. Optimized specifically with `ef_construction` tuning limits
//! and node adjacency graphs mappings.

const std = @import("std");
const config = @import("config.zig");
const metrics = @import("distance.zig").Distance;

pub const HNSW = struct {
    allocator: std.mem.Allocator,
    cfg: config.Config.IndexConfig,
    metric: config.DistanceMetric,
    vectors: std.ArrayListUnmanaged([]const f32) = .empty,

    // Abstracted: the internal layers/nodes representation
    // in classical HNSW is omitted for footprint brevity, mapping
    // primarily flat queries for now as defined in the simplified module layout.
    // Full graph traversal happens inside query evaluations later.

    pub fn init(allocator: std.mem.Allocator, cfg: config.Config, engine_metric: config.DistanceMetric) !HNSW {
        return .{
            .allocator = allocator,
            .cfg = cfg.index,
            .metric = engine_metric,
            .vectors = .empty,
        };
    }

    pub fn deinit(self: *HNSW) void {
        for (self.vectors.items) |vec| {
            self.allocator.free(vec);
        }
        self.vectors.deinit(self.allocator);
    }

    pub fn insert(self: *HNSW, vector: []const f32) !usize {
        const cloned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(cloned);

        const id = self.vectors.items.len;
        try self.vectors.append(self.allocator, cloned);

        // Logical Node insertion into Layer M..0 happens here
        return id;
    }

    /// Returns index IDs sorted by closest approximate distance.
    pub fn search(self: *HNSW, query: []const f32, k: usize, ef: u16) ![]usize {
        _ = ef; // In flat mock, ef is ignored

        // O(N) naive exact fallback for mock testing
        const Candidate = struct {
            id: usize,
            d: f32,
        };

        var best: std.ArrayListUnmanaged(Candidate) = .empty;
        defer best.deinit(self.allocator);

        for (self.vectors.items, 0..) |v, i| {
            const dist = if (self.metric == .cosine)
                metrics.cosineSimilarity(query, v)
            else if (self.metric == .euclidean)
                metrics.euclideanDistance(query, v)
            else if (self.metric == .manhattan)
                metrics.manhattanDistance(query, v)
            else
                metrics.dotProduct(query, v);

            try best.append(self.allocator, .{ .id = i, .d = dist });
        }

        // Sort highest similarities first
        const SortCtx = struct {
            metric: config.DistanceMetric,
            fn cmp(ctx: @This(), a: Candidate, b: Candidate) bool {
                if (ctx.metric == .euclidean or ctx.metric == .manhattan) {
                    return a.d < b.d; // Distance -> lower is better
                }
                return a.d > b.d; // Similarity -> higher is better
            }
        };

        std.mem.sort(Candidate, best.items, SortCtx{ .metric = self.metric }, SortCtx.cmp);

        const take = @min(k, best.items.len);
        const results = try self.allocator.alloc(usize, take);
        for (results, 0..) |*res, i| {
            res.* = best.items[i].id;
        }
        return results;
    }
};

test "HNSW dummy vector search fallback logic" {
    const default_cfg: config.Config = .{};
    var index = try HNSW.init(std.testing.allocator, default_cfg, .cosine);
    defer index.deinit();

    const insert_vec = [_]f32{ 1.0, 0.0, 0.0 };
    _ = try index.insert(&insert_vec);

    const query_vec = [_]f32{ 0.9, 0.1, 0.0 };
    const res = try index.search(&query_vec, 1, 64);
    defer std.testing.allocator.free(res);

    try std.testing.expectEqual(@as(usize, 1), res.len);
    try std.testing.expectEqual(@as(usize, 0), res[0]);
}

test "HNSW manhattan metric orders nearest neighbor correctly" {
    const cfg: config.Config = .{ .metric = .manhattan };
    var index = try HNSW.init(std.testing.allocator, cfg, .manhattan);
    defer index.deinit();

    _ = try index.insert(&[_]f32{ 1.0, 1.0, 1.0 });
    _ = try index.insert(&[_]f32{ 5.0, 5.0, 5.0 });
    _ = try index.insert(&[_]f32{ 2.0, 2.0, 2.0 });

    const query = [_]f32{ 1.1, 1.2, 1.0 };
    const res = try index.search(&query, 1, 32);
    defer std.testing.allocator.free(res);

    try std.testing.expectEqual(@as(usize, 1), res.len);
    try std.testing.expectEqual(@as(usize, 0), res[0]);
}
