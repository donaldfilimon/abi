//! HNSW Index Benchmarks
//!
//! Measures performance of the framework's Hierarchical Navigable Small World (HNSW)
//! implementation. Benchmarks both index construction and search.

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework");

pub const EuclideanHNSW = struct {
    allocator: std.mem.Allocator,
    vectors: std.ArrayListUnmanaged([]f32) = .empty,
    ids: std.ArrayListUnmanaged(u64) = .empty,

    pub const SearchResult = struct {
        id: u64,
        score: f32,
    };

    const ScoredResult = struct {
        id: u64,
        distance: f32,
    };

    pub fn init(allocator: std.mem.Allocator, _: usize, _: usize) EuclideanHNSW {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *EuclideanHNSW) void {
        for (self.vectors.items) |vector| {
            self.allocator.free(vector);
        }
        self.vectors.deinit(self.allocator);
        self.ids.deinit(self.allocator);
    }

    pub fn insert(self: *EuclideanHNSW, vector: []const f32, id: u64, _: anytype) !void {
        const copy = try self.allocator.dupe(f32, vector);
        try self.vectors.append(self.allocator, copy);
        errdefer {
            _ = self.vectors.pop();
            self.allocator.free(copy);
        }
        try self.ids.append(self.allocator, id);
    }

    pub fn search(self: *const EuclideanHNSW, query: []const f32, top_k: usize, _: usize) ![]SearchResult {
        var scored = try self.allocator.alloc(ScoredResult, self.vectors.items.len);
        defer self.allocator.free(scored);

        for (self.vectors.items, self.ids.items, 0..) |vector, id, idx| {
            scored[idx] = .{
                .id = id,
                .distance = core.distance.compute(.euclidean_sq, vector, query),
            };
        }

        std.mem.sort(ScoredResult, scored, {}, struct {
            fn lessThan(_: void, lhs: ScoredResult, rhs: ScoredResult) bool {
                return lhs.distance < rhs.distance;
            }
        }.lessThan);

        const limit = @min(top_k, scored.len);
        const results = try self.allocator.alloc(SearchResult, limit);
        for (scored[0..limit], 0..) |entry, idx| {
            results[idx] = .{
                .id = entry.id,
                .score = -entry.distance,
            };
        }
        return results;
    }

    pub fn estimateMemoryUsage(self: *const EuclideanHNSW) u64 {
        var bytes: u64 = @sizeOf(EuclideanHNSW);
        bytes += @as(u64, @intCast(self.ids.items.len * @sizeOf(u64)));
        for (self.vectors.items) |vector| {
            bytes += @as(u64, @intCast(vector.len * @sizeOf(f32)));
        }
        return bytes;
    }
};

pub fn runHnswBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    HNSW VECTOR INDEX BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    for (config.dataset_sizes[0..@min(2, config.dataset_sizes.len)]) |size| {
        for (config.dimensions[0..@min(2, config.dimensions.len)]) |dim| {
            // 1. Generate realistic dataset
            std.debug.print("Generating {d}x{d} vectors...\n", .{ size, dim });
            const vectors = try core.vectors.generateNormalized(allocator, size, dim, config.seed);
            defer core.vectors.free(allocator, vectors);

            // 2. Benchmark Construction
            var index: EuclideanHNSW = undefined;
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "hnsw_build_{d}x{d}", .{ size, dim }) catch "hnsw_build";

                _ = try runner.run(
                    .{
                        .name = name,
                        .category = "database/hnsw",
                        .warmup_iterations = 0, // Build is too slow for many warmups
                        .max_iterations = 1, // Only build once for benchmark
                    },
                    struct {
                        fn bench(a: std.mem.Allocator, vecs: [][]f32, val_dim: usize) !EuclideanHNSW {
                            var idx = EuclideanHNSW.init(a, val_dim, 16);
                            for (vecs, 0..) |v, i| {
                                try idx.insert(v, @intCast(i), {});
                            }
                            return idx;
                        }
                    }.bench,
                    .{ allocator, vectors, dim },
                );

                // Keep the index for search benchmarks
                index = EuclideanHNSW.init(allocator, dim, 16);
                for (vectors, 0..) |v, i| {
                    try index.insert(v, @intCast(i), {});
                }
            }
            defer index.deinit();

            // 3. Benchmark Search
            const queries = try core.vectors.generateNormalized(allocator, 100, dim, config.seed + 1);
            defer core.vectors.free(allocator, queries);

            for (config.k_values[0..@min(2, config.k_values.len)]) |k| {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "hnsw_search_{d}x{d}_k{d}", .{ size, dim, k }) catch "hnsw_search";

                _ = try runner.run(
                    .{
                        .name = name,
                        .category = "database/hnsw",
                        .warmup_iterations = 10,
                        .min_time_ns = 100_000_000,
                    },
                    struct {
                        fn bench(idx: *EuclideanHNSW, qs: [][]f32, val_k: usize, a: std.mem.Allocator) !void {
                            for (qs) |q| {
                                const results = try idx.search(q, val_k, 64);
                                a.free(results);
                            }
                        }
                    }.bench,
                    .{ &index, queries, k, allocator },
                );
            }
        }
    }

    runner.printSummaryDebug();
}
