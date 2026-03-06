//! HNSW Index Benchmarks
//!
//! Measures performance of the framework's Hierarchical Navigable Small World (HNSW)
//! implementation. Benchmarks both index construction and search.

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

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
            var index: abi.HnswIndex = undefined;
            {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "hnsw_build_{d}x{d}", .{ size, dim }) catch "hnsw_build";

                _ = try runner.run(
                    .{
                        .name = name,
                        .category = "database/hnsw",
                        .warmup_iterations = 0, // Build is too slow for many warmups
                        .max_iterations = 1,    // Only build once for benchmark
                    },
                    struct {
                        fn bench(a: std.mem.Allocator, vecs: [][]const f32) !abi.HnswIndex {
                            var idx = try abi.HnswIndex.init(a, .{}, .cosine);
                            for (vecs) |v| {
                                _ = try idx.insert(v);
                            }
                            return idx;
                        }
                    }.bench,
                    .{ allocator, vectors },
                );
                
                // Keep the index for search benchmarks
                index = try abi.HnswIndex.init(allocator, .{}, .cosine);
                for (vectors) |v| { _ = try index.insert(v); }
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
                        fn bench(idx: *abi.HnswIndex, qs: [][]const f32, val_k: usize) !void {
                            for (qs) |q| {
                                const results = try idx.search(q, val_k, 64);
                                allocator.free(results);
                            }
                        }
                    }.bench,
                    .{ &index, queries, k },
                );
            }
        }
    }

    runner.printSummaryDebug();
}
