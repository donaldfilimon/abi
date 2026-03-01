//! ANN-Benchmarks Compatible Suite
//!
//! Industry-standard benchmarks for approximate nearest neighbor search,
//! compatible with the ann-benchmarks.com format.
//!
//! ## Features
//!
//! - Recall vs QPS tradeoff analysis
//! - Multiple distance metrics
//! - Parameter sensitivity analysis
//! - Standard dataset configurations

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");
const hnsw = @import("hnsw.zig");
const operations = @import("operations.zig");

/// ANN-Benchmarks compatible result format
pub const AnnBenchmarkResult = struct {
    /// Algorithm name
    algorithm: []const u8,
    /// Dataset name
    dataset: []const u8,
    /// Recall value (0.0 to 1.0)
    recall: f64,
    /// Queries per second
    qps: f64,
    /// Build time in seconds
    build_time_sec: f64,
    /// Index size in bytes
    index_size_bytes: u64,
    /// Distance metric used
    distance: []const u8,
    /// Algorithm parameters (JSON string)
    parameters: []const u8,

    pub fn toJson(self: AnnBenchmarkResult, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, "{");
        try buf.writer(allocator).print(
            "\"algorithm\": \"{s}\", \"dataset\": \"{s}\", \"recall\": {d:.6}, \"qps\": {d:.2}, \"build_time\": {d:.4}, \"index_size\": {d}, \"distance\": \"{s}\", \"parameters\": {s}",
            .{
                self.algorithm,
                self.dataset,
                self.recall,
                self.qps,
                self.build_time_sec,
                self.index_size_bytes,
                self.distance,
                self.parameters,
            },
        );
        try buf.appendSlice(allocator, "}");

        return buf.toOwnedSlice(allocator);
    }
};

/// ANN-Benchmarks configuration
pub const AnnBenchConfig = struct {
    /// Dataset to benchmark
    dataset: core.config.AnnDataset = .sift_1m,
    /// Custom dataset size (if dataset == .custom)
    custom_size: usize = 100_000,
    /// Custom dimension (if dataset == .custom)
    custom_dimension: usize = 128,
    /// Number of queries
    num_queries: usize = 10_000,
    /// K values for recall calculation
    k_values: []const usize = &.{ 1, 10, 100 },
    /// Distance metric
    distance: core.Metric = .euclidean_sq,
    /// HNSW M parameter values to test
    hnsw_m_values: []const usize = &.{ 8, 16, 32, 48 },
    /// HNSW efConstruction values to test
    ef_construction_values: []const usize = &.{ 100, 200, 400 },
    /// HNSW efSearch values to test
    ef_search_values: []const usize = &.{ 10, 50, 100, 200, 400 },
};

/// Run ANN-Benchmarks compatible test suite
pub fn runAnnBenchmarks(
    allocator: std.mem.Allocator,
    config: AnnBenchConfig,
) ![]AnnBenchmarkResult {
    var results = std.ArrayListUnmanaged(AnnBenchmarkResult).empty;
    errdefer results.deinit(allocator);

    const dataset_size = if (config.dataset == .custom)
        config.custom_size
    else
        @min(config.dataset.size(), 100_000); // Limit for testing

    const dimension = if (config.dataset == .custom)
        config.custom_dimension
    else
        config.dataset.dimension();

    std.debug.print("\n=== ANN-Benchmarks Compatible Suite ===\n", .{});
    std.debug.print("Dataset: {s}, n={d}, d={d}\n\n", .{
        config.dataset.name(),
        dataset_size,
        dimension,
    });

    // Generate test data
    const vectors = try core.vectors.generateNormalized(allocator, dataset_size, dimension, 42);
    defer core.vectors.free(allocator, vectors);

    const queries = try core.vectors.generateNormalized(allocator, config.num_queries, dimension, 123);
    defer core.vectors.free(allocator, queries);

    // Compute ground truth using brute force
    std.debug.print("Computing ground truth...\n", .{});
    const ground_truth = try computeGroundTruth(allocator, vectors, queries, 100);
    defer {
        for (ground_truth) |gt| allocator.free(gt);
        allocator.free(ground_truth);
    }

    // Test different HNSW configurations
    for (config.hnsw_m_values) |m| {
        for (config.ef_construction_values) |ef_const| {
            std.debug.print("Testing M={d}, efConstruction={d}...\n", .{ m, ef_const });

            // Build index and measure time
            var build_timer = abi.shared.time.Timer.start() catch continue;
            var index = hnsw.EuclideanHNSW.init(allocator, m, ef_const);
            defer index.deinit();

            var prng = std.Random.DefaultPrng.init(12345);
            for (vectors, 0..) |vec, id| {
                try index.insert(vec, @intCast(id), prng.random());
            }
            const build_time_ns = build_timer.read();
            const build_time_sec = @as(f64, @floatFromInt(build_time_ns)) / 1_000_000_000.0;

            // Test different efSearch values
            for (config.ef_search_values) |ef_search| {
                for (config.k_values) |k| {
                    // Run queries and measure
                    var total_time_ns: u64 = 0;
                    var total_recall: f64 = 0;

                    for (queries, 0..) |query, qi| {
                        var query_timer = abi.shared.time.Timer.start() catch continue;
                        const search_results = try index.search(query, k, ef_search);
                        defer allocator.free(search_results);
                        total_time_ns += query_timer.read();

                        // Calculate recall
                        const recall = operations.calculateRecall(search_results, ground_truth[qi], k);
                        total_recall += recall;
                    }

                    const avg_recall = total_recall / @as(f64, @floatFromInt(queries.len));
                    const qps = @as(f64, @floatFromInt(queries.len)) /
                        (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

                    var params_buf: [128]u8 = undefined;
                    const params = std.fmt.bufPrint(&params_buf, "{{\"M\": {d}, \"efConstruction\": {d}, \"efSearch\": {d}}}", .{ m, ef_const, ef_search }) catch "";

                    try results.append(allocator, .{
                        .algorithm = "WDBX-HNSW",
                        .dataset = config.dataset.name(),
                        .recall = avg_recall,
                        .qps = qps,
                        .build_time_sec = build_time_sec,
                        .index_size_bytes = index.estimateMemoryUsage(),
                        .distance = "euclidean",
                        .parameters = params,
                    });

                    std.debug.print("  M={d} ef={d} k={d}: recall={d:.4}, QPS={d:.0}\n", .{
                        m,
                        ef_search,
                        k,
                        avg_recall,
                        qps,
                    });
                }
            }
        }
    }

    return results.toOwnedSlice(allocator);
}

/// Compute ground truth using brute force search
fn computeGroundTruth(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    queries: [][]f32,
    k: usize,
) ![][]u64 {
    const ground_truth = try allocator.alloc([]u64, queries.len);
    errdefer {
        for (ground_truth) |gt| allocator.free(gt);
        allocator.free(ground_truth);
    }

    for (queries, 0..) |query, qi| {
        ground_truth[qi] = try operations.bruteForceSearch(allocator, vectors, query, k, .euclidean_sq);
    }

    return ground_truth;
}

/// Generate recall-QPS tradeoff curve
pub fn generateRecallQpsCurve(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    queries: [][]f32,
    ground_truth: [][]u64,
    m: usize,
    ef_construction: usize,
    ef_search_values: []const usize,
    k: usize,
) ![]struct { recall: f64, qps: f64 } {
    var curve = std.ArrayListUnmanaged(struct { recall: f64, qps: f64 }).empty;
    errdefer curve.deinit(allocator);

    // Build index
    var index = hnsw.EuclideanHNSW.init(allocator, m, ef_construction);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    for (vectors, 0..) |vec, id| {
        try index.insert(vec, @intCast(id), prng.random());
    }

    // Test each ef_search value
    for (ef_search_values) |ef_search| {
        var total_time_ns: u64 = 0;
        var total_recall: f64 = 0;

        for (queries, 0..) |query, qi| {
            var timer = abi.shared.time.Timer.start() catch continue;
            const results = try index.search(query, k, ef_search);
            defer allocator.free(results);
            total_time_ns += timer.read();

            total_recall += operations.calculateRecall(results, ground_truth[qi], k);
        }

        const avg_recall = total_recall / @as(f64, @floatFromInt(queries.len));
        const qps = @as(f64, @floatFromInt(queries.len)) /
            (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

        try curve.append(allocator, .{ .recall = avg_recall, .qps = qps });
    }

    return curve.toOwnedSlice(allocator);
}

/// Print ANN-Benchmarks results in standard format
pub fn printResults(results: []const AnnBenchmarkResult) void {
    std.debug.print("\n=== ANN-Benchmarks Results ===\n\n", .{});
    std.debug.print("{s:<15} {s:<15} {s:>10} {s:>12} {s:>12} {s:>15}\n", .{
        "Algorithm",
        "Dataset",
        "Recall",
        "QPS",
        "Build (s)",
        "Index Size",
    });
    std.debug.print("{s:-<15} {s:-<15} {s:->10} {s:->12} {s:->12} {s:->15}\n", .{
        "",
        "",
        "",
        "",
        "",
        "",
    });

    for (results) |r| {
        std.debug.print("{s:<15} {s:<15} {d:>10.4} {d:>12.0} {d:>12.2} {d:>15}\n", .{
            r.algorithm,
            r.dataset,
            r.recall,
            r.qps,
            r.build_time_sec,
            r.index_size_bytes,
        });
    }
}

/// Export results to JSON file
pub fn exportResultsJson(allocator: std.mem.Allocator, results: []const AnnBenchmarkResult) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\n  \"results\": [\n");

    for (results, 0..) |r, i| {
        if (i > 0) try buf.appendSlice(allocator, ",\n");
        try buf.appendSlice(allocator, "    ");
        const json = try r.toJson(allocator);
        defer allocator.free(json);
        try buf.appendSlice(allocator, json);
    }

    try buf.appendSlice(allocator, "\n  ]\n}\n");

    return buf.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "ann benchmarks small" {
    const allocator = std.testing.allocator;

    const results = try runAnnBenchmarks(allocator, .{
        .dataset = .custom,
        .custom_size = 100,
        .custom_dimension = 32,
        .num_queries = 10,
        .k_values = &.{5},
        .hnsw_m_values = &.{8},
        .ef_construction_values = &.{50},
        .ef_search_values = &.{20},
    });
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expect(results[0].recall >= 0 and results[0].recall <= 1.0);
}

test "recall qps curve" {
    const allocator = std.testing.allocator;

    const vectors = try core.vectors.normalized(allocator, 100, 32);
    defer core.vectors.free(allocator, vectors);

    const queries = try core.vectors.normalized(allocator, 10, 32);
    defer core.vectors.free(allocator, queries);

    const ground_truth = try computeGroundTruth(allocator, vectors, queries, 10);
    defer {
        for (ground_truth) |gt| allocator.free(gt);
        allocator.free(ground_truth);
    }

    const curve = try generateRecallQpsCurve(
        allocator,
        vectors,
        queries,
        ground_truth,
        8,
        50,
        &.{ 10, 20, 50 },
        10,
    );
    defer allocator.free(curve);

    try std.testing.expect(curve.len == 3);
}
