//! Search Benchmarks
//!
//! Performance measurement for the full-text search module:
//! - Indexing throughput (short / medium / long documents)
//! - Query throughput (BM25 search on varying corpus sizes)
//! - Mixed workload (80% query, 15% index, 5% delete)

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

pub const SearchBenchConfig = struct {
    doc_sizes: []const usize = &.{ 64, 256, 1024, 4096 },
    corpus_sizes: []const usize = &.{ 100, 1000 },
    mixed_op_counts: []const usize = &.{ 100, 1000, 10_000 },
};

// ── Helpers ──────────────────────────────────────────────────────────

fn generateDocId(buf: *[32]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "doc-{d:0>8}", .{i}) catch return "doc-00000000";
    return len;
}

fn generateContent(allocator: std.mem.Allocator, size: usize) ![]u8 {
    const words = [_][]const u8{
        "the",   "quick",    "brown",     "fox",   "jumps",
        "over",  "lazy",     "dog",       "hello", "world",
        "zig",   "search",   "benchmark", "fast",  "index",
        "query", "document", "engine",    "text",  "score",
    };

    var content = try allocator.alloc(u8, size);
    var pos: usize = 0;
    var prng = std.Random.DefaultPrng.init(size);
    const rand = prng.random();

    while (pos < size) {
        if (pos > 0 and pos < size) {
            content[pos] = ' ';
            pos += 1;
        }
        const word = words[rand.intRangeLessThan(usize, 0, words.len)];
        const remaining = size - pos;
        const copy_len = @min(word.len, remaining);
        @memcpy(content[pos..][0..copy_len], word[0..copy_len]);
        pos += copy_len;
    }

    return content;
}

fn freeQueryResults(allocator: std.mem.Allocator, results: []abi.features.search.SearchResult) void {
    for (results) |r| {
        if (r.doc_id.len > 0) allocator.free(r.doc_id);
        if (r.snippet.len > 0) allocator.free(r.snippet);
    }
    allocator.free(results);
}

// ── Indexing Throughput ──────────────────────────────────────────────

fn benchIndexing(allocator: std.mem.Allocator, count: usize, doc_size: usize) !void {
    const search = abi.features.search;
    try search.init(allocator, .{ .default_result_limit = 10 });
    defer search.deinit();

    _ = try search.createIndex(allocator, "bench");

    const content = try generateContent(allocator, doc_size);
    defer allocator.free(content);

    var id_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const doc_id = generateDocId(&id_buf, i);
        try search.indexDocument("bench", doc_id, content);
    }
}

// ── Query Throughput (pre-populated corpus) ─────────────────────────

fn benchQuery(allocator: std.mem.Allocator, corpus_size: usize, query_count: usize) !void {
    const search = abi.features.search;
    try search.init(allocator, .{ .default_result_limit = 10 });
    defer search.deinit();

    _ = try search.createIndex(allocator, "bench");

    // Pre-populate the corpus
    var id_buf: [32]u8 = undefined;
    for (0..corpus_size) |i| {
        const doc_id = generateDocId(&id_buf, i);
        const content = try generateContent(allocator, 256);
        defer allocator.free(content);
        try search.indexDocument("bench", doc_id, content);
    }

    // Run queries
    const queries = [_][]const u8{
        "quick brown fox",
        "hello world",
        "zig benchmark",
        "search engine",
        "fast index query",
    };

    for (0..query_count) |i| {
        const q = queries[i % queries.len];
        const results = try search.query(allocator, "bench", q);
        defer freeQueryResults(allocator, results);
        std.mem.doNotOptimizeAway(&results);
    }
}

// ── Mixed Workload (80% query, 15% index, 5% delete) ───────────────

fn benchMixed(allocator: std.mem.Allocator, op_count: usize) !void {
    const search = abi.features.search;
    try search.init(allocator, .{ .default_result_limit = 10 });
    defer search.deinit();

    _ = try search.createIndex(allocator, "bench");

    // Seed the index with 200 documents
    const seed_count: usize = 200;
    var id_buf: [32]u8 = undefined;
    for (0..seed_count) |i| {
        const doc_id = generateDocId(&id_buf, i);
        const content = try generateContent(allocator, 256);
        defer allocator.free(content);
        try search.indexDocument("bench", doc_id, content);
    }

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var next_doc_id: usize = seed_count;

    const queries = [_][]const u8{
        "quick brown fox",
        "hello world",
        "zig benchmark",
        "search engine",
        "fast index query",
    };

    for (0..op_count) |_| {
        const roll = rand.float(f32);

        if (roll < 0.80) {
            // 80% query
            const q = queries[rand.intRangeLessThan(usize, 0, queries.len)];
            const results = search.query(allocator, "bench", q) catch &.{};
            if (results.len > 0) {
                freeQueryResults(allocator, @constCast(results));
            }
        } else if (roll < 0.95) {
            // 15% index
            const doc_id = generateDocId(&id_buf, next_doc_id);
            const content = try generateContent(allocator, 256);
            defer allocator.free(content);
            search.indexDocument("bench", doc_id, content) catch {};
            next_doc_id += 1;
        } else {
            // 5% delete
            const del_id = rand.intRangeLessThan(usize, 0, next_doc_id);
            const doc_id = generateDocId(&id_buf, del_id);
            _ = search.deleteDocument("bench", doc_id) catch false;
        }
    }
}

// ── Runner ───────────────────────────────────────────────────────────

pub fn runSearchBenchmarks(allocator: std.mem.Allocator, config: SearchBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                         SEARCH BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Indexing throughput by document size
    std.debug.print("[Indexing Throughput]\n", .{});
    for (config.doc_sizes) |dsize| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "index_{d}B", .{dsize}) catch "index";

        const result = try runner.run(
            .{
                .name = name,
                .category = "search/index",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
                .bytes_per_op = @intCast(dsize),
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize, ds: usize) !void {
                    try benchIndexing(a, c, ds);
                }
            }.bench,
            .{ allocator, 500, dsize },
        );
        std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} MB/s\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.throughputMBps(@intCast(dsize)),
        });
    }

    // Query throughput by corpus size
    std.debug.print("\n[Query Throughput]\n", .{});
    for (config.corpus_sizes) |corpus| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "query_{d}_docs",
            .{corpus},
        ) catch "query";

        const result = try runner.run(
            .{
                .name = name,
                .category = "search/query",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, cs: usize) !void {
                    try benchQuery(a, cs, 50);
                }
            }.bench,
            .{ allocator, corpus },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Mixed workload
    std.debug.print("\n[Mixed Workload (80/15/5 query/index/delete)]\n", .{});
    for (config.mixed_op_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "mixed_{d}", .{count}) catch "mixed";

        const result = try runner.run(
            .{
                .name = name,
                .category = "search/mixed",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchMixed(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runSearchBenchmarks(allocator, .{});
}

test "search benchmarks compile" {
    const allocator = std.testing.allocator;
    try benchIndexing(allocator, 10, 64);
    try benchQuery(allocator, 10, 5);
    try benchMixed(allocator, 20);
}
