//! Competitive Benchmark Suite
//!
//! Compares ABI performance against industry-standard solutions:
//! - FAISS (Facebook AI Similarity Search)
//! - Pinecone/Milvus (Vector Database comparison)
//! - LLM Inference (token generation speed)
//!
//! ## Running Benchmarks
//!
//! ```bash
//! zig build bench-competitive
//! ```
//!
//! ## Output Formats
//!
//! Results are output in JSON format for easy integration with
//! CI/CD pipelines and visualization tools.

const std = @import("std");
const framework = @import("../system/framework.zig");

pub const faiss = @import("faiss_comparison.zig");
pub const vector_db = @import("vector_db_comparison.zig");
pub const llm = @import("llm_comparison.zig");

/// Competitive benchmark configuration
pub const CompetitiveConfig = struct {
    /// Dataset sizes to test
    dataset_sizes: []const usize = &.{ 1_000, 10_000, 100_000, 1_000_000 },
    /// Vector dimensions to test
    dimensions: []const usize = &.{ 128, 384, 768, 1536 },
    /// Number of queries per benchmark
    num_queries: usize = 1000,
    /// Top-K values for search benchmarks
    top_k_values: []const usize = &.{ 1, 10, 100 },
    /// Output format
    output_format: OutputFormat = .json,
    /// Enable memory profiling
    profile_memory: bool = true,
    /// Enable throughput scaling tests
    scaling_tests: bool = true,
};

pub const OutputFormat = enum {
    json,
    csv,
    markdown,
};

/// Competitive benchmark result
pub const CompetitiveResult = struct {
    system: []const u8,
    operation: []const u8,
    dataset_size: usize,
    dimension: usize,
    top_k: usize,
    latency_ms: f64,
    throughput_ops_sec: f64,
    recall_at_k: f64,
    memory_mb: f64,

    pub fn format(
        self: CompetitiveResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("{s} | {s} | n={d} d={d} k={d} | {d:.2}ms | {d:.0} ops/s | recall={d:.3}", .{
            self.system,
            self.operation,
            self.dataset_size,
            self.dimension,
            self.top_k,
            self.latency_ms,
            self.throughput_ops_sec,
            self.recall_at_k,
        });
    }
};

/// Comparison summary across systems
pub const ComparisonSummary = struct {
    operation: []const u8,
    abi_result: CompetitiveResult,
    competitor_results: []const CompetitiveResult,
    abi_speedup: f64, // vs best competitor
    abi_memory_ratio: f64, // vs best competitor

    pub fn printMarkdown(self: ComparisonSummary, writer: anytype) !void {
        try writer.writeAll("| System | Latency (ms) | Throughput | Recall@K | Memory (MB) |\n");
        try writer.writeAll("|--------|-------------|------------|----------|-------------|\n");

        // ABI row
        try writer.print("| **ABI** | {d:.2} | {d:.0} | {d:.3} | {d:.1} |\n", .{
            self.abi_result.latency_ms,
            self.abi_result.throughput_ops_sec,
            self.abi_result.recall_at_k,
            self.abi_result.memory_mb,
        });

        // Competitor rows
        for (self.competitor_results) |r| {
            try writer.print("| {s} | {d:.2} | {d:.0} | {d:.3} | {d:.1} |\n", .{
                r.system,
                r.latency_ms,
                r.throughput_ops_sec,
                r.recall_at_k,
                r.memory_mb,
            });
        }

        try writer.print("\n**ABI Speedup:** {d:.1}x faster\n", .{self.abi_speedup});
        try writer.print("**Memory Efficiency:** {d:.1}x less memory\n", .{1.0 / self.abi_memory_ratio});
    }
};

/// Run all competitive benchmarks
pub fn runAllBenchmarks(allocator: std.mem.Allocator, config: CompetitiveConfig, runner: *framework.BenchmarkRunner) !void {
    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                     ABI COMPETITIVE BENCHMARK SUITE\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Vector similarity search benchmarks
    std.debug.print("[Vector Search Benchmarks]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});
    try faiss.runBenchmarks(allocator, config, runner);

    // Vector database benchmarks
    std.debug.print("\n[Vector Database Benchmarks]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});
    try vector_db.runBenchmarks(allocator, config, runner);

    // LLM inference benchmarks
    std.debug.print("\n[LLM Inference Benchmarks]\n", .{});
    std.debug.print("-" ** 80 ++ "\n", .{});
    try llm.runBenchmarks(allocator, config, runner);

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                     BENCHMARK COMPLETE\n", .{});
    std.debug.print("================================================================================\n", .{});
}

/// Generate random vectors for benchmarking
pub fn generateRandomVectors(
    allocator: std.mem.Allocator,
    count: usize,
    dimension: usize,
    seed: u64,
) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const vectors = try allocator.alloc([]f32, count);
    errdefer {
        for (vectors) |v| {
            allocator.free(v);
        }
        allocator.free(vectors);
    }

    for (vectors, 0..) |*vec, i| {
        vec.* = try allocator.alloc(f32, dimension);
        for (vec.*) |*val| {
            val.* = random.float(f32) * 2.0 - 1.0;
        }
        // Normalize
        var norm: f32 = 0;
        for (vec.*) |val| {
            norm += val * val;
        }
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*val| {
                val.* /= norm;
            }
        }
        _ = i;
    }

    return vectors;
}

/// Free generated vectors
pub fn freeVectors(allocator: std.mem.Allocator, vectors: [][]f32) void {
    for (vectors) |v| {
        allocator.free(v);
    }
    allocator.free(vectors);
}

/// Calculate recall@k between ground truth and results
pub fn calculateRecall(
    ground_truth: []const usize,
    results: []const usize,
    k: usize,
) f64 {
    const limit = @min(k, @min(ground_truth.len, results.len));
    var matches: usize = 0;

    for (results[0..limit]) |r| {
        for (ground_truth[0..limit]) |gt| {
            if (r == gt) {
                matches += 1;
                break;
            }
        }
    }

    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(limit));
}

test "generate random vectors" {
    const allocator = std.testing.allocator;
    const vectors = try generateRandomVectors(allocator, 100, 128, 42);
    defer freeVectors(allocator, vectors);

    try std.testing.expectEqual(@as(usize, 100), vectors.len);
    try std.testing.expectEqual(@as(usize, 128), vectors[0].len);

    // Check normalization
    var norm: f32 = 0;
    for (vectors[0]) |v| {
        norm += v * v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(norm), 0.001);
}

test "calculate recall" {
    const gt = [_]usize{ 1, 2, 3, 4, 5 };
    const results1 = [_]usize{ 1, 2, 3, 4, 5 };
    const results2 = [_]usize{ 1, 2, 6, 7, 8 };

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), calculateRecall(&gt, &results1, 5), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), calculateRecall(&gt, &results2, 5), 0.001);
}
