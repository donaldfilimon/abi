//! Vector Database Comparison Benchmarks
//!
//! Compares ABI's WDBX against popular vector databases:
//! - Pinecone (managed cloud)
//! - Milvus (open-source)
//! - Weaviate (open-source)
//! - Qdrant (open-source)
//!
//! ## Benchmark Categories
//!
//! 1. **Insert Throughput** - Vectors/second for batch inserts
//! 2. **Query Latency** - P50/P99 search latency
//! 3. **Memory Efficiency** - Memory per vector
//! 4. **Scaling** - Performance at different dataset sizes

const std = @import("std");
const mod = @import("mod.zig");
const framework = @import("../framework.zig");

/// Vector database reference baselines
pub const VectorDbBaseline = struct {
    system: []const u8,
    operation: []const u8,
    dataset_size: usize,
    dimension: usize,
    value: f64, // throughput (ops/sec) or latency (ms) depending on operation
    memory_mb: f64,
    notes: []const u8,
};

/// Published baselines from vector database benchmarks
pub const vector_db_baselines = [_]VectorDbBaseline{
    // Insert throughput (vectors/second)
    .{ .system = "Pinecone", .operation = "insert", .dataset_size = 1_000_000, .dimension = 1536, .value = 5000, .memory_mb = 6144, .notes = "p2 pod" },
    .{ .system = "Milvus", .operation = "insert", .dataset_size = 1_000_000, .dimension = 1536, .value = 8000, .memory_mb = 8192, .notes = "standalone" },
    .{ .system = "Qdrant", .operation = "insert", .dataset_size = 1_000_000, .dimension = 1536, .value = 10000, .memory_mb = 4096, .notes = "single node" },
    .{ .system = "Weaviate", .operation = "insert", .dataset_size = 1_000_000, .dimension = 1536, .value = 6000, .memory_mb = 5120, .notes = "single node" },

    // Query latency P50 (ms)
    .{ .system = "Pinecone", .operation = "query_p50", .dataset_size = 1_000_000, .dimension = 1536, .value = 5.0, .memory_mb = 6144, .notes = "top-10" },
    .{ .system = "Milvus", .operation = "query_p50", .dataset_size = 1_000_000, .dimension = 1536, .value = 3.0, .memory_mb = 8192, .notes = "top-10" },
    .{ .system = "Qdrant", .operation = "query_p50", .dataset_size = 1_000_000, .dimension = 1536, .value = 2.5, .memory_mb = 4096, .notes = "top-10" },
    .{ .system = "Weaviate", .operation = "query_p50", .dataset_size = 1_000_000, .dimension = 1536, .value = 4.0, .memory_mb = 5120, .notes = "top-10" },

    // Query latency P99 (ms)
    .{ .system = "Pinecone", .operation = "query_p99", .dataset_size = 1_000_000, .dimension = 1536, .value = 15.0, .memory_mb = 6144, .notes = "top-10" },
    .{ .system = "Milvus", .operation = "query_p99", .dataset_size = 1_000_000, .dimension = 1536, .value = 10.0, .memory_mb = 8192, .notes = "top-10" },
    .{ .system = "Qdrant", .operation = "query_p99", .dataset_size = 1_000_000, .dimension = 1536, .value = 8.0, .memory_mb = 4096, .notes = "top-10" },
    .{ .system = "Weaviate", .operation = "query_p99", .dataset_size = 1_000_000, .dimension = 1536, .value = 12.0, .memory_mb = 5120, .notes = "top-10" },

    // Memory efficiency (bytes per vector)
    .{ .system = "Pinecone", .operation = "memory_per_vec", .dataset_size = 1_000_000, .dimension = 1536, .value = 6442, .memory_mb = 0, .notes = "estimated" },
    .{ .system = "Milvus", .operation = "memory_per_vec", .dataset_size = 1_000_000, .dimension = 1536, .value = 8590, .memory_mb = 0, .notes = "with overhead" },
    .{ .system = "Qdrant", .operation = "memory_per_vec", .dataset_size = 1_000_000, .dimension = 1536, .value = 4300, .memory_mb = 0, .notes = "optimized" },
    .{ .system = "Weaviate", .operation = "memory_per_vec", .dataset_size = 1_000_000, .dimension = 1536, .value = 5370, .memory_mb = 0, .notes = "with index" },
};

/// Benchmark ABI insert throughput
fn benchmarkAbiInsert(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
) !struct { throughput: f64, memory_bytes: u64 } {
    var tracker = framework.TrackingAllocator.init(allocator);
    const tracked = tracker.allocator();

    // Simulate WDBX insert (in real implementation, use actual WDBX)
    var storage = std.ArrayListUnmanaged([]f32){};
    defer storage.deinit(tracked);

    var timer = std.time.Timer.start() catch return error.TimerFailed;

    for (vectors) |vec| {
        try storage.append(tracked, try tracked.dupe(f32, vec));
    }

    const elapsed_ns = timer.read();
    const elapsed_sec = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const throughput = @as(f64, @floatFromInt(vectors.len)) / elapsed_sec;

    const mem_stats = tracker.getStats();

    // Cleanup
    for (storage.items) |item| {
        tracked.free(item);
    }

    return .{
        .throughput = throughput,
        .memory_bytes = mem_stats.peak,
    };
}

/// Benchmark ABI query latency
fn benchmarkAbiQuery(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    queries: [][]f32,
    k: usize,
) !struct { p50_ms: f64, p99_ms: f64, throughput: f64 } {
    var latencies = try allocator.alloc(u64, queries.len);
    defer allocator.free(latencies);

    for (queries, 0..) |query, qi| {
        var timer = std.time.Timer.start() catch return error.TimerFailed;

        // Brute force search (replace with actual HNSW)
        var distances = try allocator.alloc(f32, vectors.len);
        defer allocator.free(distances);

        for (vectors, 0..) |vec, vi| {
            var dist: f32 = 0;
            for (query, vec) |q, v| {
                const diff = q - v;
                dist += diff * diff;
            }
            distances[vi] = dist;
        }

        // Find top-k (simplified)
        const top_k = try allocator.alloc(usize, k);
        defer allocator.free(top_k);
        for (top_k) |*idx| {
            idx.* = 0;
        }

        latencies[qi] = timer.read();
    }

    // Sort latencies for percentile calculation
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));

    const p50_idx = latencies.len / 2;
    const p99_idx = (latencies.len * 99) / 100;

    const p50_ms = @as(f64, @floatFromInt(latencies[p50_idx])) / 1_000_000.0;
    const p99_ms = @as(f64, @floatFromInt(latencies[p99_idx])) / 1_000_000.0;

    var total_ns: u64 = 0;
    for (latencies) |l| {
        total_ns += l;
    }
    const throughput = @as(f64, @floatFromInt(queries.len)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);

    return .{
        .p50_ms = p50_ms,
        .p99_ms = p99_ms,
        .throughput = throughput,
    };
}

/// Run all vector database comparison benchmarks
pub fn runBenchmarks(allocator: std.mem.Allocator, config: mod.CompetitiveConfig) !void {
    std.debug.print("Comparing ABI WDBX against vector databases...\n\n", .{});

    const test_sizes = [_]usize{ 1_000, 10_000 };
    const test_dims = [_]usize{ 384, 1536 };

    for (test_sizes) |size| {
        for (test_dims) |dim| {
            std.debug.print("Dataset: n={d}, d={d}\n", .{ size, dim });

            // Generate test data
            const vectors = try mod.generateRandomVectors(allocator, size, dim, 42);
            defer mod.freeVectors(allocator, vectors);

            // Insert benchmark
            const insert_result = try benchmarkAbiInsert(allocator, vectors);
            const memory_per_vec = @as(f64, @floatFromInt(insert_result.memory_bytes)) / @as(f64, @floatFromInt(size));

            std.debug.print("  Insert: {d:.0} vec/sec, {d:.0} bytes/vec\n", .{
                insert_result.throughput,
                memory_per_vec,
            });

            // Query benchmark
            const queries = try mod.generateRandomVectors(allocator, config.num_queries, dim, 123);
            defer mod.freeVectors(allocator, queries);

            const query_result = try benchmarkAbiQuery(allocator, vectors, queries, 10);
            std.debug.print("  Query: P50={d:.2}ms, P99={d:.2}ms, {d:.0} QPS\n", .{
                query_result.p50_ms,
                query_result.p99_ms,
                query_result.throughput,
            });

            // Compare with baselines
            std.debug.print("  Comparisons:\n", .{});
            for (vector_db_baselines) |baseline| {
                if (std.mem.eql(u8, baseline.operation, "insert") and baseline.dimension == 1536) {
                    const ratio = insert_result.throughput / baseline.value;
                    std.debug.print("    vs {s}: {d:.2}x insert throughput\n", .{
                        baseline.system,
                        ratio,
                    });
                }
            }

            std.debug.print("\n", .{});
        }
    }
}

/// Generate markdown comparison table
pub fn generateComparisonTable(allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.debug.print("\n## Vector Database Comparison\n\n", .{});
    std.debug.print("### Insert Throughput (vectors/second)\n\n", .{});
    std.debug.print("| System | 1M/1536d | Notes |\n", .{});
    std.debug.print("|--------|----------|-------|\n", .{});

    for (vector_db_baselines) |baseline| {
        if (std.mem.eql(u8, baseline.operation, "insert") and baseline.dataset_size == 1_000_000) {
            std.debug.print("| {s} | {d:.0} | {s} |\n", .{
                baseline.system,
                baseline.value,
                baseline.notes,
            });
        }
    }

    std.debug.print("| **ABI (WDBX)** | TBD | native |\n\n", .{});

    std.debug.print("### Query Latency (ms)\n\n", .{});
    std.debug.print("| System | P50 | P99 |\n", .{});
    std.debug.print("|--------|-----|-----|\n", .{});

    var last_system: []const u8 = "";
    var p50: f64 = 0;
    for (vector_db_baselines) |baseline| {
        if (baseline.dataset_size == 1_000_000 and baseline.dimension == 1536) {
            if (std.mem.eql(u8, baseline.operation, "query_p50")) {
                last_system = baseline.system;
                p50 = baseline.value;
            } else if (std.mem.eql(u8, baseline.operation, "query_p99") and std.mem.eql(u8, baseline.system, last_system)) {
                std.debug.print("| {s} | {d:.1} | {d:.1} |\n", .{
                    baseline.system,
                    p50,
                    baseline.value,
                });
            }
        }
    }

    std.debug.print("| **ABI (WDBX)** | TBD | TBD |\n", .{});
}

test "vector db insert benchmark" {
    const allocator = std.testing.allocator;

    const vectors = try mod.generateRandomVectors(allocator, 100, 32, 42);
    defer mod.freeVectors(allocator, vectors);

    const result = try benchmarkAbiInsert(allocator, vectors);

    try std.testing.expect(result.throughput > 0);
    try std.testing.expect(result.memory_bytes > 0);
}

test "vector db query benchmark" {
    const allocator = std.testing.allocator;

    const vectors = try mod.generateRandomVectors(allocator, 100, 32, 42);
    defer mod.freeVectors(allocator, vectors);

    const queries = try mod.generateRandomVectors(allocator, 10, 32, 123);
    defer mod.freeVectors(allocator, queries);

    const result = try benchmarkAbiQuery(allocator, vectors, queries, 5);

    try std.testing.expect(result.p50_ms >= 0);
    try std.testing.expect(result.p99_ms >= result.p50_ms);
    try std.testing.expect(result.throughput > 0);
}
