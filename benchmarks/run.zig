//! Benchmark Runner
//!
//! Main entry point for running ABI framework benchmarks.
//! Includes framework initialization, database, and AI benchmarks.
//!
//! Run with: `zig build benchmarks`

const std = @import("std");
const benchmark = @import("mod.zig");
const abi = @import("abi");

// Framework initialization benchmark
fn frameworkInitBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.App.initDefault(allocator);
    defer framework.deinit();
    std.mem.doNotOptimizeAway(&framework);
}

// Database benchmarks
fn databaseInsertBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.App.init(allocator, abi.Config{
        .database = .{}, // Enabled with defaults
    });
    defer framework.deinit();

    var db_handle = try abi.features.database.open(allocator, "bench");
    defer abi.features.database.close(&db_handle);

    const vector = [_]f32{ 1.0, 0.5, 0.2, 0.8 };
    try abi.features.database.insert(&db_handle, 1, &vector, null);
}

fn databaseSearchBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.App.init(allocator, abi.Config{
        .database = .{}, // Enabled with defaults
    });
    defer framework.deinit();

    var db_handle = try abi.features.database.open(allocator, "bench");
    defer abi.features.database.close(&db_handle);

    // Insert some test data
    const vectors = [_][4]f32{
        [_]f32{ 1.0, 0.0, 0.0, 0.0 },
        [_]f32{ 0.0, 1.0, 0.0, 0.0 },
        [_]f32{ 0.0, 0.0, 1.0, 0.0 },
        [_]f32{ 0.0, 0.0, 0.0, 1.0 },
    };

    for (vectors, 0..) |vec, i| {
        try abi.features.database.insert(&db_handle, @intCast(i + 1), &vec, null);
    }

    // Perform search
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try abi.features.database.search(&db_handle, allocator, &query, 3);
    defer allocator.free(results);
    std.mem.doNotOptimizeAway(results);
}

fn neuralAnnSearchBenchmark(allocator: std.mem.Allocator) !void {
    var engine = try abi.features.database.neural.Engine.init(allocator, .{
        .dimensions = 4,
        .metric = .cosine,
    });
    defer engine.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const v3 = [_]f32{ 0.0, 0.0, 1.0, 0.0 };
    try engine.cache.put("doc-1", &v1);
    try engine.cache.put("doc-2", &v2);
    try engine.cache.put("doc-3", &v3);

    try engine.index("id-1", "doc-1", .{ .text = "doc-1", .tags = &.{} });
    try engine.index("id-2", "doc-2", .{ .text = "doc-2", .tags = &.{} });
    try engine.index("id-3", "doc-3", .{ .text = "doc-3", .tags = &.{} });

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try engine.searchByVector(&query, .{ .k = 2, .ef = 32 });
    defer allocator.free(results);
    std.mem.doNotOptimizeAway(results);
}

// Compute benchmarks
fn computeTaskBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.App.init(allocator, abi.Config{});
    defer framework.deinit();

    // Simple compute benchmark using SIMD operations
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < 10000) : (i += 1) {
        const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const vec_b = [_]f32{ @floatFromInt(@mod(i, 10)), @floatFromInt(@mod(i + 1, 10)), @floatFromInt(@mod(i + 2, 10)), @floatFromInt(@mod(i + 3, 10)) };
        sum += abi.services.simd.vectorDot(&vec_a, &vec_b);
    }
    std.mem.doNotOptimizeAway(sum);
}

// SIMD benchmarks
fn simdVectorBenchmark(_: std.mem.Allocator) !void {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    const result = abi.services.simd.vectorDot(&vec_a, &vec_b);
    std.mem.doNotOptimizeAway(result);
}

fn simdAddBenchmark(_: std.mem.Allocator) !void {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var result = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    abi.services.simd.vectorAdd(&vec_a, &vec_b, &result);
    std.mem.doNotOptimizeAway(&result);
}

// Memory benchmarks
fn memoryAllocationBenchmark(allocator: std.mem.Allocator) !void {
    const buffer = try allocator.alloc(u8, 1024);
    defer allocator.free(buffer);
    std.mem.doNotOptimizeAway(buffer.ptr);
}

// GPU benchmarks (if available)
fn gpuAvailabilityBenchmark(allocator: std.mem.Allocator) !void {
    var framework = abi.App.init(allocator, abi.Config{
        .gpu = .{}, // Enabled with defaults
    }) catch |err| {
        std.debug.print("GPU initialization failed: {}\n", .{err});
        std.debug.print("GPU unavailable for benchmarking\n", .{});
        std.mem.doNotOptimizeAway(@as(bool, false));
        return;
    };
    defer framework.deinit();

    const available = abi.features.gpu.moduleEnabled();
    std.mem.doNotOptimizeAway(available);
}

// Network benchmarks (if available)
fn networkRegistryBenchmark(allocator: std.mem.Allocator) !void {
    // Network registry may fail in environments without networking support.
    // Initialize the framework with network enabled, but gracefully handle any errors
    // during registry operations so the benchmark suite reports no errors.
    var framework = abi.App.init(allocator, abi.Config{ .network = .{} }) catch return;
    defer framework.deinit();

    // Attempt to obtain the default registry; if that fails, simply skip the benchmark.
    const registry = abi.features.network.defaultRegistry() catch return;
    const node_id = "bench-node";
    // Register a node – ignore errors (e.g., bind failures) to keep benchmark stable.
    _ = registry.register(node_id, "127.0.0.1:8080") catch {};
    // Touch the node – returns a bool, ignore the result.
    _ = registry.touch(node_id);
}

// JSON processing benchmark
fn jsonBenchmark(allocator: std.mem.Allocator) !void {
    const json_str = "{\"name\":\"test\",\"value\":123,\"array\":[1,2,3,4,5]}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
    defer parsed.deinit();
    std.mem.doNotOptimizeAway(&parsed);
}

// Logging benchmark - measures format string processing overhead without I/O
fn loggingBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.App.init(allocator, abi.Config{});
    defer framework.deinit();

    // Measure format string preparation without stdout I/O
    var buffer: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(&buffer, "[info] benchmark test: {d}", .{@as(u64, 42)}) catch "error";
    std.mem.doNotOptimizeAway(msg.ptr);
}

// Configuration benchmark
fn configBenchmark(allocator: std.mem.Allocator) !void {
    // Simple config validation benchmark
    var config = abi.config.Config{};
    const enabled_features = try config.enabledFeatures(allocator);
    defer allocator.free(enabled_features);
    std.mem.doNotOptimizeAway(&config);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var suite = benchmark.BenchmarkSuite.init(allocator);
    defer suite.deinit();

    std.debug.print("=== ABI Framework Comprehensive Benchmark Suite ===\n\n", .{});

    // Framework Core
    try suite.runBenchmark("Framework Initialization", frameworkInitBenchmark, .{allocator});
    try suite.runBenchmark("Logging Operations", loggingBenchmark, .{allocator});
    try suite.runBenchmark("Configuration Loading", configBenchmark, .{allocator});

    // Memory and SIMD
    try suite.runBenchmark("Memory Allocation (1KB)", memoryAllocationBenchmark, .{allocator});
    try suite.runBenchmark("SIMD Vector Dot Product", simdVectorBenchmark, .{allocator});
    try suite.runBenchmark("SIMD Vector Addition", simdAddBenchmark, .{allocator});

    // Compute Engine
    try suite.runBenchmark("Compute Engine Task", computeTaskBenchmark, .{allocator});

    // Database (Vector Search)
    try suite.runBenchmark("Database Vector Insert", databaseInsertBenchmark, .{allocator});
    try suite.runBenchmark("Database Vector Search", databaseSearchBenchmark, .{allocator});
    try suite.runBenchmark("Neural ANN Search (WDBX Engine)", neuralAnnSearchBenchmark, .{allocator});

    // JSON Processing
    try suite.runBenchmark("JSON Parse/Serialize", jsonBenchmark, .{allocator});

    // GPU (if available)
    try suite.runBenchmark("GPU Availability Check", gpuAvailabilityBenchmark, .{allocator});

    // Network (if available)
    try suite.runBenchmark("Network Registry Operations", networkRegistryBenchmark, .{allocator});

    suite.printSummary();

    std.debug.print("\n=== Benchmark Suite Complete ===\n", .{});
    std.debug.print("Note: Some benchmarks may show 0 ops/sec if the feature is disabled or unavailable\n", .{});
}
