const std = @import("std");
const benchmark = @import("mod.zig");
const abi = @import("abi");

// Framework initialization benchmark
fn frameworkInitBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_gpu = false });
    defer abi.shutdown(&framework);
    std.mem.doNotOptimizeAway(&framework);
}

// Database benchmarks
fn databaseInsertBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_database = true, .enable_gpu = false });
    defer abi.shutdown(&framework);

    var db_handle = try abi.wdbx.createDatabase(allocator, "bench");
    defer abi.wdbx.closeDatabase(&db_handle);

    const vector = [_]f32{ 1.0, 0.5, 0.2, 0.8 };
    try abi.wdbx.insertVector(&db_handle, 1, &vector, null);
}

fn databaseSearchBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_database = true, .enable_gpu = false });
    defer abi.shutdown(&framework);

    var db_handle = try abi.wdbx.createDatabase(allocator, "bench");
    defer abi.wdbx.closeDatabase(&db_handle);

    // Insert some test data
    const vectors = [_][4]f32{
        [_]f32{ 1.0, 0.0, 0.0, 0.0 },
        [_]f32{ 0.0, 1.0, 0.0, 0.0 },
        [_]f32{ 0.0, 0.0, 1.0, 0.0 },
        [_]f32{ 0.0, 0.0, 0.0, 1.0 },
    };

    for (vectors, 0..) |vec, i| {
        try abi.wdbx.insertVector(&db_handle, @intCast(i + 1), &vec, null);
    }

    // Perform search
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try abi.wdbx.searchVectors(&db_handle, allocator, &query, 3);
    defer allocator.free(results);
    std.mem.doNotOptimizeAway(results);
}

// Compute benchmarks
fn computeTaskBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_gpu = false });
    defer abi.shutdown(&framework);

    // Simple compute benchmark using SIMD operations
    var sum: f32 = 0.0;
    var i: usize = 0;
    while (i < 10000) : (i += 1) {
        const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const vec_b = [_]f32{ @floatFromInt(@mod(i, 10)), @floatFromInt(@mod(i + 1, 10)), @floatFromInt(@mod(i + 2, 10)), @floatFromInt(@mod(i + 3, 10)) };
        sum += abi.vectorDot(&vec_a, &vec_b);
    }
    std.mem.doNotOptimizeAway(sum);
}

// SIMD benchmarks
fn simdVectorBenchmark(_: std.mem.Allocator) !void {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    const result = abi.vectorDot(&vec_a, &vec_b);
    std.mem.doNotOptimizeAway(result);
}

fn simdAddBenchmark(_: std.mem.Allocator) !void {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var result = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    abi.vectorAdd(&vec_a, &vec_b, &result);
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
    var framework = abi.init(allocator, abi.FrameworkOptions{ .enable_gpu = true }) catch |err| {
        std.debug.print("GPU initialization failed: {}\n", .{err});
        std.debug.print("GPU unavailable for benchmarking\n", .{});
        std.mem.doNotOptimizeAway(@as(bool, false));
        return;
    };
    defer abi.shutdown(&framework);

    const available = abi.gpu.moduleEnabled();
    std.mem.doNotOptimizeAway(available);
}

// Network benchmarks (if available)
fn networkRegistryBenchmark(allocator: std.mem.Allocator) !void {
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_network = true, .enable_gpu = false });
    defer abi.shutdown(&framework);

    const registry = try abi.network.defaultRegistry();
    const node_id = "bench-node";
    try registry.register(node_id, "127.0.0.1:8080");
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
    var framework = try abi.init(allocator, abi.FrameworkOptions{ .enable_gpu = false });
    defer abi.shutdown(&framework);

    // Measure format string preparation without stdout I/O
    var buffer: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(&buffer, "[info] benchmark test: {d}", .{@as(u64, 42)}) catch "error";
    std.mem.doNotOptimizeAway(msg.ptr);
}

// Configuration benchmark
fn configBenchmark(allocator: std.mem.Allocator) !void {
    // Simple config validation benchmark
    var config = abi.config.Config.init(allocator);
    defer config.deinit();
    try config.validate();
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
