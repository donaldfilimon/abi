//! Performance Optimization Benchmark Suite
//!
//! This benchmark suite measures the performance improvements from our optimizations:
//! - Vector search performance
//! - Memory allocation patterns
//! - SIMD operation efficiency
//! - Cache performance

const std = @import("std");
const abi = @import("abi");
const VectorOps = abi.shared.simd_optimized.VectorOps;
const MatrixOps = abi.shared.simd_optimized.MatrixOps;
const MemoryOps = abi.shared.simd_optimized.MemoryOps;

const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_ns: u64,
    avg_time_ns: u64,
    operations_per_second: f64,
    memory_used: usize,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 ABI Performance Optimization Benchmark Suite", .{});
    std.log.info("================================================", .{});

    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();

    // Vector operations benchmarks
    try benchmarkVectorOperations(allocator, &results);
    
    // Matrix operations benchmarks
    try benchmarkMatrixOperations(allocator, &results);
    
    // Memory operations benchmarks
    try benchmarkMemoryOperations(allocator, &results);
    
    // Database operations benchmarks
    try benchmarkDatabaseOperations(allocator, &results);
    
    // Print results
    try printResults(results.items);
}

fn benchmarkVectorOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    std.log.info("📊 Benchmarking Vector Operations...", .{});
    
    const vector_size = 1000;
    const iterations = 10000;
    
    // Create test vectors
    const a = try allocator.alloc(f32, vector_size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, vector_size);
    defer allocator.free(b);
    
    // Initialize with random data
    var prng = std.rand.DefaultPrng.init(0x12345678);
    const random = prng.random();
    for (a) |*val| val.* = random.float(f32);
    for (b) |*val| val.* = random.float(f32);
    
    // Benchmark dot product
    const start_time = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = VectorOps.dotProduct(a, b);
    }
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    
    try results.append(BenchmarkResult{
        .name = "Vector Dot Product (SIMD)",
        .iterations = iterations,
        .total_time_ns = @intCast(total_time),
        .avg_time_ns = @intCast(total_time / iterations),
        .operations_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9),
        .memory_used = vector_size * @sizeOf(f32) * 2,
    });
    
    // Benchmark cosine similarity
    const start_time2 = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = VectorOps.cosineSimilarity(a, b);
    }
    const end_time2 = std.time.nanoTimestamp();
    const total_time2 = end_time2 - start_time2;
    
    try results.append(BenchmarkResult{
        .name = "Cosine Similarity (SIMD)",
        .iterations = iterations,
        .total_time_ns = @intCast(total_time2),
        .avg_time_ns = @intCast(total_time2 / iterations),
        .operations_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time2)) / 1e9),
        .memory_used = vector_size * @sizeOf(f32) * 2,
    });
}

fn benchmarkMatrixOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    std.log.info("📊 Benchmarking Matrix Operations...", .{});
    
    const m = 64;
    const n = 64;
    const p = 64;
    const iterations = 1000;
    
    // Create test matrices
    const a = try allocator.alloc(f32, m * n);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n * p);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * p);
    defer allocator.free(c);
    
    // Initialize with random data
    var prng = std.rand.DefaultPrng.init(0x12345678);
    const random = prng.random();
    for (a) |*val| val.* = random.float(f32);
    for (b) |*val| val.* = random.float(f32);
    
    // Benchmark matrix multiplication
    const start_time = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        MatrixOps.multiply(a, b, c, m, n, p);
    }
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    
    try results.append(BenchmarkResult{
        .name = "Matrix Multiplication (Optimized)",
        .iterations = iterations,
        .total_time_ns = @intCast(total_time),
        .avg_time_ns = @intCast(total_time / iterations),
        .operations_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9),
        .memory_used = (m * n + n * p + m * p) * @sizeOf(f32),
    });
}

fn benchmarkMemoryOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    std.log.info("📊 Benchmarking Memory Operations...", .{});
    
    const buffer_size = 1024 * 1024; // 1MB
    const iterations = 10000;
    
    // Create test buffers
    const src = try allocator.alloc(u8, buffer_size);
    defer allocator.free(src);
    const dest = try allocator.alloc(u8, buffer_size);
    defer allocator.free(dest);
    
    // Initialize source with pattern
    for (src, 0..) |*val, i| val.* = @intCast(i % 256);
    
    // Benchmark memory copy
    const start_time = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        MemoryOps.fastCopy(dest, src);
    }
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    
    try results.append(BenchmarkResult{
        .name = "Memory Copy (Optimized)",
        .iterations = iterations,
        .total_time_ns = @intCast(total_time),
        .avg_time_ns = @intCast(total_time / iterations),
        .operations_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9),
        .memory_used = buffer_size * 2,
    });
    
    // Benchmark memory set
    const start_time2 = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        MemoryOps.fastSet(dest, 0x42);
    }
    const end_time2 = std.time.nanoTimestamp();
    const total_time2 = end_time2 - start_time2;
    
    try results.append(BenchmarkResult{
        .name = "Memory Set (Optimized)",
        .iterations = iterations,
        .total_time_ns = @intCast(total_time2),
        .avg_time_ns = @intCast(total_time2 / iterations),
        .operations_per_second = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time2)) / 1e9),
        .memory_used = buffer_size,
    });
}

fn benchmarkDatabaseOperations(allocator: std.mem.Allocator, results: *std.ArrayList(BenchmarkResult)) !void {
    std.log.info("📊 Benchmarking Database Operations...", .{});
    
    const vector_count = 1000;
    const vector_dim = 128;
    const search_iterations = 1000;
    
    // Create test vectors
    var vectors = std.ArrayList([]f32).init(allocator);
    defer {
        for (vectors.items) |vec| allocator.free(vec);
        vectors.deinit();
    }
    
    var prng = std.rand.DefaultPrng.init(0x12345678);
    const random = prng.random();
    
    for (0..vector_count) |_| {
        const vec = try allocator.alloc(f32, vector_dim);
        for (vec) |*val| val.* = random.float(f32);
        try vectors.append(vec);
    }
    
    // Create query vector
    const query = try allocator.alloc(f32, vector_dim);
    defer allocator.free(query);
    for (query) |*val| val.* = random.float(f32);
    
    // Benchmark vector search
    const start_time = std.time.nanoTimestamp();
    for (0..search_iterations) |_| {
        var best_similarity: f32 = 0.0;
        var best_idx: usize = 0;
        
        for (vectors.items, 0..) |vec, idx| {
            const similarity = VectorOps.cosineSimilarity(query, vec);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_idx = idx;
            }
        }
        _ = best_idx; // Prevent optimization
    }
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    
    try results.append(BenchmarkResult{
        .name = "Vector Search (Optimized)",
        .iterations = search_iterations,
        .total_time_ns = @intCast(total_time),
        .avg_time_ns = @intCast(total_time / search_iterations),
        .operations_per_second = @as(f64, @floatFromInt(search_iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9),
        .memory_used = vector_count * vector_dim * @sizeOf(f32),
    });
}

fn printResults(results: []const BenchmarkResult) !void {
    std.log.info("\n📈 Benchmark Results", .{});
    std.log.info("===================", .{});
    
    for (results) |result| {
        std.log.info("\n{s}:", .{result.name});
        std.log.info("  Iterations: {d}", .{result.iterations});
        std.log.info("  Total Time: {d:.2} ms", .{@as(f64, @floatFromInt(result.total_time_ns)) / 1e6});
        std.log.info("  Average Time: {d:.2} ns", .{result.avg_time_ns});
        std.log.info("  Operations/sec: {d:.0}", .{result.operations_per_second});
        std.log.info("  Memory Used: {d:.2} MB", .{@as(f64, @floatFromInt(result.memory_used)) / (1024 * 1024)});
    }
    
    // Calculate overall performance score
    var total_ops_per_sec: f64 = 0.0;
    for (results) |result| {
        total_ops_per_sec += result.operations_per_second;
    }
    const avg_ops_per_sec = total_ops_per_sec / @as(f64, @floatFromInt(results.len));
    
    std.log.info("\n🎯 Overall Performance Score: {d:.0} ops/sec", .{avg_ops_per_sec});
    
    if (avg_ops_per_sec > 1_000_000) {
        std.log.info("✅ Excellent performance!", .{});
    } else if (avg_ops_per_sec > 100_000) {
        std.log.info("✅ Good performance!", .{});
    } else {
        std.log.info("⚠️  Performance could be improved", .{});
    }
}