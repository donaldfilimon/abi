//! Performance Monitor Tool
//!
//! This tool provides real-time performance monitoring and analysis for the ABI framework.

const std = @import("std");
const abi = @import("abi");

const PerformanceMonitor = abi.shared.performance_config.PerformanceMonitor;
const PlatformOptimizations = abi.shared.performance_config.PlatformOptimizations;
const OptimizationLevel = abi.shared.performance_config.OptimizationLevel;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🔍 ABI Performance Monitor", .{});
    std.log.info("=========================", .{});

    var monitor = PerformanceMonitor.init();
    var config = abi.shared.performance_config.getPlatformOptimizations(.production);
    
    // Run performance tests
    try runPerformanceTests(allocator, &monitor);
    
    // Print performance statistics
    try printPerformanceStats(&monitor);
    
    // Print optimization recommendations
    try printOptimizationRecommendations(&monitor, config);
}

fn runPerformanceTests(allocator: std.mem.Allocator, monitor: *PerformanceMonitor) !void {
    std.log.info("Running performance tests...", .{});
    
    // Test 1: Vector operations
    try testVectorOperations(allocator, monitor);
    
    // Test 2: Memory allocation patterns
    try testMemoryAllocation(allocator, monitor);
    
    // Test 3: Database operations
    try testDatabaseOperations(allocator, monitor);
}

fn testVectorOperations(allocator: std.mem.Allocator, monitor: *PerformanceMonitor) !void {
    const vector_size = 1000;
    const iterations = 10000;
    
    const a = try allocator.alloc(f32, vector_size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, vector_size);
    defer allocator.free(b);
    
    // Initialize vectors
    var prng = std.rand.DefaultPrng.init(0x12345678);
    const random = prng.random();
    for (a) |*val| val.* = random.float(f32);
    for (b) |*val| val.* = random.float(f32);
    
    const start_time = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = abi.VectorOps.dotProduct(a, b);
    }
    const end_time = std.time.nanoTimestamp();
    
    const avg_time = (end_time - start_time) / iterations;
    monitor.updateOperationTime(@intCast(avg_time));
    
    monitor.recordAllocation(vector_size * @sizeOf(f32) * 2);
}

fn testMemoryAllocation(allocator: std.mem.Allocator, monitor: *PerformanceMonitor) !void {
    const allocation_count = 1000;
    const allocation_size = 1024;
    
    var allocations = std.ArrayList([]u8).init(allocator);
    defer {
        for (allocations.items) |allocation| {
            allocator.free(allocation);
        }
        allocations.deinit();
    }
    
    const start_time = std.time.nanoTimestamp();
    for (0..allocation_count) |_| {
        const allocation = try allocator.alloc(u8, allocation_size);
        try allocations.append(allocation);
        monitor.recordAllocation(allocation_size);
    }
    const end_time = std.time.nanoTimestamp();
    
    const avg_time = (end_time - start_time) / allocation_count;
    monitor.updateOperationTime(@intCast(avg_time));
}

fn testDatabaseOperations(allocator: std.mem.Allocator, monitor: *PerformanceMonitor) !void {
    // Simulate database operations
    const operation_count = 1000;
    
    const start_time = std.time.nanoTimestamp();
    for (0..operation_count) |_| {
        // Simulate vector search operation
        const query = try allocator.alloc(f32, 128);
        defer allocator.free(query);
        
        var prng = std.rand.DefaultPrng.init(0x12345678);
        const random = prng.random();
        for (query) |*val| val.* = random.float(f32);
        
        // Simulate similarity calculation
        const target = try allocator.alloc(f32, 128);
        defer allocator.free(target);
        for (target) |*val| val.* = random.float(f32);
        
        _ = abi.VectorOps.cosineSimilarity(query, target);
        monitor.recordAllocation(128 * @sizeOf(f32) * 2);
    }
    const end_time = std.time.nanoTimestamp();
    
    const avg_time = (end_time - start_time) / operation_count;
    monitor.updateOperationTime(@intCast(avg_time));
}

fn printPerformanceStats(monitor: *PerformanceMonitor) !void {
    const stats = monitor.getStats();
    
    std.log.info("\n📊 Performance Statistics", .{});
    std.log.info("=========================", .{});
    std.log.info("Total Allocations: {d}", .{stats.allocation_count});
    std.log.info("Total Memory Used: {d:.2} MB", .{@as(f64, @floatFromInt(stats.total_allocated)) / (1024 * 1024)});
    std.log.info("Cache Hit Rate: {d:.1}%", .{stats.cache_hit_rate * 100});
    std.log.info("Average Operation Time: {d:.2} ns", .{stats.avg_operation_time});
    
    // Calculate performance score
    const performance_score = calculatePerformanceScore(stats);
    std.log.info("Performance Score: {d:.1}/100", .{performance_score});
    
    if (performance_score >= 80) {
        std.log.info("✅ Excellent performance!", .{});
    } else if (performance_score >= 60) {
        std.log.info("✅ Good performance", .{});
    } else if (performance_score >= 40) {
        std.log.info("⚠️  Average performance", .{});
    } else {
        std.log.info("❌ Poor performance - optimization needed", .{});
    }
}

fn calculatePerformanceScore(stats: anytype) f32 {
    var score: f32 = 100.0;
    
    // Penalize high allocation count
    if (stats.allocation_count > 10000) {
        score -= 20.0;
    } else if (stats.allocation_count > 5000) {
        score -= 10.0;
    }
    
    // Penalize high memory usage
    const memory_mb = @as(f64, @floatFromInt(stats.total_allocated)) / (1024 * 1024);
    if (memory_mb > 100) {
        score -= 20.0;
    } else if (memory_mb > 50) {
        score -= 10.0;
    }
    
    // Penalize low cache hit rate
    if (stats.cache_hit_rate < 0.5) {
        score -= 15.0;
    } else if (stats.cache_hit_rate < 0.7) {
        score -= 10.0;
    }
    
    // Penalize slow operations
    if (stats.avg_operation_time > 1000) {
        score -= 25.0;
    } else if (stats.avg_operation_time > 500) {
        score -= 15.0;
    } else if (stats.avg_operation_time > 100) {
        score -= 5.0;
    }
    
    return @max(0.0, score);
}

fn printOptimizationRecommendations(monitor: *PerformanceMonitor, config: PlatformOptimizations) !void {
    const stats = monitor.getStats();
    
    std.log.info("\n💡 Optimization Recommendations", .{});
    std.log.info("=================================", .{});
    
    if (stats.allocation_count > 5000) {
        std.log.info("• Consider using memory pools to reduce allocation overhead", .{});
    }
    
    if (stats.cache_hit_rate < 0.7) {
        std.log.info("• Enable cache optimization features", .{});
    }
    
    if (stats.avg_operation_time > 500) {
        std.log.info("• Enable SIMD optimizations for better performance", .{});
    }
    
    if (!config.enable_memory_pooling) {
        std.log.info("• Enable memory pooling for better memory management", .{});
    }
    
    if (!config.enable_simd) {
        std.log.info("• Enable SIMD operations for vector calculations", .{});
    }
    
    if (!config.enable_batch_ops) {
        std.log.info("• Enable batch operations for better throughput", .{});
    }
    
    if (!config.enable_parallel) {
        std.log.info("• Enable parallel processing for multi-core utilization", .{});
    }
    
    std.log.info("\n🔧 Current Configuration:", .{});
    std.log.info("  SIMD: {s}", .{if (config.enable_simd) "enabled" else "disabled"});
    std.log.info("  Vectorization: {s}", .{if (config.enable_vectorization) "enabled" else "disabled"});
    std.log.info("  Memory Pooling: {s}", .{if (config.enable_memory_pooling) "enabled" else "disabled"});
    std.log.info("  Batch Operations: {s}", .{if (config.enable_batch_ops) "enabled" else "disabled"});
    std.log.info("  Cache Optimization: {s}", .{if (config.enable_cache_optimization) "enabled" else "disabled"});
    std.log.info("  Parallel Processing: {s}", .{if (config.enable_parallel) "enabled" else "disabled"});
}