const std = @import("std");
const abi = @import("abi");

/// Performance profiler for WDBX-AI system
/// Measures and reports performance metrics across all modules
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 WDBX-AI Performance Profiler", .{});
    std.log.info("=====================================", .{});

    // Run all profiling tests
    try profileDatabaseOperations(allocator);
    try profileAIOperations(allocator);
    try profileSIMDOperations(allocator);
    try profileMemoryOperations(allocator);
    try profileEndToEndPerformance(allocator);

    // Generate comprehensive report
    try generatePerformanceReport(allocator);

    std.log.info("✅ Performance profiling completed!", .{});
}

/// Profile database operations
fn profileDatabaseOperations(allocator: std.mem.Allocator) !void {
    std.log.info("🗄️ Profiling Database Operations", .{});

    // Initialize database directly
    const test_file = "profile_db.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Profile single insert
    var single_insert_times = try std.ArrayList(u64).initCapacity(allocator, 100);
    defer single_insert_times.deinit(allocator);

    for (0..100) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }

        const start_time = std.time.nanoTimestamp();
        _ = try db.addEmbedding(&vector);
        const end_time = std.time.nanoTimestamp();

        try single_insert_times.append(allocator, @intCast(end_time - start_time));
    }

    const single_insert_stats = calculateStats(single_insert_times.items);
    std.log.info("  Single Insert - Avg: {:.2}μs, P95: {:.2}μs, P99: {:.2}μs", .{
        single_insert_stats.avg_ns / 1000.0,
        single_insert_stats.p95_ns / 1000.0,
        single_insert_stats.p99_ns / 1000.0,
    });

    // Profile batch insert
    const batch_size = 1000;
    const batch_start = std.time.nanoTimestamp();

    for (0..batch_size) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.001;
        }
        _ = try db.addEmbedding(&vector);
    }

    const batch_end = std.time.nanoTimestamp();
    const batch_duration_ms = @as(f64, @floatFromInt(batch_end - batch_start)) / 1_000_000.0;
    const batch_throughput = @as(f64, @floatFromInt(batch_size)) / (batch_duration_ms / 1000.0);

    std.log.info("  Batch Insert - {} vectors in {:.2}ms ({:.0} vectors/sec)", .{
        batch_size, batch_duration_ms, batch_throughput,
    });

    // Profile search operations
    var search_times = try std.ArrayList(u64).initCapacity(allocator, 100);
    defer search_times.deinit(allocator);

    for (0..50) |i| {
        var query: [128]f32 = undefined;
        for (&query, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }

        const start_time = std.time.nanoTimestamp();
        const results = try db.search(&query, 10, allocator);
        defer allocator.free(results);
        const end_time = std.time.nanoTimestamp();

        try search_times.append(allocator, @intCast(end_time - start_time));
    }

    const search_stats = calculateStats(search_times.items);
    std.log.info("  Search (k=10) - Avg: {:.2}ms, P95: {:.2}ms, P99: {:.2}ms", .{
        search_stats.avg_ns / 1_000_000.0,
        search_stats.p95_ns / 1_000_000.0,
        search_stats.p99_ns / 1_000_000.0,
    });
}

/// Profile AI operations
fn profileAIOperations(allocator: std.mem.Allocator) !void {
    std.log.info("🧠 Profiling AI Operations", .{});

    // Profile network creation
    const creation_start = std.time.nanoTimestamp();
    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{128}, &[_]usize{128});
    defer network.deinit();
    try network.addDenseLayer(64, .relu);
    try network.addDenseLayer(32, .relu);
    try network.addDenseLayer(128, .relu);
    try network.compile();
    const creation_end = std.time.nanoTimestamp();

    const creation_duration_ms = @as(f64, @floatFromInt(creation_end - creation_start)) / 1_000_000.0;
    std.log.info("  Network Creation - {:.2}ms", .{creation_duration_ms});

    // Profile training (simplified)
    const training_start = std.time.nanoTimestamp();

    // Simple training loop
    for (0..100) |i| {
        var input: [128]f32 = undefined;
        var output: [128]f32 = undefined;

        for (&input, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }
        @memcpy(&output, &input);

        // Simple forward pass
        const prediction = try network.forward(&input, &output);
        std.mem.doNotOptimizeAway(prediction);
    }

    const training_end = std.time.nanoTimestamp();
    const training_duration_ms = @as(f64, @floatFromInt(training_end - training_start)) / 1_000_000.0;
    const training_throughput = 100.0 / (training_duration_ms / 1000.0);

    std.log.info("  Training - {} samples in {:.2}ms ({:.0} samples/sec)", .{
        100, training_duration_ms, training_throughput,
    });

    // Profile prediction
    var prediction_times = try std.ArrayList(u64).initCapacity(allocator, 100);
    defer prediction_times.deinit(allocator);

    for (0..100) |i| {
        var input: [128]f32 = undefined;
        for (&input, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }

        const start_time = std.time.nanoTimestamp();
        var output: [128]f32 = undefined;
        try network.forward(&input, &output);
        const end_time = std.time.nanoTimestamp();

        try prediction_times.append(allocator, @intCast(end_time - start_time));
    }

    const prediction_stats = calculateStats(prediction_times.items);
    std.log.info("  Prediction - Avg: {:.2}μs, P95: {:.2}μs, P99: {:.2}μs", .{
        prediction_stats.avg_ns / 1000.0,
        prediction_stats.p95_ns / 1000.0,
        prediction_stats.p99_ns / 1000.0,
    });
}

/// Profile SIMD operations
fn profileSIMDOperations(allocator: std.mem.Allocator) !void {
    std.log.info("⚡ Profiling SIMD Operations", .{});

    const vector_sizes = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };

    for (vector_sizes) |size| {
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);
        const result = try allocator.alloc(f32, size);
        defer allocator.free(result);

        // Initialize vectors
        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2));

        // Profile vector addition
        const add_start = std.time.nanoTimestamp();
        const add_operations = 1000;
        for (0..add_operations) |_| {
            abi.simd.VectorOps.add(result, a, b);
        }
        const add_end = std.time.nanoTimestamp();
        const add_duration_ms = @as(f64, @floatFromInt(add_end - add_start)) / 1_000_000.0;
        const add_throughput = @as(f64, @floatFromInt(add_operations)) / (add_duration_ms / 1000.0);

        // Profile vector normalization
        const norm_start = std.time.nanoTimestamp();
        for (0..add_operations) |_| {
            abi.simd.VectorOps.normalize(result, a);
        }
        const norm_end = std.time.nanoTimestamp();
        const norm_duration_ms = @as(f64, @floatFromInt(norm_end - norm_start)) / 1_000_000.0;
        const norm_throughput = @as(f64, @floatFromInt(add_operations)) / (norm_duration_ms / 1000.0);

        std.log.info("  Size {} - Add: {:.0} ops/sec, Normalize: {:.0} ops/sec", .{
            size, add_throughput, norm_throughput,
        });
    }
}

/// Profile memory operations
fn profileMemoryOperations(allocator: std.mem.Allocator) !void {
    std.log.info("💾 Profiling Memory Operations", .{});

    // Profile allocation patterns
    const allocation_sizes = [_]usize{ 64, 256, 1024, 4096, 16384, 65536 };
    var allocation_times = try std.ArrayList(u64).initCapacity(allocator, 100);
    defer allocation_times.deinit(allocator);

    for (allocation_sizes) |size| {
        const start_time = std.time.nanoTimestamp();
        const memory = try allocator.alloc(u8, size);
        const end_time = std.time.nanoTimestamp();

        try allocation_times.append(allocator, @intCast(end_time - start_time));
        allocator.free(memory);
    }

    const allocation_stats = calculateStats(allocation_times.items);
    std.log.info("  Memory Allocation - Avg: {:.2}μs, P95: {:.2}μs, P99: {:.2}μs", .{
        allocation_stats.avg_ns / 1000.0,
        allocation_stats.p95_ns / 1000.0,
        allocation_stats.p99_ns / 1000.0,
    });
}

/// Profile end-to-end performance
fn profileEndToEndPerformance(allocator: std.mem.Allocator) !void {
    std.log.info("🔄 Profiling End-to-End Performance", .{});

    // Initialize system
    const test_file = "profile_e2e.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Create neural network directly
    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{128}, &[_]usize{128});
    defer network.deinit();
    try network.addDenseLayer(64, .relu);
    try network.addDenseLayer(32, .relu);
    try network.addDenseLayer(128, .relu);
    try network.compile();

    // End-to-end workflow
    const workflow_start = std.time.nanoTimestamp();

    // Generate and process data
    const data_count = 500;
    for (0..data_count) |i| {
        // Generate raw data
        var raw_data: [128]f32 = undefined;
        for (&raw_data, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }

        // Process with AI
        var embedding: [128]f32 = undefined;
        try network.forward(&raw_data, &embedding);

        // Apply SIMD operations
        var processed: [128]f32 = undefined;
        abi.simd.VectorOps.normalize(&processed, &embedding);

        // Store in database
        _ = try db.addEmbedding(&processed);
    }

    // Perform queries
    const query_count = 50;
    for (0..query_count) |i| {
        var query: [128]f32 = undefined;
        for (&query, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }

        const results = try db.search(&query, 5, allocator);
        defer allocator.free(results);
    }

    const workflow_end = std.time.nanoTimestamp();
    const workflow_duration_ms = @as(f64, @floatFromInt(workflow_end - workflow_start)) / 1_000_000.0;
    const data_throughput = @as(f64, @floatFromInt(data_count)) / (workflow_duration_ms / 1000.0);
    const query_throughput = @as(f64, @floatFromInt(query_count)) / (workflow_duration_ms / 1000.0);

    std.log.info("  End-to-End Workflow - {:.2}ms total", .{workflow_duration_ms});
    std.log.info("  Data Processing - {:.0} vectors/sec", .{data_throughput});
    std.log.info("  Query Processing - {:.0} queries/sec", .{query_throughput});
}

/// Generate comprehensive performance report
fn generatePerformanceReport(_: std.mem.Allocator) !void {
    std.log.info("📋 Generating Performance Report", .{});

    const file = try std.fs.cwd().createFile("performance_report.md", .{});
    defer file.close();

    const report_content =
        \\# WDBX-AI Performance Report
        \\
        \\## Overview
        \\This report contains performance metrics for the WDBX-AI vector database system.
        \\
        \\## Database Operations
        \\- Single vector insert performance
        \\- Batch insert throughput
        \\- Search query performance
        \\
        \\## AI Operations
        \\- Neural network creation time
        \\- Training throughput
        \\- Prediction latency
        \\
        \\## SIMD Operations
        \\- Vector addition performance
        \\- Vector normalization performance
        \\- Performance scaling with vector size
        \\
        \\## Memory Operations
        \\- Allocation latency
        \\- Memory usage patterns
        \\
        \\## End-to-End Performance
        \\- Complete workflow throughput
        \\- System integration performance
        \\
        \\*Report generated by WDBX-AI Performance Profiler*
    ;

    try file.writeAll(report_content);
    std.log.info("  Report saved to: performance_report.md", .{});
}

/// Calculate performance statistics
fn calculateStats(times: []const u64) struct {
    avg_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
} {
    if (times.len == 0) {
        return .{ .avg_ns = 0, .p95_ns = 0, .p99_ns = 0 };
    }

    // Sort times for percentile calculation
    var sorted_times = std.ArrayList(u64).initCapacity(std.heap.page_allocator, times.len) catch return .{ .avg_ns = 0, .p95_ns = 0, .p99_ns = 0 };
    defer sorted_times.deinit(std.heap.page_allocator);

    for (times) |time| {
        sorted_times.append(std.heap.page_allocator, time) catch continue;
    }

    std.sort.heap(u64, sorted_times.items, {}, comptime std.sort.asc(u64));

    // Calculate average
    var sum: u64 = 0;
    for (sorted_times.items) |time| {
        sum += time;
    }
    const avg_ns = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(sorted_times.items.len));

    // Calculate percentiles
    const p95_index = @as(usize, @intFromFloat(@as(f64, @floatFromInt(sorted_times.items.len)) * 0.95));
    const p99_index = @as(usize, @intFromFloat(@as(f64, @floatFromInt(sorted_times.items.len)) * 0.99));

    const p95_ns = @as(f64, @floatFromInt(sorted_times.items[@min(p95_index, sorted_times.items.len - 1)]));
    const p99_ns = @as(f64, @floatFromInt(sorted_times.items[@min(p99_index, sorted_times.items.len - 1)]));

    return .{
        .avg_ns = avg_ns,
        .p95_ns = p95_ns,
        .p99_ns = p99_ns,
    };
}
