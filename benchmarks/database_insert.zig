//! Database insertion benchmarks.
//! Tests bulk insertion performance with various configurations.

const std = @import("std");
const Database = @import("../src/database/database.zig").Database;
const DatabaseConfig = @import("../src/database/database.zig").DatabaseConfig;
const batch = @import("../src/database/batch.zig");

pub const VECTOR_DIM = 128;
pub const VECTOR_COUNT = 10_000;

/// Run all insertion benchmarks.
pub fn run(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n=== Insertion Benchmarks ===\n\n", .{});

    try benchmarkStandardInsert(allocator, vectors);
    try benchmarkCachedNormsInsert(allocator, vectors);
    try benchmarkBatchInsert(allocator, vectors);
}

fn benchmarkStandardInsert(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("--- Standard Bulk Insertion ---\n", .{});

    var db = try Database.init(allocator, "bench_standard");
    defer db.deinit();

    var timer = try std.time.Timer.start();
    for (vectors, 0..) |vec, i| {
        try db.insert(@intCast(i), vec, null);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const rate = @as(f64, @floatFromInt(vectors.len)) / (elapsed_ms / 1000.0);

    std.debug.print("  Inserted {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{
        vectors.len,
        elapsed_ms,
        rate,
    });
}

fn benchmarkCachedNormsInsert(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n--- Insertion with Cached Norms ---\n", .{});

    var db = try Database.initWithConfig(allocator, "bench_cached", .{
        .cache_norms = true,
        .initial_capacity = vectors.len,
    });
    defer db.deinit();

    var timer = try std.time.Timer.start();
    for (vectors, 0..) |vec, i| {
        try db.insert(@intCast(i), vec, null);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const rate = @as(f64, @floatFromInt(vectors.len)) / (elapsed_ms / 1000.0);

    std.debug.print("  Inserted {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{
        vectors.len,
        elapsed_ms,
        rate,
    });
}

fn benchmarkBatchInsert(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n--- Batch Insert ---\n", .{});

    var processor = batch.BatchProcessor.init(allocator, .{
        .batch_size = 1000,
        .parallel_workers = 4,
        .prefetch_distance = 4,
    });
    defer processor.deinit();

    // Create batch records
    const records = try allocator.alloc(batch.BatchRecord, vectors.len);
    defer allocator.free(records);
    for (vectors, 0..) |vec, i| {
        records[i] = .{ .id = @intCast(i), .vector = vec };
    }

    var timer = try std.time.Timer.start();
    const result = try processor.insertBatch(records);
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    std.debug.print("  Batch: {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{
        result.successful,
        elapsed_ms,
        result.throughput,
    });
}
