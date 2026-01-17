//! Database memory and cache benchmarks.
//! Tests cache-aligned data access and memory usage.

const std = @import("std");
const HotVectorData = @import("../src/database/database.zig").HotVectorData;
const Database = @import("../src/database/database.zig").Database;

/// Run all memory benchmarks.
pub fn run(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n=== Memory & Cache Benchmarks ===\n\n", .{});

    try benchmarkCacheAlignedAccess(allocator, vectors);
    try benchmarkMemoryUsage(allocator, vectors);
    try benchmarkPrefetching(allocator, vectors);
}

fn benchmarkCacheAlignedAccess(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("--- Cache-Aligned Hot Data Access ---\n", .{});

    if (vectors.len == 0) return;
    const dim = vectors[0].len;

    var hot_data = try HotVectorData.init(allocator, dim, vectors.len);
    defer hot_data.deinit(allocator);

    // Populate with vectors
    for (vectors) |vec| {
        var norm: f32 = 0.0;
        for (vec) |v| norm += v * v;
        norm = @sqrt(norm);
        try hot_data.append(vec, norm);
    }

    // Benchmark sequential access without prefetch
    var timer = try std.time.Timer.start();
    var sum: f32 = 0.0;
    for (0..hot_data.count) |i| {
        const vec = hot_data.getVector(i);
        sum += vec[0];
    }
    const no_prefetch_ns = timer.read();

    // Benchmark sequential access with prefetch
    timer = try std.time.Timer.start();
    var sum_prefetch: f32 = 0.0;
    for (0..hot_data.count) |i| {
        // Prefetch next vector
        if (i + 4 < hot_data.count) {
            hot_data.prefetch(i + 4);
        }
        const vec = hot_data.getVector(i);
        sum_prefetch += vec[0];
    }
    const with_prefetch_ns = timer.read();

    const no_prefetch_us = @as(f64, @floatFromInt(no_prefetch_ns)) / 1000.0;
    const with_prefetch_us = @as(f64, @floatFromInt(with_prefetch_ns)) / 1000.0;

    std.debug.print("  Without prefetch: {d:.1}us\n", .{no_prefetch_us});
    std.debug.print("  With prefetch: {d:.1}us\n", .{with_prefetch_us});
    std.debug.print("  Speedup: {d:.2}x\n", .{no_prefetch_us / with_prefetch_us});
    std.debug.print("  (Checksum: {d:.4}, {d:.4})\n", .{ sum, sum_prefetch });
}

fn benchmarkMemoryUsage(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n--- Memory Usage Analysis ---\n", .{});

    if (vectors.len == 0) return;
    const dim = vectors[0].len;
    const count = vectors.len;

    // Calculate memory breakdown
    const vector_bytes = count * dim * @sizeOf(f32);
    const norm_bytes = count * @sizeOf(f32);
    const id_bytes = count * @sizeOf(u64);
    const overhead_per_record = @sizeOf(std.ArrayListUnmanaged(u8).Slice);

    std.debug.print("  Vector data: {d} x {d} dims = {d} bytes ({d:.2} MB)\n", .{
        count,
        dim,
        vector_bytes,
        @as(f64, @floatFromInt(vector_bytes)) / (1024 * 1024),
    });
    std.debug.print("  Cached norms: {d} x {d} = {d} bytes\n", .{
        count,
        @sizeOf(f32),
        norm_bytes,
    });
    std.debug.print("  IDs: {d} x {d} = {d} bytes\n", .{
        count,
        @sizeOf(u64),
        id_bytes,
    });

    const total = vector_bytes + norm_bytes + id_bytes;
    const norm_overhead_pct = @as(f64, @floatFromInt(norm_bytes)) / @as(f64, @floatFromInt(vector_bytes)) * 100.0;

    std.debug.print("  Total: {d} bytes ({d:.2} MB)\n", .{
        total,
        @as(f64, @floatFromInt(total)) / (1024 * 1024),
    });
    std.debug.print("  Norm cache overhead: {d:.2}%\n", .{norm_overhead_pct});

    _ = allocator;
    _ = overhead_per_record;
}

fn benchmarkPrefetching(allocator: std.mem.Allocator, vectors: []const []const f32) !void {
    std.debug.print("\n--- Prefetch Distance Analysis ---\n", .{});

    if (vectors.len == 0) return;
    const dim = vectors[0].len;

    var hot_data = try HotVectorData.init(allocator, dim, vectors.len);
    defer hot_data.deinit(allocator);

    for (vectors) |vec| {
        var norm: f32 = 0.0;
        for (vec) |v| norm += v * v;
        norm = @sqrt(norm);
        try hot_data.append(vec, norm);
    }

    // Test different prefetch distances
    const distances = [_]usize{ 1, 2, 4, 8, 16 };

    for (distances) |distance| {
        var timer = try std.time.Timer.start();
        var sum: f32 = 0.0;

        for (0..hot_data.count) |i| {
            if (i + distance < hot_data.count) {
                hot_data.prefetch(i + distance);
            }
            const vec = hot_data.getVector(i);
            sum += vec[0];
        }

        const elapsed_us = @as(f64, @floatFromInt(timer.read())) / 1000.0;
        std.debug.print("  Distance {d:2}: {d:.1}us (sum={d:.4})\n", .{ distance, elapsed_us, sum });
    }
}
