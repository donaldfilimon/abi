//! Benchmark suite for database performance improvements.
//! Executes various workloads and reports timings.

const std = @import("std");
const Bench = std.benchmark;
const Database = @import("../src/features/database/database.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var db = try Database.init(allocator);
    defer db.deinit();

    const vectors = try allocator.alloc([]f32, 1000);
    defer allocator.free(vectors);
    // Populate with dummy data.
    var rng = std.rand.DefaultPrng.init(0);
    const rand = rng.random();
    for (vectors) |*vec| {
        const dim = 128;
        const arr = try allocator.alloc(f32, dim);
        for (arr) |*v| v.* = rand.float(f32);
        vec.* = arr;
    }

    // Benchmark bulk insertion.
    const ins = Bench.start("bulk_insert", .{});
    for (vectors, 0..) |vec, i| {
        try db.insert(@intCast(i), vec, null);
    }
    ins.stop();

    // Benchmark a single search.
    const query = vectors[0];
    const search = Bench.start("search_one", .{});
    var results = try allocator.alloc(Database.SearchResult, 10);
    defer allocator.free(results);
    _ = try db.search(query, 10, results);
    search.stop();
}

