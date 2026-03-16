//! Storage Benchmarks
//!
//! Performance measurement for the unified object storage module:
//! - Put throughput (various value sizes: 64B, 1KB, 64KB)
//! - Get throughput (pre-populated keys)
//! - Delete throughput
//! - Exists check throughput (hits and misses)
//! - Mixed CRUD workload (40% get, 30% put, 20% exists, 10% delete)

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

pub const StorageBenchConfig = struct {
    value_sizes: []const usize = &.{ 64, 1024, 65536 },
    entry_counts: []const usize = &.{ 100, 1000, 10_000 },
};

// ── Helpers ──────────────────────────────────────────────────────────

fn generateKey(buf: *[32]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "obj-{d:0>8}", .{i}) catch return "obj-00000000";
    return len;
}

fn generateValue(allocator: std.mem.Allocator, size: usize) ![]u8 {
    const val = try allocator.alloc(u8, size);
    @memset(val, 'x');
    return val;
}

// ── Put Benchmarks ───────────────────────────────────────────────────

fn benchStoragePut(allocator: std.mem.Allocator, count: usize, val_size: usize) !void {
    const storage = abi.features.storage;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    const value = try generateValue(allocator, val_size);
    defer allocator.free(value);

    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try storage.putObject(allocator, key, value);
    }
}

// ── Get Benchmarks (all keys pre-populated) ─────────────────────────

fn benchStorageGet(allocator: std.mem.Allocator, count: usize, val_size: usize) !void {
    const storage = abi.features.storage;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    const value = try generateValue(allocator, val_size);
    defer allocator.free(value);

    // Pre-populate
    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try storage.putObject(allocator, key, value);
    }

    // Read all keys — caller owns returned slice
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const data = try storage.getObject(allocator, key);
        defer allocator.free(data);
        std.mem.doNotOptimizeAway(data.ptr);
    }
}

// ── Delete Benchmarks ────────────────────────────────────────────────

fn benchStorageDelete(allocator: std.mem.Allocator, count: usize) !void {
    const storage = abi.features.storage;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    // Pre-populate
    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try storage.putObject(allocator, key, "delete-me");
    }

    // Delete all
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const deleted = try storage.deleteObject(key);
        std.mem.doNotOptimizeAway(&deleted);
    }
}

// ── Exists Benchmarks (hit and miss) ─────────────────────────────────

fn benchStorageExistsHit(allocator: std.mem.Allocator, count: usize) !void {
    const storage = abi.features.storage;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    // Pre-populate
    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try storage.putObject(allocator, key, "data");
    }

    // Check existence — all hits
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const exists = try storage.objectExists(key);
        std.mem.doNotOptimizeAway(&exists);
    }
}

fn benchStorageExistsMiss(count: usize) !void {
    const storage = abi.features.storage;
    // Init with memory backend (empty store — all checks are misses)
    try storage.init(std.heap.page_allocator, .{ .backend = .memory });
    defer storage.deinit();

    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const exists = try storage.objectExists(key);
        std.mem.doNotOptimizeAway(&exists);
    }
}

// ── Mixed CRUD Benchmark ─────────────────────────────────────────────
// Workload: 40% get, 30% put, 20% exists, 10% delete

fn benchStorageMixed(allocator: std.mem.Allocator, count: usize) !void {
    const storage = abi.features.storage;
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();

    const value = "benchmark-mixed-workload-payload";
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    // Pre-populate half the keyspace so gets/exists have partial hits
    var key_buf: [32]u8 = undefined;
    for (0..count / 2) |i| {
        const key = generateKey(&key_buf, i);
        try storage.putObject(allocator, key, value);
    }

    for (0..count) |_| {
        const key_id = rand.intRangeLessThan(usize, 0, count);
        const key = generateKey(&key_buf, key_id);
        const roll = rand.float(f32);

        if (roll < 0.4) {
            // 40% get
            const data = storage.getObject(allocator, key) catch null;
            if (data) |d| allocator.free(d);
        } else if (roll < 0.7) {
            // 30% put
            storage.putObject(allocator, key, value) catch {};
        } else if (roll < 0.9) {
            // 20% exists
            const exists = storage.objectExists(key) catch false;
            std.mem.doNotOptimizeAway(&exists);
        } else {
            // 10% delete
            const deleted = storage.deleteObject(key) catch false;
            std.mem.doNotOptimizeAway(&deleted);
        }
    }
}

// ── Runner ───────────────────────────────────────────────────────────

pub fn runStorageBenchmarks(allocator: std.mem.Allocator, config: StorageBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                        STORAGE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Put throughput by value size
    std.debug.print("[Put Throughput]\n", .{});
    for (config.value_sizes) |vsize| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "put_{d}B", .{vsize}) catch "put";

        const result = try runner.run(
            .{
                .name = name,
                .category = "storage/put",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
                .bytes_per_op = @intCast(vsize),
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize, vs: usize) !void {
                    try benchStoragePut(a, c, vs);
                }
            }.bench,
            .{ allocator, 1000, vsize },
        );
        std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} MB/s\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.throughputMBps(@intCast(vsize)),
        });
    }

    // Get throughput by value size
    std.debug.print("\n[Get Throughput]\n", .{});
    for (config.value_sizes) |vsize| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "get_{d}B", .{vsize}) catch "get";

        const result = try runner.run(
            .{
                .name = name,
                .category = "storage/get",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
                .bytes_per_op = @intCast(vsize),
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize, vs: usize) !void {
                    try benchStorageGet(a, c, vs);
                }
            }.bench,
            .{ allocator, 1000, vsize },
        );
        std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} MB/s\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.throughputMBps(@intCast(vsize)),
        });
    }

    // Delete throughput
    std.debug.print("\n[Delete Throughput]\n", .{});
    for (config.entry_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "delete_{d}", .{count}) catch "delete";

        const result = try runner.run(
            .{
                .name = name,
                .category = "storage/delete",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchStorageDelete(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    // Exists check (hit vs miss)
    std.debug.print("\n[Exists Check Throughput]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "exists_hit",
                .category = "storage/exists",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchStorageExistsHit(a, c);
                }
            }.bench,
            .{ allocator, 1000 },
        );
        std.debug.print("  exists_hit: {d:.0} ops/sec\n", .{
            result.stats.opsPerSecond(),
        });
    }
    {
        const result = try runner.run(
            .{
                .name = "exists_miss",
                .category = "storage/exists",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(c: usize) !void {
                    try benchStorageExistsMiss(c);
                }
            }.bench,
            .{1000},
        );
        std.debug.print("  exists_miss: {d:.0} ops/sec\n", .{
            result.stats.opsPerSecond(),
        });
    }

    // Mixed CRUD workload
    std.debug.print("\n[Mixed CRUD (40%% get, 30%% put, 20%% exists, 10%% delete)]\n", .{});
    for (config.entry_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "mixed_{d}", .{count}) catch "mixed";

        const result = try runner.run(
            .{
                .name = name,
                .category = "storage/mixed",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchStorageMixed(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{
            name,
            result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runStorageBenchmarks(allocator, .{});
}

test "storage benchmarks compile" {
    const allocator = std.testing.allocator;
    try benchStoragePut(allocator, 10, 64);
    try benchStorageGet(allocator, 10, 64);
    try benchStorageDelete(allocator, 10);
    try benchStorageExistsHit(allocator, 10);
    try benchStorageExistsMiss(10);
    try benchStorageMixed(allocator, 50);
}
