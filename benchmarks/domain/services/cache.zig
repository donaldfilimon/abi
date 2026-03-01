//! Cache Benchmarks
//!
//! Performance measurement for the in-memory cache module:
//! - Put throughput (various value sizes)
//! - Get throughput (hits vs misses)
//! - Mixed read/write workloads (realistic)
//! - Eviction overhead (LRU, LFU, FIFO)
//! - TTL expiry impact

const std = @import("std");
const abi = @import("abi");
const framework = @import("../../system/framework.zig");

pub const CacheBenchConfig = struct {
    value_sizes: []const usize = &.{ 64, 256, 1024, 4096 },
    entry_counts: []const usize = &.{ 100, 1000, 10_000 },
    read_ratio: f64 = 0.8, // 80% reads, 20% writes
};

// ── Helpers ──────────────────────────────────────────────────────────

fn generateKey(buf: *[32]u8, i: usize) []const u8 {
    const len = std.fmt.bufPrint(buf, "key-{d:0>8}", .{i}) catch return "key-00000000";
    return len;
}

fn generateValue(allocator: std.mem.Allocator, size: usize) ![]u8 {
    const val = try allocator.alloc(u8, size);
    @memset(val, 'x');
    return val;
}

// ── Put Benchmarks ───────────────────────────────────────────────────

fn benchCachePut(allocator: std.mem.Allocator, count: usize, val_size: usize) !void {
    const cache = abi.features.cache;
    try cache.init(allocator, .{
        .max_entries = @intCast(@min(count * 2, 100_000)),
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    const value = try generateValue(allocator, val_size);
    defer allocator.free(value);

    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try cache.put(key, value);
    }
}

// ── Get Benchmarks (100% hit rate — all keys pre-populated) ─────────

fn benchCacheGetHit(allocator: std.mem.Allocator, count: usize, val_size: usize) !void {
    const cache = abi.features.cache;
    try cache.init(allocator, .{
        .max_entries = @intCast(@min(count * 2, 100_000)),
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    const value = try generateValue(allocator, val_size);
    defer allocator.free(value);

    // Pre-populate
    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        try cache.put(key, value);
    }

    // Read all keys (all hits)
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const result = try cache.get(key);
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Get Benchmarks (0% hit rate — all misses) ───────────────────────

fn benchCacheGetMiss(allocator: std.mem.Allocator, count: usize) !void {
    const cache = abi.features.cache;
    try cache.init(allocator, .{
        .max_entries = 100,
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    var key_buf: [32]u8 = undefined;
    for (0..count) |i| {
        const key = generateKey(&key_buf, i);
        const result = cache.get(key) catch null;
        std.mem.doNotOptimizeAway(&result);
    }
}

// ── Eviction Benchmark (cache full, every put triggers eviction) ────

fn benchCacheEviction(allocator: std.mem.Allocator, count: usize, policy: abi.features.cache.EvictionPolicy) !void {
    const cache = abi.features.cache;
    const capacity: u32 = 500;
    try cache.init(allocator, .{
        .max_entries = capacity,
        .eviction_policy = policy,
    });
    defer cache.deinit();

    const value = "short-value";
    var key_buf: [32]u8 = undefined;

    // Fill to capacity, then overflow by `count` entries (each triggers eviction)
    for (0..@as(usize, capacity) + count) |i| {
        const key = generateKey(&key_buf, i);
        try cache.put(key, value);
    }
}

// ── Mixed Read/Write Benchmark ──────────────────────────────────────

fn benchCacheMixed(allocator: std.mem.Allocator, count: usize) !void {
    const cache = abi.features.cache;
    try cache.init(allocator, .{
        .max_entries = 5000,
        .eviction_policy = .lru,
    });
    defer cache.deinit();

    const value = "benchmark-value-payload-data-here";
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var key_buf: [32]u8 = undefined;

    for (0..count) |_| {
        const key_id = rand.intRangeLessThan(usize, 0, 10_000);
        const key = generateKey(&key_buf, key_id);

        if (rand.float(f32) < 0.8) {
            // 80% reads
            const result = cache.get(key) catch null;
            std.mem.doNotOptimizeAway(&result);
        } else {
            // 20% writes
            cache.put(key, value) catch {};
        }
    }
}

// ── Runner ───────────────────────────────────────────────────────────

pub fn runCacheBenchmarks(allocator: std.mem.Allocator, config: CacheBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                         CACHE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Put throughput by value size
    std.debug.print("[Put Throughput]\n", .{});
    for (config.value_sizes) |vsize| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "put_{d}B", .{vsize}) catch "put";

        const result = try runner.run(
            .{
                .name = name,
                .category = "cache/put",
                .warmup_iterations = 3,
                .min_time_ns = 100_000_000,
                .bytes_per_op = @intCast(vsize),
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize, vs: usize) !void {
                    try benchCachePut(a, c, vs);
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

    // Get hit/miss ratio
    std.debug.print("\n[Get Throughput]\n", .{});
    {
        const result = try runner.run(
            .{ .name = "get_hit_100pct", .category = "cache/get", .warmup_iterations = 3, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator) !void {
                    try benchCacheGetHit(a, 1000, 64);
                }
            }.bench,
            .{allocator},
        );
        std.debug.print("  get_hit_100pct: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }
    {
        const result = try runner.run(
            .{ .name = "get_miss_100pct", .category = "cache/get", .warmup_iterations = 3, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator) !void {
                    try benchCacheGetMiss(a, 1000);
                }
            }.bench,
            .{allocator},
        );
        std.debug.print("  get_miss_100pct: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // Eviction overhead by policy
    std.debug.print("\n[Eviction Overhead]\n", .{});
    const policies = [_]abi.features.cache.EvictionPolicy{ .lru, .lfu, .fifo, .random };
    const policy_names = [_][]const u8{ "evict_lru", "evict_lfu", "evict_fifo", "evict_random" };
    for (policies, policy_names) |policy, pname| {
        const result = try runner.run(
            .{ .name = pname, .category = "cache/eviction", .warmup_iterations = 3, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, c: usize, p: abi.features.cache.EvictionPolicy) !void {
                    try benchCacheEviction(a, c, p);
                }
            }.bench,
            .{ allocator, 2000, policy },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ pname, result.stats.opsPerSecond() });
    }

    // Mixed workload
    std.debug.print("\n[Mixed Workload (80/20 read/write)]\n", .{});
    for (config.entry_counts) |count| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "mixed_{d}", .{count}) catch "mixed";

        const result = try runner.run(
            .{ .name = name, .category = "cache/mixed", .warmup_iterations = 3, .min_time_ns = 100_000_000 },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchCacheMixed(a, c);
                }
            }.bench,
            .{ allocator, count },
        );
        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn run(allocator: std.mem.Allocator) !void {
    try runCacheBenchmarks(allocator, .{});
}

test "cache benchmarks compile" {
    const allocator = std.testing.allocator;
    try benchCachePut(allocator, 10, 64);
    try benchCacheGetHit(allocator, 10, 64);
    try benchCacheGetMiss(allocator, 10);
    try benchCacheEviction(allocator, 10, .lru);
    try benchCacheMixed(allocator, 50);
}
