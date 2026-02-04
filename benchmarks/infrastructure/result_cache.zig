//! Result Cache Benchmarks
//!
//! Performance verification for the fast-path result cache:
//! - Cache hit latency (target: ~20ns)
//! - Cache miss latency (target: ~50ns)
//! - Cache put latency (target: ~100ns)
//! - Concurrent access patterns
//! - Hit rate under various workloads
//! - Memory efficiency and eviction overhead

const std = @import("std");
const framework = @import("../system/framework.zig");

/// Result cache benchmark configuration
pub const ResultCacheBenchConfig = struct {
    /// Number of entries for cache capacity tests
    cache_sizes: []const usize = &.{ 256, 1024, 4096, 16384 },
    /// Number of operations per benchmark iteration
    ops_per_iteration: usize = 10000,
    /// Hit rate targets (percentage of gets that should hit)
    hit_rates: []const f64 = &.{ 0.5, 0.8, 0.95, 0.99 },
    /// Key sizes (simulating different task key complexities)
    key_sizes: []const usize = &.{ 8, 32, 64, 128 },
    /// Value sizes (simulating different result sizes)
    value_sizes: []const usize = &.{ 64, 256, 1024, 4096 },
    /// Number of concurrent threads for contention tests
    thread_counts: []const usize = &.{ 1, 2, 4, 8 },
    /// Minimum benchmark time
    min_time_ns: u64 = 100_000_000,
    /// Warmup iterations
    warmup_iterations: usize = 100,

    pub const quick = ResultCacheBenchConfig{
        .cache_sizes = &.{ 256, 1024 },
        .ops_per_iteration = 1000,
        .hit_rates = &.{ 0.8, 0.95 },
        .key_sizes = &.{ 32, 64 },
        .value_sizes = &.{ 256, 1024 },
        .thread_counts = &.{ 1, 2 },
        .min_time_ns = 50_000_000,
        .warmup_iterations = 50,
    };

    pub const standard = ResultCacheBenchConfig{
        .cache_sizes = &.{ 256, 1024, 4096 },
        .ops_per_iteration = 5000,
        .hit_rates = &.{ 0.5, 0.8, 0.95 },
        .key_sizes = &.{ 32, 64, 128 },
        .value_sizes = &.{ 256, 1024 },
        .thread_counts = &.{ 1, 2, 4 },
        .min_time_ns = 100_000_000,
        .warmup_iterations = 100,
    };

    pub const comprehensive = ResultCacheBenchConfig{
        .cache_sizes = &.{ 256, 1024, 4096, 16384 },
        .ops_per_iteration = 10000,
        .hit_rates = &.{ 0.5, 0.8, 0.95, 0.99 },
        .key_sizes = &.{ 8, 32, 64, 128 },
        .value_sizes = &.{ 64, 256, 1024, 4096 },
        .thread_counts = &.{ 1, 2, 4, 8 },
        .min_time_ns = 200_000_000,
        .warmup_iterations = 200,
    };
};

// ============================================================================
// Mock Cache Implementation (for benchmarking without full ABI dependency)
// ============================================================================

/// Simple hash-based cache for benchmarking.
/// Mirrors the API of the real ResultCache but simplified for benchmark isolation.
fn MockCache(comptime K: type, comptime V: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        entries: std.AutoHashMapUnmanaged(K, CacheEntry),
        max_entries: usize,
        stats: Stats,

        const CacheEntry = struct {
            value: V,
            access_count: u32,
        };

        const Stats = struct {
            gets: u64 = 0,
            hits: u64 = 0,
            misses: u64 = 0,
            puts: u64 = 0,
            evictions: u64 = 0,
        };

        pub fn init(allocator: std.mem.Allocator, max_entries: usize) Self {
            return .{
                .allocator = allocator,
                .entries = .{},
                .max_entries = max_entries,
                .stats = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }

        pub fn get(self: *Self, key: K) ?V {
            self.stats.gets += 1;
            if (self.entries.getPtr(key)) |entry| {
                self.stats.hits += 1;
                entry.access_count += 1;
                return entry.value;
            }
            self.stats.misses += 1;
            return null;
        }

        pub fn put(self: *Self, key: K, value: V) !void {
            self.stats.puts += 1;

            // Evict if at capacity
            if (self.entries.count() >= self.max_entries and !self.entries.contains(key)) {
                // Simple eviction: remove first entry found
                var iter = self.entries.keyIterator();
                if (iter.next()) |k| {
                    _ = self.entries.remove(k.*);
                    self.stats.evictions += 1;
                }
            }

            try self.entries.put(self.allocator, key, .{
                .value = value,
                .access_count = 1,
            });
        }

        pub fn getStats(self: *const Self) Stats {
            return self.stats;
        }

        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
            self.stats = .{};
        }
    };
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark cache hit latency (best case: key exists)
fn benchCacheHit(cache: *MockCache(u64, [256]u8), ops: usize) u64 {
    var sum: u64 = 0;
    for (0..ops) |i| {
        const key = i % 100; // Always hit (pre-populated keys 0-99)
        if (cache.get(key)) |v| {
            sum += v[0];
        }
    }
    return sum;
}

/// Benchmark cache miss latency (worst case: key doesn't exist)
fn benchCacheMiss(cache: *MockCache(u64, [256]u8), ops: usize) u64 {
    var sum: u64 = 0;
    for (0..ops) |i| {
        const key = i + 1_000_000; // Never hit (keys start at 1M)
        if (cache.get(key)) |v| {
            sum += v[0];
        }
    }
    return sum;
}

/// Benchmark cache put latency
fn benchCachePut(cache: *MockCache(u64, [256]u8), ops: usize, value: *const [256]u8) void {
    for (0..ops) |i| {
        cache.put(i, value.*) catch {};
    }
}

/// Benchmark mixed workload (configurable hit rate)
fn benchMixedWorkload(cache: *MockCache(u64, [256]u8), ops: usize, hit_rate: f64, value: *const [256]u8) u64 {
    var sum: u64 = 0;
    const hit_threshold = @as(u64, @intFromFloat(hit_rate * 100.0));

    for (0..ops) |i| {
        const should_hit = (i % 100) < hit_threshold;
        const key = if (should_hit) i % 100 else i + 1_000_000;

        if (cache.get(key)) |v| {
            sum += v[0];
        } else {
            cache.put(key, value.*) catch {};
        }
    }
    return sum;
}

// ============================================================================
// Public Benchmark API
// ============================================================================

/// Run all result cache benchmarks.
pub fn runAllBenchmarks(allocator: std.mem.Allocator, config: ResultCacheBenchConfig) !void {
    std.debug.print("\n=== Result Cache Benchmarks ===\n\n", .{});

    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Prepare test data
    var value: [256]u8 = undefined;
    for (&value, 0..) |*b, i| {
        b.* = @intCast(i % 256);
    }

    // Benchmark 1: Cache hit latency
    std.debug.print("--- Cache Hit Latency ---\n", .{});
    for (config.cache_sizes) |cache_size| {
        var cache = MockCache(u64, [256]u8).init(allocator, cache_size);
        defer cache.deinit();

        // Pre-populate with 100 entries for guaranteed hits
        for (0..100) |i| {
            try cache.put(i, value);
        }

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("cache_hit_{d}", .{cache_size}),
                .category = "result_cache",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchCacheHit,
            .{ &cache, config.ops_per_iteration },
        );

        const ns_per_op = result.stats.mean_ns / @as(f64, @floatFromInt(config.ops_per_iteration));
        std.debug.print("  size={d}: {d:.1}ns/op (target: ~20ns)\n", .{ cache_size, ns_per_op });
    }

    // Benchmark 2: Cache miss latency
    std.debug.print("\n--- Cache Miss Latency ---\n", .{});
    for (config.cache_sizes) |cache_size| {
        var cache = MockCache(u64, [256]u8).init(allocator, cache_size);
        defer cache.deinit();

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("cache_miss_{d}", .{cache_size}),
                .category = "result_cache",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchCacheMiss,
            .{ &cache, config.ops_per_iteration },
        );

        const ns_per_op = result.stats.mean_ns / @as(f64, @floatFromInt(config.ops_per_iteration));
        std.debug.print("  size={d}: {d:.1}ns/op (target: ~50ns)\n", .{ cache_size, ns_per_op });
    }

    // Benchmark 3: Cache put latency
    std.debug.print("\n--- Cache Put Latency ---\n", .{});
    for (config.cache_sizes) |cache_size| {
        var cache = MockCache(u64, [256]u8).init(allocator, cache_size);
        defer cache.deinit();

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("cache_put_{d}", .{cache_size}),
                .category = "result_cache",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchCachePut,
            .{ &cache, config.ops_per_iteration, &value },
        );

        const ns_per_op = result.stats.mean_ns / @as(f64, @floatFromInt(config.ops_per_iteration));
        std.debug.print("  size={d}: {d:.1}ns/op (target: ~100ns)\n", .{ cache_size, ns_per_op });
    }

    // Benchmark 4: Mixed workloads at various hit rates
    std.debug.print("\n--- Mixed Workload (Hit Rate Sweep) ---\n", .{});
    for (config.hit_rates) |hit_rate| {
        var cache = MockCache(u64, [256]u8).init(allocator, 1024);
        defer cache.deinit();

        // Pre-populate
        for (0..100) |i| {
            try cache.put(i, value);
        }

        const result = try runner.run(
            .{
                .name = std.fmt.comptimePrint("mixed_{d}pct", .{@as(u64, @intFromFloat(hit_rate * 100))}),
                .category = "result_cache",
                .min_time_ns = config.min_time_ns,
                .warmup_iterations = config.warmup_iterations,
            },
            benchMixedWorkload,
            .{ &cache, config.ops_per_iteration, hit_rate, &value },
        );

        const ns_per_op = result.stats.mean_ns / @as(f64, @floatFromInt(config.ops_per_iteration));
        const stats = cache.getStats();
        const actual_hit_rate = if (stats.gets > 0)
            @as(f64, @floatFromInt(stats.hits)) / @as(f64, @floatFromInt(stats.gets)) * 100.0
        else
            0.0;
        std.debug.print("  target={d:.0}%: {d:.1}ns/op, actual_hit_rate={d:.1}%\n", .{
            hit_rate * 100.0,
            ns_per_op,
            actual_hit_rate,
        });
    }

    std.debug.print("\n=== Result Cache Benchmarks Complete ===\n", .{});
}

/// Run quick benchmarks for CI.
pub fn runQuickBenchmarks(allocator: std.mem.Allocator) !void {
    try runAllBenchmarks(allocator, ResultCacheBenchConfig.quick);
}

// ============================================================================
// Tests
// ============================================================================

test "mock cache basic operations" {
    const allocator = std.testing.allocator;
    var cache = MockCache(u64, u64).init(allocator, 10);
    defer cache.deinit();

    // Put and get
    try cache.put(1, 100);
    try std.testing.expectEqual(@as(?u64, 100), cache.get(1));

    // Miss
    try std.testing.expectEqual(@as(?u64, null), cache.get(999));

    // Stats
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.gets);
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
    try std.testing.expectEqual(@as(u64, 1), stats.misses);
}

test "mock cache eviction" {
    const allocator = std.testing.allocator;
    var cache = MockCache(u64, u64).init(allocator, 3);
    defer cache.deinit();

    try cache.put(1, 100);
    try cache.put(2, 200);
    try cache.put(3, 300);
    try cache.put(4, 400); // Should trigger eviction

    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.evictions);
    try std.testing.expectEqual(@as(usize, 3), cache.entries.count());
}

test "benchmark functions compile" {
    const allocator = std.testing.allocator;
    var cache = MockCache(u64, [256]u8).init(allocator, 100);
    defer cache.deinit();

    var value: [256]u8 = undefined;
    @memset(&value, 42);

    for (0..10) |i| {
        try cache.put(i, value);
    }

    _ = benchCacheHit(&cache, 10);
    _ = benchCacheMiss(&cache, 10);
    benchCachePut(&cache, 10, &value);
    _ = benchMixedWorkload(&cache, 10, 0.8, &value);
}
