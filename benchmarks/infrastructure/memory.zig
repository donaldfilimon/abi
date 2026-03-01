//! Memory Allocator Benchmarks
//!
//! Industry-standard benchmarks for memory allocation:
//! - Small object allocation (1-64 bytes)
//! - Medium object allocation (64-4096 bytes)
//! - Large object allocation (4KB-1MB)
//! - Fragmentation resistance
//! - Multi-threaded allocation patterns
//! - Arena allocator performance
//! - Pool allocator performance
//! - Memory pool recycling
//! - Allocation/deallocation patterns (LIFO, FIFO, random)

const std = @import("std");
const framework = @import("../system/framework.zig");

/// Memory benchmark configuration
pub const MemoryBenchConfig = struct {
    /// Small allocation sizes (bytes)
    small_sizes: []const usize = &.{ 8, 16, 32, 64 },
    /// Medium allocation sizes (bytes)
    medium_sizes: []const usize = &.{ 128, 256, 512, 1024, 2048, 4096 },
    /// Large allocation sizes (bytes)
    large_sizes: []const usize = &.{ 8192, 16384, 65536, 262144, 1048576 },
    /// Number of allocations per iteration
    alloc_count: usize = 1000,
    /// Number of threads for concurrent tests
    thread_counts: []const usize = &.{ 1, 2, 4, 8 },
    /// Whether to test fragmentation resistance
    test_fragmentation: bool = true,
};

// ============================================================================
// Allocation Pattern Benchmarks
// ============================================================================

/// Sequential allocation and deallocation (LIFO - stack-like)
fn benchLifoPattern(allocator: std.mem.Allocator, size: usize, count: usize) !void {
    var allocations = try allocator.alloc([]u8, count);
    defer allocator.free(allocations);

    // Allocate all
    for (0..count) |i| {
        allocations[i] = try allocator.alloc(u8, size);
    }

    // Free in reverse order (LIFO)
    var i = count;
    while (i > 0) {
        i -= 1;
        allocator.free(allocations[i]);
    }
}

/// Sequential allocation and deallocation (FIFO - queue-like)
fn benchFifoPattern(allocator: std.mem.Allocator, size: usize, count: usize) !void {
    var allocations = try allocator.alloc([]u8, count);
    defer allocator.free(allocations);

    // Allocate all
    for (0..count) |i| {
        allocations[i] = try allocator.alloc(u8, size);
    }

    // Free in same order (FIFO)
    for (0..count) |i| {
        allocator.free(allocations[i]);
    }
}

/// Mixed allocation/deallocation (realistic workload)
fn benchMixedPattern(allocator: std.mem.Allocator, size: usize, count: usize) !void {
    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    var live_allocations = std.ArrayListUnmanaged([]u8).empty;
    defer live_allocations.deinit(allocator);

    var operations: usize = 0;
    while (operations < count * 2) {
        const do_alloc = live_allocations.items.len == 0 or
            (live_allocations.items.len < count and rand.boolean());

        if (do_alloc) {
            const ptr = allocator.alloc(u8, size) catch continue;
            try live_allocations.append(allocator, ptr);
        } else {
            const idx = rand.intRangeLessThan(usize, 0, live_allocations.items.len);
            const ptr = live_allocations.swapRemove(idx);
            allocator.free(ptr);
        }
        operations += 1;
    }

    // Clean up remaining
    for (live_allocations.items) |alloc| {
        allocator.free(alloc);
    }
}

/// Variable size allocation (realistic memory pressure)
fn benchVariableSizes(allocator: std.mem.Allocator, count: usize) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const size_classes = [_]usize{ 16, 32, 64, 128, 256, 512, 1024, 2048 };

    const AllocInfo = struct { ptr: []u8, size: usize };
    var allocations = try allocator.alloc(AllocInfo, count);
    defer allocator.free(allocations);

    // Allocate with random sizes
    for (0..count) |i| {
        const alloc_size = size_classes[rand.intRangeLessThan(usize, 0, size_classes.len)];
        const ptr = allocator.alloc(u8, alloc_size) catch continue;
        allocations[i] = .{ .ptr = ptr, .size = alloc_size };
    }

    // Free in random order
    for (0..count) |i| {
        const alloc = allocations[count - 1 - i];
        allocator.free(alloc.ptr);
    }
}

// ============================================================================
// Arena Allocator Benchmarks
// ============================================================================

/// Arena allocator - bulk allocation then free
fn benchArenaAllocator(backing: std.mem.Allocator, size: usize, count: usize) !void {
    var arena = std.heap.ArenaAllocator.init(backing);
    defer arena.deinit();

    const allocator = arena.allocator();

    // Allocate many objects
    for (0..count) |_| {
        const ptr = try allocator.alloc(u8, size);
        std.mem.doNotOptimizeAway(ptr.ptr);
    }

    // Reset is O(1)
    _ = arena.reset(.retain_capacity);
}

/// Arena with mixed sizes
fn benchArenaMixedSizes(backing: std.mem.Allocator, count: usize) !void {
    var arena = std.heap.ArenaAllocator.init(backing);
    defer arena.deinit();

    const allocator = arena.allocator();
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    for (0..count) |_| {
        const size = rand.intRangeAtMost(usize, 8, 4096);
        const ptr = try allocator.alloc(u8, size);
        std.mem.doNotOptimizeAway(ptr.ptr);
    }
}

// ============================================================================
// Fixed Buffer Allocator Benchmarks
// ============================================================================

fn benchFixedBufferAllocator(backing: std.mem.Allocator, size: usize, count: usize) !void {
    const buffer = try backing.alloc(u8, size * count * 2);
    defer backing.free(buffer);

    var fba = std.heap.FixedBufferAllocator.init(buffer);

    for (0..count) |_| {
        const ptr = fba.allocator().alloc(u8, size) catch break;
        std.mem.doNotOptimizeAway(ptr.ptr);
    }

    fba.reset();
}

// ============================================================================
// Memory Pool Benchmarks
// ============================================================================

/// Simple pool allocator for fixed-size objects
fn PoolAllocator(comptime T: type) type {
    return struct {
        const Self = @This();
        const POOL_SIZE = 4096;

        backing: std.mem.Allocator,
        free_list: ?*Node = null,
        slabs: std.ArrayListUnmanaged([]u8) = .{},

        const Node = struct {
            next: ?*Node,
            data: T,
        };

        pub fn init(backing: std.mem.Allocator) Self {
            return .{ .backing = backing };
        }

        pub fn deinit(self: *Self) void {
            for (self.slabs.items) |slab| {
                self.backing.free(slab);
            }
            self.slabs.deinit(self.backing);
        }

        fn allocSlab(self: *Self) !void {
            // Use regular alloc with proper size
            const slab = try self.backing.alloc(u8, POOL_SIZE * @sizeOf(Node));
            try self.slabs.append(self.backing, slab);

            // Initialize free list
            var i: usize = 0;
            while (i < POOL_SIZE - 1) : (i += 1) {
                const node: *Node = @ptrCast(@alignCast(slab.ptr + i * @sizeOf(Node)));
                const next: *Node = @ptrCast(@alignCast(slab.ptr + (i + 1) * @sizeOf(Node)));
                node.next = next;
            }
            const last: *Node = @ptrCast(@alignCast(slab.ptr + (POOL_SIZE - 1) * @sizeOf(Node)));
            last.next = self.free_list;
            self.free_list = @ptrCast(@alignCast(slab.ptr));
        }

        pub fn alloc(self: *Self) !*T {
            if (self.free_list == null) {
                try self.allocSlab();
            }
            const node = self.free_list.?;
            self.free_list = node.next;
            return &node.data;
        }

        pub fn free(self: *Self, ptr: *T) void {
            const node: *Node = @fieldParentPtr("data", ptr);
            node.next = self.free_list;
            self.free_list = node;
        }
    };
}

fn benchPoolAllocator(allocator: std.mem.Allocator, count: usize) !void {
    const TestObject = struct {
        data: [64]u8,
        value: u64,
    };

    var pool = PoolAllocator(TestObject).init(allocator);
    defer pool.deinit();

    // Allocate all
    var ptrs = try allocator.alloc(*TestObject, count);
    defer allocator.free(ptrs);

    for (0..count) |i| {
        ptrs[i] = try pool.alloc();
    }

    // Free all
    for (ptrs) |ptr| {
        pool.free(ptr);
    }
}

// ============================================================================
// Fragmentation Benchmarks
// ============================================================================

fn benchFragmentationResistance(allocator: std.mem.Allocator, iterations: usize) !void {
    var prng = std.Random.DefaultPrng.init(1337);
    const rand = prng.random();

    var live = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (live.items) |item| {
            allocator.free(item);
        }
        live.deinit(allocator);
    }

    for (0..iterations) |_| {
        // Random operation
        if (live.items.len < 1000 and (live.items.len == 0 or rand.boolean())) {
            // Allocate random size (power of 2 + random offset for worst case)
            const base_size: usize = @as(usize, 1) << rand.intRangeAtMost(u4, 3, 12);
            const size = base_size + rand.intRangeLessThan(usize, 0, base_size / 2);
            const ptr = allocator.alloc(u8, size) catch continue;
            try live.append(allocator, ptr);
        } else if (live.items.len > 0) {
            // Free random allocation
            const idx = rand.intRangeLessThan(usize, 0, live.items.len);
            allocator.free(live.swapRemove(idx));
        }
    }
}

// ============================================================================
// Multi-threaded Benchmarks
// ============================================================================

const ThreadedAllocArgs = struct {
    allocator: std.mem.Allocator,
    count: usize,
    size: usize,
    done: *std.atomic.Value(usize),
};

fn threadedAllocWorker(args: ThreadedAllocArgs) void {
    for (0..args.count) |_| {
        const ptr = args.allocator.alloc(u8, args.size) catch continue;
        std.mem.doNotOptimizeAway(ptr.ptr);
        args.allocator.free(ptr);
    }
    _ = args.done.fetchAdd(1, .release);
}

fn benchMultithreadedAlloc(
    allocator: std.mem.Allocator,
    thread_count: usize,
    count_per_thread: usize,
    size: usize,
) !void {
    var done = std.atomic.Value(usize).init(0);
    var threads: [16]std.Thread = undefined;

    const actual_threads = @min(thread_count, 16);

    for (0..actual_threads) |i| {
        threads[i] = try std.Thread.spawn(.{}, threadedAllocWorker, .{ThreadedAllocArgs{
            .allocator = allocator,
            .count = count_per_thread,
            .size = size,
            .done = &done,
        }});
    }

    for (0..actual_threads) |i| {
        threads[i].join();
    }
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runMemoryBenchmarks(allocator: std.mem.Allocator, config: MemoryBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    MEMORY ALLOCATOR BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Small allocations
    std.debug.print("[Small Object Allocation (LIFO Pattern)]\n", .{});
    for (config.small_sizes) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "lifo_{d}B", .{size}) catch "lifo";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/small",
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
                .track_memory = true,
            },
            struct {
                fn bench(a: std.mem.Allocator, s: usize, c: usize) !void {
                    try benchLifoPattern(a, s, c);
                }
            }.bench,
            .{ allocator, size, config.alloc_count },
        );

        std.debug.print("  {s}: {d:.0} ops/sec, {d:.0} allocs/sec\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(config.alloc_count)),
        });
    }

    std.debug.print("\n[Small Object Allocation (FIFO Pattern)]\n", .{});
    for (config.small_sizes) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "fifo_{d}B", .{size}) catch "fifo";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/small",
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, s: usize, c: usize) !void {
                    try benchFifoPattern(a, s, c);
                }
            }.bench,
            .{ allocator, size, config.alloc_count },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    std.debug.print("\n[Mixed Allocation Pattern]\n", .{});
    for (config.medium_sizes) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "mixed_{d}B", .{size}) catch "mixed";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/mixed",
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, s: usize, c: usize) !void {
                    try benchMixedPattern(a, s, c);
                }
            }.bench,
            .{ allocator, size, config.alloc_count / 10 },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    std.debug.print("\n[Variable Size Allocation]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "variable_sizes",
                .category = "memory/variable",
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchVariableSizes(a, c);
                }
            }.bench,
            .{ allocator, config.alloc_count },
        );

        std.debug.print("  variable_sizes: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    std.debug.print("\n[Arena Allocator]\n", .{});
    for ([_]usize{ 64, 256, 1024 }) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "arena_{d}B", .{size}) catch "arena";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/arena",
                .warmup_iterations = 100,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, s: usize, c: usize) !void {
                    try benchArenaAllocator(a, s, c);
                }
            }.bench,
            .{ allocator, size, config.alloc_count },
        );

        std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} allocs/sec)\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(config.alloc_count)),
        });
    }

    std.debug.print("\n[Pool Allocator (Fixed Size)]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "pool_64B",
                .category = "memory/pool",
                .warmup_iterations = 100,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, c: usize) !void {
                    try benchPoolAllocator(a, c);
                }
            }.bench,
            .{ allocator, config.alloc_count },
        );

        std.debug.print("  pool_64B: {d:.0} ops/sec ({d:.0} allocs/sec)\n", .{
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(config.alloc_count)),
        });
    }

    std.debug.print("\n[Fixed Buffer Allocator]\n", .{});
    for ([_]usize{ 64, 256 }) |size| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "fba_{d}B", .{size}) catch "fba";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/fba",
                .warmup_iterations = 100,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, s: usize, c: usize) !void {
                    try benchFixedBufferAllocator(a, s, c);
                }
            }.bench,
            .{ allocator, size, config.alloc_count },
        );

        std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
    }

    if (config.test_fragmentation) {
        std.debug.print("\n[Fragmentation Resistance]\n", .{});
        const result = try runner.run(
            .{
                .name = "fragmentation_test",
                .category = "memory/fragmentation",
                .warmup_iterations = 5,
                .min_time_ns = 1_000_000_000,
                .max_iterations = 100,
            },
            struct {
                fn bench(a: std.mem.Allocator, iters: usize) !void {
                    try benchFragmentationResistance(a, iters);
                }
            }.bench,
            .{ allocator, 10000 },
        );

        std.debug.print("  fragmentation_test: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    std.debug.print("\n[Multi-threaded Allocation]\n", .{});
    for (config.thread_counts) |threads| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "mt_{d}threads", .{threads}) catch "mt";

        const result = try runner.run(
            .{
                .name = name,
                .category = "memory/threaded",
                .warmup_iterations = 5,
                .min_time_ns = 100_000_000,
                .max_iterations = 100,
            },
            struct {
                fn bench(a: std.mem.Allocator, t: usize, c: usize, s: usize) !void {
                    try benchMultithreadedAlloc(a, t, c, s);
                }
            }.bench,
            .{ allocator, threads, 1000, 64 },
        );

        const total_allocs = @as(f64, @floatFromInt(threads * 1000));
        std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} total allocs/sec)\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * total_allocs,
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

test "memory benchmarks compile" {
    const allocator = std.testing.allocator;

    try benchLifoPattern(allocator, 64, 10);
    try benchFifoPattern(allocator, 64, 10);
    try benchMixedPattern(allocator, 64, 10);
    try benchVariableSizes(allocator, 10);
    try benchArenaAllocator(allocator, 64, 10);
    try benchPoolAllocator(allocator, 10);
}
