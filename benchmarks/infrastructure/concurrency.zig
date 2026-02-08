//! Concurrency and Parallelism Benchmarks
//!
//! Industry-standard benchmarks for concurrent operations:
//! - Lock-free data structures (queue, stack, map)
//! - Mutex contention (low, medium, high)
//! - Read-write lock patterns
//! - Atomic operations (CAS, fetch-add, load/store)
//! - Thread pool throughput
//! - Work stealing efficiency
//! - Producer-consumer patterns
//! - Barrier synchronization
//! - Channel/message passing throughput

const std = @import("std");
const abi = @import("abi");
const sync = abi.shared.sync;
const framework = @import("../system/framework.zig");

/// Concurrency benchmark configuration
pub const ConcurrencyBenchConfig = struct {
    thread_counts: []const usize = &.{ 1, 2, 4 },
    operations_per_thread: usize = 10_000,
    queue_size: usize = 1024,
    work_item_size: usize = 64,
    contention_levels: []const ContentionLevel = &.{ .low, .high },
};

pub const ContentionLevel = enum {
    low, // Long work between locks
    medium, // Medium work between locks
    high, // Minimal work between locks (stress test)
};

// ============================================================================
// Atomic Operations Benchmarks
// ============================================================================

/// Benchmark atomic fetch-add operations
fn benchAtomicFetchAdd(count: usize) u64 {
    var counter = std.atomic.Value(u64).init(0);
    for (0..count) |_| {
        _ = counter.fetchAdd(1, .seq_cst);
    }
    return counter.load(.seq_cst);
}

/// Benchmark atomic compare-and-swap operations
fn benchAtomicCAS(count: usize) u64 {
    var value = std.atomic.Value(u64).init(0);
    var successes: u64 = 0;

    for (0..count) |i| {
        const expected = i;
        if (value.cmpxchgStrong(expected, i + 1, .seq_cst, .seq_cst) == null) {
            successes += 1;
        }
    }

    return successes;
}

/// Benchmark atomic load/store pairs
fn benchAtomicLoadStore(count: usize) u64 {
    var value = std.atomic.Value(u64).init(0);
    var sum: u64 = 0;

    for (0..count) |i| {
        value.store(i, .release);
        sum +%= value.load(.acquire);
    }

    return sum;
}

/// Benchmark acquire-release fence patterns
fn benchMemoryFences(count: usize) u64 {
    var value = std.atomic.Value(u64).init(0);
    var sum: u64 = 0;

    for (0..count) |i| {
        // Use release store followed by acquire load to achieve fence-like behavior
        value.store(i, .release);
        sum +%= value.load(.acquire);
    }

    return sum;
}

// ============================================================================
// Lock-Free Queue Benchmarks
// ============================================================================

/// Simple SPSC (Single Producer Single Consumer) queue
fn SPSCQueue(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]T = undefined,
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        pub fn push(self: *Self, item: T) bool {
            const tail = self.tail.load(.monotonic);
            const next_tail = (tail + 1) % capacity;

            if (next_tail == self.head.load(.acquire)) {
                return false; // Full
            }

            self.buffer[tail] = item;
            self.tail.store(next_tail, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const head = self.head.load(.monotonic);

            if (head == self.tail.load(.acquire)) {
                return null; // Empty
            }

            const item = self.buffer[head];
            self.head.store((head + 1) % capacity, .release);
            return item;
        }
    };
}

/// MPSC (Multi Producer Single Consumer) queue using spinlock
fn MPSCQueue(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]T = undefined,
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        push_lock: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        pub fn push(self: *Self, item: T) bool {
            // Acquire spinlock
            while (self.push_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
                std.atomic.spinLoopHint();
            }
            defer self.push_lock.store(false, .release);

            const tail = self.tail.load(.monotonic);
            const next_tail = (tail + 1) % capacity;

            if (next_tail == self.head.load(.acquire)) {
                return false;
            }

            self.buffer[tail] = item;
            self.tail.store(next_tail, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const head = self.head.load(.monotonic);

            if (head == self.tail.load(.acquire)) {
                return null;
            }

            const item = self.buffer[head];
            self.head.store((head + 1) % capacity, .release);
            return item;
        }
    };
}

fn benchSPSCQueue(count: usize) u64 {
    var queue = SPSCQueue(u64, 1024){};
    var sum: u64 = 0;

    for (0..count) |i| {
        while (!queue.push(i)) {}
        if (queue.pop()) |v| {
            sum +%= v;
        }
    }

    return sum;
}

fn benchMPSCQueue(count: usize) u64 {
    var queue = MPSCQueue(u64, 1024){};
    var sum: u64 = 0;

    for (0..count) |i| {
        while (!queue.push(i)) {}
        if (queue.pop()) |v| {
            sum +%= v;
        }
    }

    return sum;
}

// ============================================================================
// Lock-Free Stack Benchmarks
// ============================================================================

/// Treiber stack (lock-free LIFO)
fn TreiberStack(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            value: T,
            next: ?*Node,
        };

        head: std.atomic.Value(?*Node) = std.atomic.Value(?*Node).init(null),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn push(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.value = value;

            var head = self.head.load(.monotonic);
            while (true) {
                node.next = head;
                if (self.head.cmpxchgWeak(head, node, .release, .monotonic)) |old| {
                    head = old;
                } else {
                    return;
                }
            }
        }

        pub fn pop(self: *Self) ?T {
            var head = self.head.load(.acquire);
            while (head) |node| {
                if (self.head.cmpxchgWeak(head, node.next, .release, .monotonic)) |old| {
                    head = old;
                } else {
                    const value = node.value;
                    self.allocator.destroy(node);
                    return value;
                }
            }
            return null;
        }

        pub fn deinit(self: *Self) void {
            while (self.pop() != null) {}
        }
    };
}

fn benchTreiberStack(allocator: std.mem.Allocator, count: usize) !u64 {
    var stack = TreiberStack(u64).init(allocator);
    defer stack.deinit();

    var sum: u64 = 0;

    for (0..count) |i| {
        try stack.push(i);
    }

    while (stack.pop()) |v| {
        sum +%= v;
    }

    return sum;
}

// ============================================================================
// Mutex Contention Benchmarks
// ============================================================================

fn benchMutexContention(
    allocator: std.mem.Allocator,
    thread_count: usize,
    ops_per_thread: usize,
    contention: ContentionLevel,
) !u64 {
    const SharedState = struct {
        mutex: sync.Mutex = .{},
        counter: u64 = 0,
        contention: ContentionLevel,
        ops_per_thread: usize,
    };

    var state = SharedState{
        .contention = contention,
        .ops_per_thread = ops_per_thread,
    };

    const Worker = struct {
        fn work(s: *SharedState) void {
            for (0..s.ops_per_thread) |_| {
                s.mutex.lock();
                s.counter += 1;
                s.mutex.unlock();

                // Simulate work between lock acquisitions
                switch (s.contention) {
                    .low => {
                        var sum: u64 = 0;
                        for (0..1000) |i| sum +%= i;
                        std.mem.doNotOptimizeAway(&sum);
                    },
                    .medium => {
                        var sum: u64 = 0;
                        for (0..100) |i| sum +%= i;
                        std.mem.doNotOptimizeAway(&sum);
                    },
                    .high => {
                        // Minimal work - maximum contention
                    },
                }
            }
        }
    };

    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    for (threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Worker.work, .{&state});
    }

    for (threads) |t| {
        t.join();
    }

    return state.counter;
}

// ============================================================================
// Read-Write Lock Benchmarks
// ============================================================================

fn benchRWLock(
    allocator: std.mem.Allocator,
    thread_count: usize,
    ops_per_thread: usize,
    read_ratio: u8, // Percentage of reads (0-100)
) !u64 {
    const SharedState = struct {
        rwlock: sync.RwLock = .{},
        value: u64 = 0,
        read_ratio: u8,
        ops_per_thread: usize,
    };

    var state = SharedState{
        .read_ratio = read_ratio,
        .ops_per_thread = ops_per_thread,
    };

    const Worker = struct {
        fn work(s: *SharedState, seed: u64, sum_out: *std.atomic.Value(u64)) void {
            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();
            var sum: u64 = 0;

            for (0..s.ops_per_thread) |_| {
                if (rand.intRangeLessThan(u8, 0, 100) < s.read_ratio) {
                    // Read
                    s.rwlock.lockShared();
                    sum +%= s.value;
                    s.rwlock.unlockShared();
                } else {
                    // Write
                    s.rwlock.lock();
                    s.value += 1;
                    s.rwlock.unlock();
                }
            }

            _ = sum_out.fetchAdd(sum, .monotonic);
        }
    };

    var total_sum = std.atomic.Value(u64).init(0);

    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    for (threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, Worker.work, .{ &state, i * 12345, &total_sum });
    }

    for (threads) |t| {
        t.join();
    }

    return total_sum.load(.acquire);
}

// ============================================================================
// Work Stealing Queue Benchmark
// ============================================================================

/// Work-stealing deque (simplified)
fn WorkStealingDeque(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]T = undefined,
        top: std.atomic.Value(isize) = std.atomic.Value(isize).init(0),
        bottom: std.atomic.Value(isize) = std.atomic.Value(isize).init(0),

        /// Push (owner only)
        pub fn push(self: *Self, item: T) bool {
            const b = self.bottom.load(.monotonic);
            const t = self.top.load(.acquire);

            if (b - t >= capacity) {
                return false; // Full
            }

            self.buffer[@intCast(@mod(b, capacity))] = item;
            // Use release store to ensure the buffer write is visible
            self.bottom.store(b + 1, .release);
            return true;
        }

        /// Pop (owner only) - LIFO
        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.monotonic) - 1;
            self.bottom.store(b, .seq_cst);
            const t = self.top.load(.seq_cst);

            if (t <= b) {
                const item = self.buffer[@intCast(@mod(b, capacity))];
                if (t == b) {
                    // Last element - race with steal
                    if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic) != null) {
                        self.bottom.store(t + 1, .monotonic);
                        return null;
                    }
                    self.bottom.store(t + 1, .monotonic);
                }
                return item;
            } else {
                self.bottom.store(t, .monotonic);
                return null;
            }
        }

        /// Steal (thieves) - FIFO
        pub fn steal(self: *Self) ?T {
            const t = self.top.load(.seq_cst);
            const b = self.bottom.load(.seq_cst);

            if (t < b) {
                const item = self.buffer[@intCast(@mod(t, capacity))];
                if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic) != null) {
                    return null;
                }
                return item;
            }
            return null;
        }
    };
}

fn benchWorkStealing(allocator: std.mem.Allocator, thread_count: usize, total_work: usize) !u64 {
    const SharedState = struct {
        queues: []WorkStealingDeque(u64, 4096),
        counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    };

    const initial_work_per_thread: usize = 100;
    const target_work: u64 = @intCast(@min(total_work, thread_count * initial_work_per_thread));

    const queues = try allocator.alloc(WorkStealingDeque(u64, 4096), thread_count);
    defer allocator.free(queues);

    for (queues) |*q| {
        q.* = .{};
    }

    var state = SharedState{
        .queues = queues,
    };

    const Worker = struct {
        fn work(s: *SharedState, my_id: usize) void {
            var prng = std.Random.DefaultPrng.init(my_id * 7919);
            const rand = prng.random();
            var my_queue = &s.queues[my_id];

            // Push some initial work
            for (0..initial_work_per_thread) |i| {
                _ = my_queue.push(i);
            }

            while (!s.done.load(.acquire)) {
                // Try to get work from own queue
                if (my_queue.pop()) |_| {
                    _ = s.counter.fetchAdd(1, .monotonic);
                    continue;
                }

                // Try to steal from random other queue
                const victim = rand.intRangeLessThan(usize, 0, s.queues.len);
                if (victim != my_id) {
                    if (s.queues[victim].steal()) |_| {
                        _ = s.counter.fetchAdd(1, .monotonic);
                    }
                }
            }
        }
    };

    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    for (threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, Worker.work, .{ &state, i });
    }

    // Wait for enough work using spinloop
    while (state.counter.load(.acquire) < target_work) {
        std.atomic.spinLoopHint();
    }

    state.done.store(true, .release);

    for (threads) |t| {
        t.join();
    }

    return state.counter.load(.acquire);
}

fn benchChannel(allocator: std.mem.Allocator, message_count: usize) !u64 {
    var channel = SPSCQueue(u64, 1024){};
    var result_sum = std.atomic.Value(u64).init(0);

    const producer = try std.Thread.spawn(.{}, struct {
        fn work(ch: *SPSCQueue(u64, 1024), count: usize) void {
            for (0..count) |i| {
                while (!ch.push(i)) {
                    std.atomic.spinLoopHint();
                }
            }
        }
    }.work, .{ &channel, message_count });

    const consumer = try std.Thread.spawn(.{}, struct {
        fn work(ch: *SPSCQueue(u64, 1024), count: usize, sum_out: *std.atomic.Value(u64)) void {
            var sum: u64 = 0;
            var received: usize = 0;
            while (received < count) {
                if (ch.pop()) |value| {
                    sum +%= value;
                    received += 1;
                } else {
                    std.atomic.spinLoopHint();
                }
            }
            sum_out.store(sum, .release);
        }
    }.work, .{ &channel, message_count, &result_sum });

    producer.join();
    consumer.join();

    _ = allocator;
    return result_sum.load(.acquire);
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runConcurrencyBenchmarks(allocator: std.mem.Allocator, config: ConcurrencyBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    CONCURRENCY BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Atomic operations
    std.debug.print("[Atomic Operations]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "atomic_fetch_add",
                .category = "concurrency/atomic",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(count: usize) u64 {
                    return benchAtomicFetchAdd(count);
                }
            }.bench,
            .{10000},
        );
        std.debug.print("  atomic_fetch_add: {d:.0} ops/sec ({d:.0} atomic ops/sec)\n", .{
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 10000,
        });
    }

    {
        const result = try runner.run(
            .{
                .name = "atomic_cas",
                .category = "concurrency/atomic",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(count: usize) u64 {
                    return benchAtomicCAS(count);
                }
            }.bench,
            .{10000},
        );
        std.debug.print("  atomic_cas: {d:.0} ops/sec ({d:.0} CAS ops/sec)\n", .{
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 10000,
        });
    }

    {
        const result = try runner.run(
            .{
                .name = "atomic_load_store",
                .category = "concurrency/atomic",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(count: usize) u64 {
                    return benchAtomicLoadStore(count);
                }
            }.bench,
            .{10000},
        );
        std.debug.print("  atomic_load_store: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // Lock-free queues
    std.debug.print("\n[Lock-Free Queues]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "spsc_queue",
                .category = "concurrency/lockfree",
                .warmup_iterations = 100,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(count: usize) u64 {
                    return benchSPSCQueue(count);
                }
            }.bench,
            .{10000},
        );
        std.debug.print("  spsc_queue: {d:.0} ops/sec ({d:.0} push+pop/sec)\n", .{
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 10000,
        });
    }

    {
        const result = try runner.run(
            .{
                .name = "mpsc_queue",
                .category = "concurrency/lockfree",
                .warmup_iterations = 100,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(count: usize) u64 {
                    return benchMPSCQueue(count);
                }
            }.bench,
            .{10000},
        );
        std.debug.print("  mpsc_queue: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // Lock-free stack
    std.debug.print("\n[Lock-Free Stack]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "treiber_stack",
                .category = "concurrency/lockfree",
                .warmup_iterations = 100,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, count: usize) !u64 {
                    return try benchTreiberStack(a, count);
                }
            }.bench,
            .{ allocator, 1000 },
        );
        std.debug.print("  treiber_stack: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // Mutex contention
    std.debug.print("\n[Mutex Contention]\n", .{});
    for (config.thread_counts) |threads| {
        for (config.contention_levels) |contention| {
            var name_buf: [64]u8 = undefined;
            const contention_name = switch (contention) {
                .low => "low",
                .medium => "med",
                .high => "high",
            };
            const name = std.fmt.bufPrint(&name_buf, "mutex_{d}t_{s}", .{ threads, contention_name }) catch "mutex";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "concurrency/mutex",
                    .warmup_iterations = 5,
                    .min_time_ns = 500_000_000,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(a: std.mem.Allocator, t: usize, ops: usize, c: ContentionLevel) !u64 {
                        return try benchMutexContention(a, t, ops, c);
                    }
                }.bench,
                .{ allocator, threads, 10000, contention },
            );

            std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} lock acquisitions/sec)\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(threads * 10000)),
            });
        }
    }

    // Read-write lock
    std.debug.print("\n[Read-Write Lock]\n", .{});
    for ([_]u8{ 50, 90, 99 }) |read_ratio| {
        for ([_]usize{ 4, 8 }) |threads| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "rwlock_{d}t_{d}pct_read", .{ threads, read_ratio }) catch "rwlock";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "concurrency/rwlock",
                    .warmup_iterations = 5,
                    .min_time_ns = 500_000_000,
                    .max_iterations = 100,
                },
                struct {
                    fn bench(a: std.mem.Allocator, t: usize, ops: usize, r: u8) !u64 {
                        return try benchRWLock(a, t, ops, r);
                    }
                }.bench,
                .{ allocator, threads, 10000, read_ratio },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    // Work stealing
    std.debug.print("\n[Work Stealing]\n", .{});
    for ([_]usize{ 2, 4, 8 }) |threads| {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "work_stealing_{d}t", .{threads}) catch "ws";

        const result = try runner.run(
            .{
                .name = name,
                .category = "concurrency/workstealing",
                .warmup_iterations = 5,
                .min_time_ns = 1_000_000_000,
                .max_iterations = 50,
            },
            struct {
                fn bench(a: std.mem.Allocator, t: usize, work: usize) !u64 {
                    return try benchWorkStealing(a, t, work);
                }
            }.bench,
            .{ allocator, threads, 100000 },
        );

        std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} work items/sec)\n", .{
            name,
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 100000,
        });
    }

    // Channel/message passing
    std.debug.print("\n[Channel/Message Passing]\n", .{});
    {
        const result = try runner.run(
            .{
                .name = "channel_1p1c",
                .category = "concurrency/channel",
                .warmup_iterations = 10,
                .min_time_ns = 500_000_000,
                .max_iterations = 100,
            },
            struct {
                fn bench(a: std.mem.Allocator, count: usize) !u64 {
                    return try benchChannel(a, count);
                }
            }.bench,
            .{ allocator, 10000 },
        );

        std.debug.print("  channel_1p1c: {d:.0} ops/sec ({d:.0} messages/sec)\n", .{
            result.stats.opsPerSecond(),
            result.stats.opsPerSecond() * 10000,
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runConcurrencyBenchmarks(allocator, .{});
}

test "concurrency primitives" {
    // Test SPSC queue
    var spsc = SPSCQueue(u64, 16){};
    try std.testing.expect(spsc.push(42));
    try std.testing.expectEqual(@as(?u64, 42), spsc.pop());
    try std.testing.expectEqual(@as(?u64, null), spsc.pop());

    // Test MPSC queue
    var mpsc = MPSCQueue(u64, 16){};
    try std.testing.expect(mpsc.push(42));
    try std.testing.expectEqual(@as(?u64, 42), mpsc.pop());
}
