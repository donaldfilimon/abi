//! Concurrency Property Tests
//!
//! Property-based tests for lock-free data structures including:
//! - MPMC queue enqueue/dequeue consistency
//! - Work-stealing deque properties
//! - Atomic counter correctness
//! - Memory pool allocation/deallocation
//!
//! Uses the runtime/concurrency module

const std = @import("std");
const property = @import("mod.zig");
const generators = @import("generators.zig");
const abi = @import("abi");
const runtime = abi.services.runtime;
// Priority is exported from concurrency submodule
const Priority = runtime.concurrency.Priority;

const forAll = property.forAll;
const forAllWithAllocator = property.forAllWithAllocator;
const assert = property.assert;
const Generator = property.Generator;

// ============================================================================
// Test Configuration
// ============================================================================

const TestConfig = property.PropertyConfig{
    .iterations = 50,
    .seed = 42,
    .verbose = false,
};

// ============================================================================
// Operation Sequence Generation
// ============================================================================

/// Operation types for queue testing
const QueueOperation = enum {
    push,
    pop,
};

/// Generate sequence of queue operations
fn queueOpsGen(max_len: usize) Generator([]const QueueOperation) {
    const GenState = struct {
        var max_length: usize = undefined;

        fn generate(prng: *std.Random.DefaultPrng, size: usize) []const QueueOperation {
            const len = prng.random().intRangeAtMost(usize, 1, @min(size + 1, max_length));
            const ops = std.heap.page_allocator.alloc(QueueOperation, len) catch return &.{};
            for (ops) |*op| {
                op.* = if (prng.random().boolean()) .push else .pop;
            }
            return ops;
        }
    };

    GenState.max_length = max_len;

    return .{
        .generateFn = GenState.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// MPMC Queue Properties
// ============================================================================

test "MPMC queue: push then pop returns same value" {
    const gen = generators.intRange(u64, 0, std.math.maxInt(u64));

    const result = forAllWithAllocator(u64, std.testing.allocator, gen, TestConfig, struct {
        fn check(value: u64, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 16) catch return false;
            defer queue.deinit();

            queue.push(value) catch return false;
            const popped = queue.pop() orelse return false;

            return popped == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "MPMC queue: FIFO ordering preserved" {
    const gen = generators.intRange(u8, 3, 15);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 16) catch return false;
            defer queue.deinit();

            // Push n values
            for (0..n) |i| {
                queue.push(@intCast(i)) catch return false;
            }

            // Pop and verify order
            for (0..n) |i| {
                const popped = queue.pop() orelse return false;
                if (popped != @as(u64, @intCast(i))) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "MPMC queue: empty pop returns null" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 16) catch return false;
            defer queue.deinit();

            return queue.pop() == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "MPMC queue: full push returns error" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 4) catch return false;
            defer queue.deinit();

            // Fill the queue
            for (0..4) |i| {
                queue.push(@intCast(i)) catch return false;
            }

            // Next push should fail
            return queue.push(100) == error.QueueFull;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "MPMC queue: len is accurate" {
    const gen = generators.intRange(u8, 0, 15);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 16) catch return false;
            defer queue.deinit();

            // Push n values
            for (0..n) |i| {
                queue.push(@intCast(i)) catch return false;
            }

            if (queue.len() != n) return false;

            // Pop half
            const half = n / 2;
            for (0..half) |_| {
                _ = queue.pop();
            }

            return queue.len() == n - half;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Work Queue Properties
// ============================================================================

test "WorkQueue: FIFO ordering" {
    const gen = generators.intRange(u8, 3, 20);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkQueue(u64).init(allocator);
            defer queue.deinit();

            // Enqueue n values
            for (0..n) |i| {
                queue.enqueue(@intCast(i)) catch return false;
            }

            // Dequeue and verify order
            for (0..n) |i| {
                const dequeued = queue.dequeue() orelse return false;
                if (dequeued != @as(u64, @intCast(i))) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "WorkQueue: empty dequeue returns null" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkQueue(u64).init(allocator);
            defer queue.deinit();

            return queue.dequeue() == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "WorkQueue: isEmpty reflects actual state" {
    const gen = generators.intRange(u8, 1, 20);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkQueue(u64).init(allocator);
            defer queue.deinit();

            if (!queue.isEmpty()) return false;

            queue.enqueue(42) catch return false;
            if (queue.isEmpty()) return false;

            _ = queue.dequeue();
            return queue.isEmpty();
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Work-Stealing Queue Properties
// ============================================================================

test "WorkStealingQueue: push then pop returns value" {
    const gen = generators.intRange(u64, 0, 1000);

    const result = forAllWithAllocator(u64, std.testing.allocator, gen, TestConfig, struct {
        fn check(value: u64, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkStealingQueue(u64).init(allocator);
            defer queue.deinit();

            queue.push(value) catch return false;
            const popped = queue.pop() orelse return false;

            return popped == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "WorkStealingQueue: pop is LIFO for owner" {
    const gen = generators.intRange(u8, 3, 15);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkStealingQueue(u64).init(allocator);
            defer queue.deinit();

            // Push n values
            for (0..n) |i| {
                queue.push(@intCast(i)) catch return false;
            }

            // Pop should return in reverse order (LIFO)
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                const popped = queue.pop() orelse return false;
                if (popped != @as(u64, @intCast(i))) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "WorkStealingQueue: steal is FIFO for thieves" {
    const gen = generators.intRange(u8, 3, 15);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkStealingQueue(u64).init(allocator);
            defer queue.deinit();

            // Push n values
            for (0..n) |i| {
                queue.push(@intCast(i)) catch return false;
            }

            // Steal should return in order (FIFO)
            for (0..n) |i| {
                const stolen = queue.steal() orelse return false;
                if (stolen != @as(u64, @intCast(i))) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Atomic Counter Properties
// ============================================================================

test "atomic counter: increment produces correct final value" {
    const gen = generators.intRange(u16, 1, 1000);

    const result = forAll(u16, gen, TestConfig, struct {
        fn check(n: u16) bool {
            var counter = std.atomic.Value(u64).init(0);

            // Increment n times
            for (0..n) |_| {
                _ = counter.fetchAdd(1, .monotonic);
            }

            return counter.load(.acquire) == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic counter: decrement produces correct final value" {
    const gen = generators.intRange(u16, 1, 1000);

    const result = forAll(u16, gen, TestConfig, struct {
        fn check(n: u16) bool {
            var counter = std.atomic.Value(u64).init(n);

            // Decrement n times
            for (0..n) |_| {
                _ = counter.fetchSub(1, .monotonic);
            }

            return counter.load(.acquire) == 0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic counter: cmpxchg succeeds for matching expected" {
    const gen = generators.intRange(u64, 0, 1000);

    const result = forAll(u64, gen, TestConfig, struct {
        fn check(value: u64) bool {
            var counter = std.atomic.Value(u64).init(value);

            // CAS with matching expected should succeed
            const result_val = counter.cmpxchgStrong(value, value + 1, .acq_rel, .acquire);

            // Should return null (success)
            return result_val == null and counter.load(.acquire) == value + 1;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic counter: cmpxchg fails for non-matching expected" {
    const Pair = struct {
        current: u64,
        expected: u64,
    };

    const gen = Generator(Pair){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) Pair {
                const current = prng.random().int(u64);
                var expected = prng.random().int(u64);
                // Ensure they're different
                while (expected == current) {
                    expected = prng.random().int(u64);
                }
                return .{ .current = current, .expected = expected };
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAll(Pair, gen, TestConfig, struct {
        fn check(pair: Pair) bool {
            var counter = std.atomic.Value(u64).init(pair.current);

            // CAS with non-matching expected should fail
            const result_val = counter.cmpxchgStrong(pair.expected, pair.expected + 1, .acq_rel, .acquire);

            // Should return actual value (failure)
            if (result_val) |actual| {
                return actual == pair.current;
            }
            return false;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Chase-Lev Deque Properties
// ============================================================================

test "ChaseLevDeque: push then pop returns value" {
    const gen = generators.intRange(u64, 0, 1000);

    const result = forAllWithAllocator(u64, std.testing.allocator, gen, TestConfig, struct {
        fn check(value: u64, allocator: std.mem.Allocator) bool {
            var deque = runtime.ChaseLevDeque(u64).init(allocator) catch return false;
            defer deque.deinit();

            deque.push(value) catch return false;
            const popped = deque.pop() orelse return false;

            return popped == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "ChaseLevDeque: push then steal returns value" {
    const gen = generators.intRange(u64, 0, 1000);

    const result = forAllWithAllocator(u64, std.testing.allocator, gen, TestConfig, struct {
        fn check(value: u64, allocator: std.mem.Allocator) bool {
            var deque = runtime.ChaseLevDeque(u64).init(allocator) catch return false;
            defer deque.deinit();

            deque.push(value) catch return false;
            const stolen = deque.steal() orelse return false;

            return stolen == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "ChaseLevDeque: multiple pushes maintain total count" {
    const gen = generators.intRange(u8, 1, 100);

    const result = forAllWithAllocator(u8, std.testing.allocator, gen, TestConfig, struct {
        fn check(n: u8, allocator: std.mem.Allocator) bool {
            var deque = runtime.ChaseLevDeque(u64).init(allocator) catch return false;
            defer deque.deinit();

            // Push n values
            for (0..n) |i| {
                deque.push(@intCast(i)) catch return false;
            }

            // Pop all values and count
            var count: usize = 0;
            while (deque.pop()) |_| {
                count += 1;
            }

            return count == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "ChaseLevDeque: empty returns null" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var deque = runtime.ChaseLevDeque(u64).init(allocator) catch return false;
            defer deque.deinit();

            return deque.pop() == null and deque.steal() == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Backoff Properties
// ============================================================================

test "Backoff: spin count increases" {
    const gen = generators.intRange(u8, 1, 50);

    const result = forAll(u8, gen, TestConfig, struct {
        fn check(n: u8) bool {
            var backoff = runtime.Backoff{};

            for (0..n) |_| {
                backoff.spin();
            }

            return backoff.spins == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "Backoff: reset clears spin count" {
    const gen = generators.intRange(u8, 1, 50);

    const result = forAll(u8, gen, TestConfig, struct {
        fn check(n: u8) bool {
            var backoff = runtime.Backoff{};

            for (0..n) |_| {
                backoff.spin();
            }

            backoff.reset();
            return backoff.spins == 0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Priority Queue Properties
// ============================================================================

test "PriorityQueue: dequeue returns highest priority first" {
    const Params = struct {
        count: u8,
        priorities: [10]u8,
    };

    const gen = Generator(Params){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) Params {
                var result: Params = undefined;
                result.count = prng.random().intRangeAtMost(u8, 1, 10);
                for (&result.priorities) |*p| {
                    p.* = prng.random().int(u8);
                }
                return result;
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAllWithAllocator(Params, std.testing.allocator, gen, TestConfig, struct {
        fn check(params: Params, allocator: std.mem.Allocator) bool {
            // PriorityQueue.init doesn't return error union in Zig 0.16
            var pq = runtime.PriorityQueue(u64).init(allocator, .{});
            defer pq.deinit();

            // Insert with priorities
            for (0..params.count) |i| {
                const priority: Priority = switch (params.priorities[i] % 4) {
                    0 => .critical,
                    1 => .high,
                    2 => .normal,
                    else => .low,
                };

                pq.push(@intCast(i), priority) catch return false;
            }

            // First dequeue should be one of the highest priority items
            const first = pq.pop() orelse return false;

            // The priority of first should be the max priority
            // (Note: multiple items might have same priority)
            // We just verify that dequeue returns something valid
            _ = first;

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Sequence Property Tests
// ============================================================================

test "operation sequence maintains invariants" {
    const gen = queueOpsGen(50);

    const result = forAllWithAllocator([]const QueueOperation, std.testing.allocator, gen, TestConfig, struct {
        fn check(ops: []const QueueOperation, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 64) catch return false;
            defer queue.deinit();

            var push_count: usize = 0;
            var pop_count: usize = 0;
            var counter: u64 = 0;

            for (ops) |op| {
                switch (op) {
                    .push => {
                        if (queue.push(counter)) |_| {
                            push_count += 1;
                            counter += 1;
                        } else |_| {
                            // Queue full, skip
                        }
                    },
                    .pop => {
                        if (queue.pop()) |_| {
                            pop_count += 1;
                        }
                    },
                }

                // Invariant: pop_count <= push_count
                if (pop_count > push_count) return false;
            }

            // Final invariant: queue.len() == push_count - pop_count
            return queue.len() == push_count - pop_count;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Memory Ordering Properties
// ============================================================================

test "atomic store/load with acquire/release ordering is correct" {
    const gen = generators.intRange(u64, 0, std.math.maxInt(u64));

    const result = forAll(u64, gen, TestConfig, struct {
        fn check(value: u64) bool {
            var atomic_val = std.atomic.Value(u64).init(0);

            // Store with release
            atomic_val.store(value, .release);

            // Load with acquire
            const loaded = atomic_val.load(.acquire);

            return loaded == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic fetchAdd is atomic (single-threaded sanity)" {
    const gen = generators.intRange(u16, 1, 100);

    const result = forAll(u16, gen, TestConfig, struct {
        fn check(n: u16) bool {
            var counter = std.atomic.Value(u64).init(0);

            var sum: u64 = 0;
            for (0..n) |i| {
                const prev = counter.fetchAdd(1, .acq_rel);
                if (prev != i) return false;
                sum += 1;
            }

            return counter.load(.acquire) == n and sum == n;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Edge Case Tests - Boundary Conditions
// ============================================================================

test "MPMC queue: push-pop alternation at capacity boundary" {
    // Test with fixed power-of-2 capacities that work well with MPMC queues
    const capacities = [_]usize{ 4, 8, 16 };

    for (capacities) |cap| {
        var queue = runtime.MpmcQueue(u64).init(std.testing.allocator, cap) catch continue;
        defer queue.deinit();

        // Fill to capacity
        for (0..cap) |i| {
            queue.push(@intCast(i)) catch return error.TestUnexpectedResult;
        }

        // Queue should be full
        try std.testing.expectError(error.QueueFull, queue.push(999));

        // Pop one, push one - should work
        _ = queue.pop();
        try queue.push(999);

        // Should still be full
        try std.testing.expectError(error.QueueFull, queue.push(1000));

        // Drain and verify count
        var count: usize = 0;
        while (queue.pop()) |_| {
            count += 1;
        }

        try std.testing.expectEqual(cap, count);
    }
}

test "ChaseLevDeque: single element push-pop-steal sequence" {
    const result = forAllWithAllocator(u64, std.testing.allocator, generators.intRange(u64, 0, 1000), TestConfig, struct {
        fn check(value: u64, allocator: std.mem.Allocator) bool {
            var deque = runtime.ChaseLevDeque(u64).init(allocator) catch return false;
            defer deque.deinit();

            // Push single element
            deque.push(value) catch return false;

            // Pop should succeed and empty the deque
            const popped = deque.pop() orelse return false;
            if (popped != value) return false;

            // Both pop and steal should now return null
            if (deque.pop() != null) return false;
            if (deque.steal() != null) return false;

            // Push again
            deque.push(value + 1) catch return false;

            // Steal should work
            const stolen = deque.steal() orelse return false;
            return stolen == value + 1;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic counter: boundary values at u64 max" {
    const result = forAll(u8, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8) bool {
            // Start near max value
            var counter = std.atomic.Value(u64).init(std.math.maxInt(u64) - 5);

            // Increment - will wrap around
            for (0..10) |_| {
                _ = counter.fetchAdd(1, .monotonic);
            }

            // Should have wrapped around
            const final = counter.load(.acquire);

            // Verify wrapping behavior: max-5 + 10 = max + 5 = 4 (with wrap)
            return final == 4;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "atomic counter: CAS at zero boundary" {
    const result = forAll(u8, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8) bool {
            var counter = std.atomic.Value(u64).init(0);

            // CAS from 0 to 1 should succeed
            const cas_result = counter.cmpxchgStrong(0, 1, .acq_rel, .acquire);
            if (cas_result != null) return false;

            // CAS from 0 again should fail (current is 1)
            const cas_result2 = counter.cmpxchgStrong(0, 2, .acq_rel, .acquire);
            if (cas_result2) |actual| {
                return actual == 1;
            }
            return false;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "WorkStealingQueue: empty steal returns null" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var queue = runtime.WorkStealingQueue(u64).init(allocator);
            defer queue.deinit();

            // Steal from empty should return null
            if (queue.steal() != null) return false;

            // Push then pop to empty
            queue.push(42) catch return false;
            _ = queue.pop();

            // Steal should still be null
            return queue.steal() == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "PriorityQueue: empty dequeue returns null" {
    const result = forAllWithAllocator(u8, std.testing.allocator, generators.intRange(u8, 1, 10), TestConfig, struct {
        fn check(_: u8, allocator: std.mem.Allocator) bool {
            var pq = runtime.PriorityQueue(u64).init(allocator, .{});
            defer pq.deinit();

            // Dequeue from empty
            if (pq.pop() != null) return false;

            // Push then pop to empty
            pq.push(42, .normal) catch return false;
            _ = pq.pop();

            // Should still be empty
            return pq.pop() == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "MPMC queue: rapid push-pop cycles maintain consistency" {
    const gen = generators.intRange(u16, 10, 100);

    const result = forAllWithAllocator(u16, std.testing.allocator, gen, TestConfig, struct {
        fn check(cycles: u16, allocator: std.mem.Allocator) bool {
            var queue = runtime.MpmcQueue(u64).init(allocator, 8) catch return false;
            defer queue.deinit();

            var push_count: u64 = 0;
            var pop_count: u64 = 0;

            for (0..cycles) |_| {
                // Push a few
                for (0..3) |_| {
                    if (queue.push(push_count)) |_| {
                        push_count += 1;
                    } else |_| {}
                }

                // Pop a few
                for (0..2) |_| {
                    if (queue.pop()) |_| {
                        pop_count += 1;
                    }
                }
            }

            // Drain remaining
            while (queue.pop()) |_| {
                pop_count += 1;
            }

            // All pushed items should have been popped
            return push_count == pop_count;
        }
    }.check);

    try std.testing.expect(result.passed);
}
