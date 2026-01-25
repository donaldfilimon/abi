//! Chase-Lev Work-Stealing Deque
//!
//! A lock-free deque designed for work-stealing schedulers. The owner thread
//! pushes and pops from the bottom (LIFO), while thieves steal from the top (FIFO).
//!
//! Based on: "Dynamic Circular Work-Stealing Deque" by Chase and Lev (SPAA 2005)
//!
//! ## Properties
//!
//! - Lock-free for all operations
//! - Owner push/pop: O(1) amortized, no contention with other owner ops
//! - Thief steal: O(1), may contend with owner or other thieves
//! - Dynamic resizing (grows as needed)
//!
//! ## Memory Ordering
//!
//! Uses careful memory ordering to ensure correctness:
//! - bottom: only modified by owner, read by thieves
//! - top: modified by thieves (CAS), read by owner
//! - buffer: array accesses synchronized via bottom/top
//!
//! ## Usage
//!
//! ```zig
//! var deque = ChaseLevDeque(Task).init(allocator);
//! defer deque.deinit();
//!
//! // Owner thread
//! try deque.push(task);
//! const my_task = deque.pop();
//!
//! // Thief threads
//! const stolen = deque.steal();
//! ```

const std = @import("std");

/// Minimum capacity (must be power of 2)
const MIN_CAPACITY: usize = 16;

/// Growth factor when resizing
const GROWTH_FACTOR: usize = 2;

/// Chase-Lev work-stealing deque
pub fn ChaseLevDeque(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Circular buffer for storing items
        const Buffer = struct {
            data: []std.atomic.Value(T),
            mask: usize,

            fn init(allocator: std.mem.Allocator, cap: usize) !*Buffer {
                const buf = try allocator.create(Buffer);
                errdefer allocator.destroy(buf);

                buf.data = try allocator.alloc(std.atomic.Value(T), cap);
                for (buf.data) |*slot| {
                    slot.* = std.atomic.Value(T).init(undefined);
                }
                buf.mask = cap - 1;

                return buf;
            }

            fn deinit(self: *Buffer, allocator: std.mem.Allocator) void {
                allocator.free(self.data);
                allocator.destroy(self);
            }

            fn capacity(self: *const Buffer) usize {
                return self.mask + 1;
            }

            fn get(self: *Buffer, index: isize) T {
                const idx = @as(usize, @intCast(@mod(index, @as(isize, @intCast(self.capacity())))));
                return self.data[idx].load(.acquire);
            }

            fn put(self: *Buffer, index: isize, value: T) void {
                const idx = @as(usize, @intCast(@mod(index, @as(isize, @intCast(self.capacity())))));
                self.data[idx].store(value, .release);
            }

            /// Create a new buffer with double capacity and copy elements.
            fn grow(self: *Buffer, allocator: std.mem.Allocator, bottom: isize, top: isize) !*Buffer {
                const new_capacity = self.capacity() * GROWTH_FACTOR;
                const new_buf = try Buffer.init(allocator, new_capacity);

                // Copy elements from old buffer to new buffer
                var i = top;
                while (i < bottom) : (i += 1) {
                    new_buf.put(i, self.get(i));
                }

                return new_buf;
            }
        };

        allocator: std.mem.Allocator,
        /// Bottom of the deque (modified only by owner)
        bottom: std.atomic.Value(isize) = std.atomic.Value(isize).init(0),
        /// Top of the deque (modified by thieves via CAS)
        top: std.atomic.Value(isize) = std.atomic.Value(isize).init(0),
        /// Current buffer (may be replaced on grow)
        buffer: std.atomic.Value(*Buffer),
        /// Old buffers waiting for reclamation
        old_buffers: std.ArrayListUnmanaged(*Buffer) = .empty,

        /// Initialize a new deque with default capacity.
        pub fn init(allocator: std.mem.Allocator) !Self {
            return initWithCapacity(allocator, MIN_CAPACITY);
        }

        /// Initialize with a specific starting capacity (must be power of 2).
        pub fn initWithCapacity(allocator: std.mem.Allocator, initial_capacity: usize) !Self {
            if (initial_capacity < MIN_CAPACITY or !std.math.isPowerOfTwo(initial_capacity)) {
                return error.InvalidCapacity;
            }

            const buf = try Buffer.init(allocator, initial_capacity);

            return Self{
                .allocator = allocator,
                .buffer = std.atomic.Value(*Buffer).init(buf),
            };
        }

        /// Deinitialize and free all resources.
        pub fn deinit(self: *Self) void {
            // Free current buffer
            self.buffer.load(.acquire).deinit(self.allocator);

            // Free any old buffers
            for (self.old_buffers.items) |buf| {
                buf.deinit(self.allocator);
            }
            self.old_buffers.deinit(self.allocator);

            self.* = undefined;
        }

        /// Push a value onto the bottom of the deque (owner only).
        pub fn push(self: *Self, value: T) !void {
            const b = self.bottom.load(.monotonic);
            const t = self.top.load(.acquire);
            var buf = self.buffer.load(.monotonic);

            const size = b - t;
            if (size >= @as(isize, @intCast(buf.capacity() - 1))) {
                // Need to grow
                buf = try buf.grow(self.allocator, b, t);

                // Keep old buffer for later reclamation
                const old_buf = self.buffer.swap(buf, .release);
                try self.old_buffers.append(self.allocator, old_buf);
            }

            buf.put(b, value);
            self.bottom.store(b + 1, .release); // release fence via store
        }

        /// Pop a value from the bottom of the deque (owner only).
        /// Returns null if the deque is empty.
        pub fn pop(self: *Self) ?T {
            const b = self.bottom.load(.monotonic) - 1;
            const buf = self.buffer.load(.monotonic);
            self.bottom.store(b, .seq_cst); // seq_cst provides fence

            const t = self.top.load(.seq_cst); // seq_cst load

            if (t <= b) {
                // Non-empty
                const value = buf.get(b);

                if (t == b) {
                    // Last element - race with thieves
                    if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |_| {
                        // Lost the race
                        self.bottom.store(t + 1, .monotonic);
                        return null;
                    }
                    self.bottom.store(t + 1, .monotonic);
                }

                return value;
            } else {
                // Empty
                self.bottom.store(t, .monotonic);
                return null;
            }
        }

        /// Steal a value from the top of the deque (thief threads).
        /// Returns null if the deque is empty.
        pub fn steal(self: *Self) ?T {
            const t = self.top.load(.seq_cst); // seq_cst provides fence
            const b = self.bottom.load(.seq_cst);

            if (t < b) {
                // Non-empty
                const buf = self.buffer.load(.acquire);
                const value = buf.get(t);

                if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |_| {
                    // Lost race with another thief or owner
                    return null;
                }

                return value;
            }

            return null;
        }

        /// Get the current size of the deque (approximate for thieves).
        pub fn len(self: *Self) usize {
            const b = self.bottom.load(.acquire);
            const t = self.top.load(.acquire);
            const diff = b - t;
            return if (diff > 0) @intCast(diff) else 0;
        }

        /// Check if the deque is empty.
        pub fn isEmpty(self: *Self) bool {
            return self.len() == 0;
        }

        /// Get the current capacity.
        pub fn capacity(self: *Self) usize {
            return self.buffer.load(.acquire).capacity();
        }
    };
}

/// Work-stealing scheduler using Chase-Lev deques.
pub fn WorkStealingScheduler(comptime Task: type) type {
    return struct {
        const Self = @This();
        const Deque = ChaseLevDeque(Task);

        allocator: std.mem.Allocator,
        /// Per-worker deques
        worker_deques: []Deque,
        /// Number of workers
        worker_count: usize,
        /// Random state for victim selection
        rng: std.Random.DefaultPrng,

        pub fn init(allocator: std.mem.Allocator, worker_count: usize) !Self {
            const deques = try allocator.alloc(Deque, worker_count);
            errdefer allocator.free(deques);

            for (deques, 0..) |*d, i| {
                d.* = Deque.init(allocator) catch |err| {
                    for (deques[0..i]) |*prev| {
                        prev.deinit();
                    }
                    return err;
                };
            }

            return Self{
                .allocator = allocator,
                .worker_deques = deques,
                .worker_count = worker_count,
                .rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp())),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.worker_deques) |*d| {
                d.deinit();
            }
            self.allocator.free(self.worker_deques);
            self.* = undefined;
        }

        /// Push a task to a worker's local deque.
        pub fn push(self: *Self, worker_id: usize, task: Task) !void {
            try self.worker_deques[worker_id].push(task);
        }

        /// Pop a task from a worker's local deque.
        pub fn pop(self: *Self, worker_id: usize) ?Task {
            return self.worker_deques[worker_id].pop();
        }

        /// Steal a task from another worker's deque.
        /// Tries random victims until finding work or exhausting attempts.
        pub fn steal(self: *Self, worker_id: usize) ?Task {
            const max_attempts = self.worker_count * 2;

            for (0..max_attempts) |_| {
                // Pick a random victim (not self)
                var victim = self.rng.random().uintLessThan(usize, self.worker_count);
                if (victim == worker_id) {
                    victim = (victim + 1) % self.worker_count;
                }

                if (self.worker_deques[victim].steal()) |task| {
                    return task;
                }
            }

            return null;
        }

        /// Get a task for a worker, trying local pop first, then stealing.
        pub fn getTask(self: *Self, worker_id: usize) ?Task {
            // Try local deque first
            if (self.pop(worker_id)) |task| {
                return task;
            }

            // Try stealing from others
            return self.steal(worker_id);
        }

        /// Get total pending tasks across all workers.
        pub fn totalPending(self: *Self) usize {
            var total: usize = 0;
            for (self.worker_deques) |*d| {
                total += d.len();
            }
            return total;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "chase-lev deque basic operations" {
    var deque = try ChaseLevDeque(u32).init(std.testing.allocator);
    defer deque.deinit();

    // Push and pop (LIFO for owner)
    try deque.push(1);
    try deque.push(2);
    try deque.push(3);

    try std.testing.expectEqual(@as(?u32, 3), deque.pop());
    try std.testing.expectEqual(@as(?u32, 2), deque.pop());
    try std.testing.expectEqual(@as(?u32, 1), deque.pop());
    try std.testing.expectEqual(@as(?u32, null), deque.pop());
}

test "chase-lev deque steal operations" {
    var deque = try ChaseLevDeque(u32).init(std.testing.allocator);
    defer deque.deinit();

    try deque.push(1);
    try deque.push(2);
    try deque.push(3);

    // Steal gets from top (FIFO for thieves)
    try std.testing.expectEqual(@as(?u32, 1), deque.steal());
    try std.testing.expectEqual(@as(?u32, 2), deque.steal());
    try std.testing.expectEqual(@as(?u32, 3), deque.steal());
    try std.testing.expectEqual(@as(?u32, null), deque.steal());
}

test "chase-lev deque grow" {
    var deque = try ChaseLevDeque(u32).initWithCapacity(std.testing.allocator, 16);
    defer deque.deinit();

    // Push more than initial capacity
    for (0..32) |i| {
        try deque.push(@intCast(i));
    }

    try std.testing.expect(deque.capacity() >= 32);
    try std.testing.expectEqual(@as(usize, 32), deque.len());

    // Verify LIFO order for pop
    var expected: u32 = 31;
    while (deque.pop()) |val| {
        try std.testing.expectEqual(expected, val);
        if (expected > 0) expected -= 1;
    }
}

test "chase-lev deque concurrent push-pop-steal" {
    var deque = try ChaseLevDeque(u32).init(std.testing.allocator);
    defer deque.deinit();

    const producer_count = 100;

    // Producer thread (owner)
    const producer = try std.Thread.spawn(.{}, struct {
        fn run(d: *ChaseLevDeque(u32)) !void {
            for (0..producer_count) |i| {
                try d.push(@intCast(i));
            }
        }
    }.run, .{&deque});

    // Thief threads
    const thief_count = 3;
    var thieves: [thief_count]std.Thread = undefined;
    var stolen_counts: [thief_count]std.atomic.Value(usize) = undefined;

    for (&stolen_counts) |*s| {
        s.* = std.atomic.Value(usize).init(0);
    }

    for (&thieves, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(d: *ChaseLevDeque(u32), count: *std.atomic.Value(usize)) void {
                var stolen: usize = 0;
                for (0..200) |_| {
                    if (d.steal()) |_| {
                        stolen += 1;
                    }
                    std.atomic.spinLoopHint();
                }
                count.store(stolen, .release);
            }
        }.run, .{ &deque, &stolen_counts[i] });
    }

    // Wait for producer
    producer.join();

    // Wait for thieves
    for (&thieves) |*t| {
        t.join();
    }

    // Pop remaining items
    var popped: usize = 0;
    while (deque.pop()) |_| {
        popped += 1;
    }

    // Count total stolen
    var total_stolen: usize = 0;
    for (&stolen_counts) |*s| {
        total_stolen += s.load(.acquire);
    }

    // Total should equal producer_count
    try std.testing.expectEqual(@as(usize, producer_count), popped + total_stolen);
}

test "work-stealing scheduler basic" {
    var scheduler = try WorkStealingScheduler(u32).init(std.testing.allocator, 4);
    defer scheduler.deinit();

    // Push to worker 0
    try scheduler.push(0, 100);
    try scheduler.push(0, 200);

    // Pop from worker 0
    try std.testing.expectEqual(@as(?u32, 200), scheduler.pop(0));

    // Push to worker 1
    try scheduler.push(1, 300);

    // Worker 0 can steal from worker 1
    // Note: steal is probabilistic, so we just verify it doesn't crash
    _ = scheduler.steal(0);

    try std.testing.expectEqual(@as(usize, 2), scheduler.totalPending());
}
