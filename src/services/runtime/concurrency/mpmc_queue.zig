//! Lock-Free Multi-Producer Multi-Consumer (MPMC) Bounded Queue
//!
//! A high-performance, lock-free queue that supports multiple producers
//! and multiple consumers concurrently. Uses a circular buffer with
//! per-slot sequence numbers to ensure correct ordering.
//!
//! ## Complexity
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | `push()` | O(1) | Single CAS, may spin on contention |
//! | `pop()` | O(1) | Single CAS, may spin on contention |
//! | `tryPush()` | O(1) | Non-blocking variant |
//! | `len()` | O(1) | Approximate (non-atomic read of both positions) |
//!
//! ## Memory
//!
//! - O(capacity) fixed allocation at init
//! - No dynamic resizing (bounded queue)
//!
//! ## Algorithm
//!
//! Based on the Dmitry Vyukov bounded MPMC queue:
//! - Each slot has a sequence number
//! - Producers atomically claim slots by CAS on sequence
//! - Consumers atomically consume by CAS on sequence
//! - No ABA problem due to monotonically increasing sequences
//!
//! ## Properties
//!
//! - Lock-free for all operations
//! - Bounded memory usage (fixed capacity)
//! - FIFO ordering
//! - Wait-free for common cases (single CAS success)
//!
//! ## Usage
//!
//! ```zig
//! var queue = try MpmcQueue(Task).init(allocator, 1024);
//! defer queue.deinit();
//!
//! // Any thread can push
//! try queue.push(task);
//!
//! // Any thread can pop
//! if (queue.pop()) |task| {
//!     // process task
//! }
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// MPMC bounded queue.
pub fn MpmcQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Slot in the circular buffer
        const Slot = struct {
            /// Sequence number for this slot
            sequence: std.atomic.Value(usize),
            /// Value stored in this slot
            value: T,
        };

        allocator: std.mem.Allocator,
        /// Circular buffer of slots
        buffer: []Slot,
        /// Buffer capacity (power of 2)
        capacity: usize,
        /// Mask for fast modulo
        mask: usize,
        /// Enqueue position
        enqueue_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        /// Dequeue position
        dequeue_pos: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        /// Initialize with the given capacity (must be power of 2).
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            if (capacity == 0 or !std.math.isPowerOfTwo(capacity)) {
                return error.InvalidCapacity;
            }

            const buffer = try allocator.alloc(Slot, capacity);

            // Initialize sequence numbers
            for (buffer, 0..) |*slot, i| {
                slot.sequence = std.atomic.Value(usize).init(i);
                slot.value = undefined;
            }

            return Self{
                .allocator = allocator,
                .buffer = buffer,
                .capacity = capacity,
                .mask = capacity - 1,
            };
        }

        /// Deinitialize and free memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
            self.* = undefined;
        }

        /// Push a value to the queue.
        /// Returns error.QueueFull if the queue is at capacity.
        pub fn push(self: *Self, value: T) !void {
            var pos = self.enqueue_pos.load(.monotonic);

            while (true) {
                const slot = &self.buffer[pos & self.mask];
                const seq = slot.sequence.load(.acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(pos));

                if (diff == 0) {
                    // Slot is ready for writing
                    if (self.enqueue_pos.cmpxchgWeak(
                        pos,
                        pos + 1,
                        .monotonic,
                        .monotonic,
                    )) |new_pos| {
                        pos = new_pos;
                        continue;
                    }

                    // Successfully claimed the slot
                    slot.value = value;
                    slot.sequence.store(pos + 1, .release);
                    return;
                } else if (diff < 0) {
                    // Queue is full
                    return error.QueueFull;
                } else {
                    // Slot has been enqueued but not dequeued yet
                    pos = self.enqueue_pos.load(.monotonic);
                }
            }
        }

        /// Try to push without blocking.
        /// Returns false if queue is full.
        pub fn tryPush(self: *Self, value: T) bool {
            return if (self.push(value)) |_| true else |_| false;
        }

        /// Pop a value from the queue.
        /// Returns null if the queue is empty.
        pub fn pop(self: *Self) ?T {
            var pos = self.dequeue_pos.load(.monotonic);

            while (true) {
                const slot = &self.buffer[pos & self.mask];
                const seq = slot.sequence.load(.acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(pos + 1));

                if (diff == 0) {
                    // Slot is ready for reading
                    if (self.dequeue_pos.cmpxchgWeak(
                        pos,
                        pos + 1,
                        .monotonic,
                        .monotonic,
                    )) |new_pos| {
                        pos = new_pos;
                        continue;
                    }

                    // Successfully claimed the slot
                    const value = slot.value;
                    slot.sequence.store(pos + self.capacity, .release);
                    return value;
                } else if (diff < 0) {
                    // Queue is empty
                    return null;
                } else {
                    // Slot is being written
                    pos = self.dequeue_pos.load(.monotonic);
                }
            }
        }

        /// Check if the queue is empty (approximate).
        pub fn isEmpty(self: *Self) bool {
            const enq = self.enqueue_pos.load(.acquire);
            const deq = self.dequeue_pos.load(.acquire);
            return enq <= deq;
        }

        /// Get the current size (approximate).
        pub fn len(self: *Self) usize {
            const enq = self.enqueue_pos.load(.acquire);
            const deq = self.dequeue_pos.load(.acquire);
            return if (enq >= deq) enq - deq else 0;
        }

        /// Check if the queue is full (approximate).
        pub fn isFull(self: *Self) bool {
            return self.len() >= self.capacity;
        }
    };
}

/// Blocking MPMC queue with wait/notify support.
pub fn BlockingMpmcQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        inner: MpmcQueue(T),
        /// Condition for waiting pushers
        push_cond: std.Thread.Condition = .{},
        /// Condition for waiting poppers
        pop_cond: std.Thread.Condition = .{},
        /// Mutex for condition variables
        mutex: sync.Mutex = .{},
        /// Whether the queue is closed
        closed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            return Self{
                .inner = try MpmcQueue(T).init(allocator, capacity),
            };
        }

        pub fn deinit(self: *Self) void {
            self.close();
            self.inner.deinit();
            self.* = undefined;
        }

        /// Push with blocking if full.
        pub fn push(self: *Self, value: T) !void {
            while (true) {
                if (self.closed.load(.acquire)) {
                    return error.QueueClosed;
                }

                if (self.inner.push(value)) |_| {
                    // Notify waiting poppers
                    self.mutex.lock();
                    self.pop_cond.signal();
                    self.mutex.unlock();
                    return;
                } else |err| {
                    if (err == error.QueueFull) {
                        // Wait for space
                        self.mutex.lock();
                        self.push_cond.wait(&self.mutex);
                        self.mutex.unlock();
                    } else {
                        return err;
                    }
                }
            }
        }

        /// Pop with blocking if empty.
        pub fn pop(self: *Self) ?T {
            while (true) {
                if (self.inner.pop()) |value| {
                    // Notify waiting pushers
                    self.mutex.lock();
                    self.push_cond.signal();
                    self.mutex.unlock();
                    return value;
                }

                if (self.closed.load(.acquire)) {
                    return null;
                }

                // Wait for items
                self.mutex.lock();
                self.pop_cond.wait(&self.mutex);
                self.mutex.unlock();
            }
        }

        /// Try push without blocking.
        pub fn tryPush(self: *Self, value: T) bool {
            if (self.closed.load(.acquire)) return false;
            return self.inner.tryPush(value);
        }

        /// Try pop without blocking.
        pub fn tryPop(self: *Self) ?T {
            return self.inner.pop();
        }

        /// Close the queue, waking all waiters.
        pub fn close(self: *Self) void {
            self.closed.store(true, .release);
            self.mutex.lock();
            self.push_cond.broadcast();
            self.pop_cond.broadcast();
            self.mutex.unlock();
        }

        /// Check if closed.
        pub fn isClosed(self: *Self) bool {
            return self.closed.load(.acquire);
        }

        pub fn isEmpty(self: *Self) bool {
            return self.inner.isEmpty();
        }

        pub fn len(self: *Self) usize {
            return self.inner.len();
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "mpmc queue basic" {
    var queue = try MpmcQueue(u32).init(std.testing.allocator, 16);
    defer queue.deinit();

    try queue.push(1);
    try queue.push(2);
    try queue.push(3);

    try std.testing.expectEqual(@as(?u32, 1), queue.pop());
    try std.testing.expectEqual(@as(?u32, 2), queue.pop());
    try std.testing.expectEqual(@as(?u32, 3), queue.pop());
    try std.testing.expectEqual(@as(?u32, null), queue.pop());
}

test "mpmc queue full" {
    var queue = try MpmcQueue(u32).init(std.testing.allocator, 4);
    defer queue.deinit();

    try queue.push(1);
    try queue.push(2);
    try queue.push(3);
    try queue.push(4);

    // Queue is full
    try std.testing.expectError(error.QueueFull, queue.push(5));

    // Pop one, now we can push
    _ = queue.pop();
    try queue.push(5);
}

test "mpmc queue concurrent" {
    const producer_count = 4;
    const consumer_count = 4;
    const items_per_producer = 100;

    var queue = try MpmcQueue(u32).init(std.testing.allocator, 256);
    defer queue.deinit();

    var produced = std.atomic.Value(usize).init(0);
    var consumed = std.atomic.Value(usize).init(0);

    var producers: [producer_count]std.Thread = undefined;
    var consumers: [consumer_count]std.Thread = undefined;

    // Start producers
    for (&producers) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(q: *MpmcQueue(u32), count: *std.atomic.Value(usize)) !void {
                var i: u32 = 0;
                while (i < items_per_producer) {
                    if (q.push(i)) |_| {
                        i += 1;
                        _ = count.fetchAdd(1, .monotonic);
                    } else |_| {
                        std.atomic.spinLoopHint();
                    }
                }
            }
        }.run, .{ &queue, &produced });
    }

    // Start consumers
    for (&consumers) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(q: *MpmcQueue(u32), count: *std.atomic.Value(usize), target: usize) void {
                while (count.load(.acquire) < target) {
                    if (q.pop()) |_| {
                        _ = count.fetchAdd(1, .monotonic);
                    } else {
                        std.atomic.spinLoopHint();
                    }
                }
            }
        }.run, .{ &queue, &consumed, producer_count * items_per_producer });
    }

    // Wait for all
    for (&producers) |*t| t.join();
    for (&consumers) |*t| t.join();

    // Drain any remaining
    while (queue.pop()) |_| {
        _ = consumed.fetchAdd(1, .monotonic);
    }

    try std.testing.expectEqual(
        @as(usize, producer_count * items_per_producer),
        produced.load(.acquire),
    );
}

test "blocking mpmc queue" {
    var queue = try BlockingMpmcQueue(u32).init(std.testing.allocator, 8);
    defer queue.deinit();

    // Non-blocking operations
    try std.testing.expect(queue.tryPush(1));
    try std.testing.expect(queue.tryPush(2));
    try std.testing.expectEqual(@as(?u32, 1), queue.tryPop());
    try std.testing.expectEqual(@as(?u32, 2), queue.tryPop());
    try std.testing.expectEqual(@as(?u32, null), queue.tryPop());
}
