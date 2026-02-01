//! Priority Queue for Task Scheduling
//!
//! Thread-safe priority queues for task scheduling with support for
//! multiple priority levels, aging (starvation prevention), and
//! deadline-based scheduling.
//!
//! ## Available Queue Types
//!
//! | Queue | Use Case |
//! |-------|----------|
//! | `PriorityQueue` | General priority-based scheduling |
//! | `MultilevelQueue` | Multi-level feedback scheduling (CPU-style) |
//! | `DeadlineQueue` | Earliest-deadline-first (EDF) scheduling |
//!
//! ## Priority Levels
//!
//! Five priority levels are available (from highest to lowest):
//!
//! | Priority | Weight | Use Case |
//! |----------|--------|----------|
//! | `.critical` | 16 | System-critical tasks, immediate execution |
//! | `.high` | 8 | Time-sensitive user operations |
//! | `.normal` | 4 | Standard tasks (default) |
//! | `.low` | 2 | Background processing |
//! | `.background` | 1 | Idle-time tasks |
//!
//! ## Aging (Starvation Prevention)
//!
//! Low-priority tasks are automatically boosted over time to prevent
//! starvation. Configure via `PriorityQueueConfig`:
//!
//! ```zig
//! const config = PriorityQueueConfig{
//!     .aging_enabled = true,
//!     .aging_threshold = 100,  // Ticks before boost
//!     .max_age_boost = 2,      // Max priority levels to boost
//! };
//! var queue = PriorityQueue(Task).init(allocator, config);
//!
//! // Call periodically to advance aging
//! queue.tick();
//! ```
//!
//! ## Fair Scheduling
//!
//! When `fair_scheduling = true` (default), tasks with the same priority
//! are processed in FIFO order, ensuring fairness.
//!
//! ## Usage Example
//!
//! ```zig
//! var queue = PriorityQueue(Task).init(allocator, .{});
//! defer queue.deinit();
//!
//! // Push with priority
//! try queue.push(critical_task, .critical);
//! try queue.push(normal_task, .normal);
//! try queue.push(background_task, .background);
//!
//! // Pop returns highest priority first
//! const task = queue.pop(); // Gets critical_task
//!
//! // Check statistics
//! const stats = queue.getStats();
//! std.debug.print("Queue size: {}, Critical: {}\n", .{
//!     stats.size,
//!     stats.critical_count,
//! });
//! ```
//!
//! ## Thread Safety
//!
//! All operations are protected by a mutex. For high-contention scenarios,
//! consider using per-worker queues with work-stealing instead.

const std = @import("std");

/// Task priority levels.
pub const Priority = enum(u8) {
    /// Critical priority - immediate execution.
    critical = 0,
    /// High priority.
    high = 1,
    /// Normal priority (default).
    normal = 2,
    /// Low priority.
    low = 3,
    /// Background priority - execute when idle.
    background = 4,

    pub fn toWeight(self: Priority) u32 {
        return switch (self) {
            .critical => 16,
            .high => 8,
            .normal => 4,
            .low => 2,
            .background => 1,
        };
    }

    pub fn fromInt(value: u8) Priority {
        return switch (value) {
            0 => .critical,
            1 => .high,
            2 => .normal,
            3 => .low,
            else => .background,
        };
    }
};

/// Priority queue configuration.
pub const PriorityQueueConfig = struct {
    /// Enable priority aging (prevents starvation).
    aging_enabled: bool = true,
    /// Ticks before priority is boosted.
    aging_threshold: u32 = 100,
    /// Maximum priority boost from aging.
    max_age_boost: u8 = 2,
    /// Use fair scheduling within same priority.
    fair_scheduling: bool = true,
};

/// A prioritized item with metadata.
pub fn PrioritizedItem(comptime T: type) type {
    return struct {
        item: T,
        priority: Priority,
        sequence: u64,
        insert_tick: u64,
        age_boost: u8,

        const Self = @This();

        /// Get effective priority considering aging.
        pub fn effectivePriority(self: Self, current_tick: u64, config: PriorityQueueConfig) Priority {
            if (!config.aging_enabled) return self.priority;

            const age = current_tick -| self.insert_tick;
            const boost_count = @min(age / config.aging_threshold, config.max_age_boost);
            const boosted = @intFromEnum(self.priority) -| @as(u8, @intCast(boost_count));
            return Priority.fromInt(boosted);
        }

        /// Compare for heap ordering (lower effective priority value = higher priority).
        pub fn lessThan(self: Self, other: Self, current_tick: u64, config: PriorityQueueConfig) bool {
            const self_eff = @intFromEnum(self.effectivePriority(current_tick, config));
            const other_eff = @intFromEnum(other.effectivePriority(current_tick, config));

            if (self_eff != other_eff) {
                return self_eff < other_eff;
            }

            // Same effective priority - use FIFO within priority
            if (config.fair_scheduling) {
                return self.sequence < other.sequence;
            }

            return false;
        }
    };
}

/// Thread-safe priority queue.
pub fn PriorityQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        const Item = PrioritizedItem(T);

        allocator: std.mem.Allocator,
        config: PriorityQueueConfig,
        items: std.ArrayListUnmanaged(Item),
        mutex: std.Thread.Mutex,
        sequence_counter: std.atomic.Value(u64),
        tick_counter: std.atomic.Value(u64),
        total_pushed: std.atomic.Value(u64),
        total_popped: std.atomic.Value(u64),

        /// Initialize a new priority queue.
        pub fn init(allocator: std.mem.Allocator, config: PriorityQueueConfig) Self {
            return .{
                .allocator = allocator,
                .config = config,
                .items = .{},
                .mutex = .{},
                .sequence_counter = std.atomic.Value(u64).init(0),
                .tick_counter = std.atomic.Value(u64).init(0),
                .total_pushed = std.atomic.Value(u64).init(0),
                .total_popped = std.atomic.Value(u64).init(0),
            };
        }

        /// Deinitialize the queue.
        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        /// Push an item with the given priority.
        pub fn push(self: *Self, item: T, priority: Priority) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const seq = self.sequence_counter.fetchAdd(1, .monotonic);
            const current_tick = self.tick_counter.load(.monotonic);

            const prioritized = Item{
                .item = item,
                .priority = priority,
                .sequence = seq,
                .insert_tick = current_tick,
                .age_boost = 0,
            };

            try self.items.append(self.allocator, prioritized);
            self.siftUp(self.items.items.len - 1);
            _ = self.total_pushed.fetchAdd(1, .monotonic);
        }

        /// Push an item with normal priority.
        pub fn pushDefault(self: *Self, item: T) !void {
            return self.push(item, .normal);
        }

        /// Pop the highest priority item.
        pub fn pop(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.items.items.len == 0) return null;

            const result = self.items.items[0].item;

            if (self.items.items.len == 1) {
                self.items.clearRetainingCapacity();
            } else {
                self.items.items[0] = self.items.items[self.items.items.len - 1];
                self.items.items.len -= 1;
                self.siftDown(0);
            }

            _ = self.total_popped.fetchAdd(1, .monotonic);
            return result;
        }

        /// Peek at the highest priority item without removing.
        pub fn peek(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.items.items.len == 0) return null;
            return self.items.items[0].item;
        }

        /// Get the number of items in the queue.
        pub fn len(self: *Self) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.items.items.len;
        }

        /// Check if the queue is empty.
        pub fn isEmpty(self: *Self) bool {
            return self.len() == 0;
        }

        /// Advance the tick counter (call periodically for aging).
        pub fn tick(self: *Self) void {
            _ = self.tick_counter.fetchAdd(1, .monotonic);
        }

        /// Get queue statistics.
        pub fn getStats(self: *Self) QueueStats {
            self.mutex.lock();
            defer self.mutex.unlock();

            var priority_counts = [_]usize{0} ** 5;
            for (self.items.items) |item| {
                priority_counts[@intFromEnum(item.priority)] += 1;
            }

            return .{
                .size = self.items.items.len,
                .total_pushed = self.total_pushed.load(.monotonic),
                .total_popped = self.total_popped.load(.monotonic),
                .current_tick = self.tick_counter.load(.monotonic),
                .critical_count = priority_counts[0],
                .high_count = priority_counts[1],
                .normal_count = priority_counts[2],
                .low_count = priority_counts[3],
                .background_count = priority_counts[4],
            };
        }

        /// Clear all items from the queue.
        pub fn clear(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.items.clearRetainingCapacity();
        }

        /// Drain all items as a slice (caller owns memory).
        pub fn drain(self: *Self) ![]T {
            self.mutex.lock();
            defer self.mutex.unlock();

            var result = try self.allocator.alloc(T, self.items.items.len);
            var i: usize = 0;

            while (self.items.items.len > 0) : (i += 1) {
                result[i] = self.items.items[0].item;

                if (self.items.items.len == 1) {
                    self.items.clearRetainingCapacity();
                } else {
                    self.items.items[0] = self.items.items[self.items.items.len - 1];
                    self.items.items.len -= 1;
                    self.siftDown(0);
                }
            }

            return result[0..i];
        }

        // Heap operations
        fn siftUp(self: *Self, start_idx: usize) void {
            var idx = start_idx;
            const current_tick = self.tick_counter.load(.monotonic);

            while (idx > 0) {
                const parent = (idx - 1) / 2;
                if (self.items.items[idx].lessThan(self.items.items[parent], current_tick, self.config)) {
                    std.mem.swap(Item, &self.items.items[idx], &self.items.items[parent]);
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        fn siftDown(self: *Self, start_idx: usize) void {
            var idx = start_idx;
            const current_tick = self.tick_counter.load(.monotonic);
            const items_len = self.items.items.len;

            while (true) {
                var smallest = idx;
                const left = 2 * idx + 1;
                const right = 2 * idx + 2;

                if (left < items_len and
                    self.items.items[left].lessThan(self.items.items[smallest], current_tick, self.config))
                {
                    smallest = left;
                }

                if (right < items_len and
                    self.items.items[right].lessThan(self.items.items[smallest], current_tick, self.config))
                {
                    smallest = right;
                }

                if (smallest == idx) break;

                std.mem.swap(Item, &self.items.items[idx], &self.items.items[smallest]);
                idx = smallest;
            }
        }
    };
}

/// Queue statistics.
pub const QueueStats = struct {
    size: usize,
    total_pushed: u64,
    total_popped: u64,
    current_tick: u64,
    critical_count: usize,
    high_count: usize,
    normal_count: usize,
    low_count: usize,
    background_count: usize,
};

/// Multi-level feedback queue for adaptive scheduling.
pub fn MultilevelQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        const Queue = PriorityQueue(T);

        allocator: std.mem.Allocator,
        levels: [5]Queue,
        time_slices: [5]u32,

        /// Initialize with default time slices.
        pub fn init(allocator: std.mem.Allocator) Self {
            var levels: [5]Queue = undefined;
            for (&levels) |*level| {
                level.* = Queue.init(allocator, .{});
            }

            return .{
                .allocator = allocator,
                .levels = levels,
                .time_slices = .{ 1, 2, 4, 8, 16 }, // Increasing time slices
            };
        }

        /// Deinitialize.
        pub fn deinit(self: *Self) void {
            for (&self.levels) |*level| {
                level.deinit();
            }
            self.* = undefined;
        }

        /// Push to a specific level.
        pub fn push(self: *Self, item: T, level: usize) !void {
            const idx = @min(level, 4);
            try self.levels[idx].pushDefault(item);
        }

        /// Pop from highest non-empty level.
        pub fn pop(self: *Self) ?T {
            for (&self.levels) |*level| {
                if (level.pop()) |item| {
                    return item;
                }
            }
            return null;
        }

        /// Get time slice for a level.
        pub fn getTimeSlice(self: *const Self, level: usize) u32 {
            return self.time_slices[@min(level, 4)];
        }

        /// Total items across all levels.
        pub fn totalLen(self: *Self) usize {
            var total: usize = 0;
            for (&self.levels) |*level| {
                total += level.len();
            }
            return total;
        }
    };
}

/// Deadline-based priority (earliest deadline first).
pub fn DeadlineQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        const DeadlineItem = struct {
            item: T,
            deadline_ns: i128,
            sequence: u64,

            fn lessThan(_: void, a: DeadlineItem, b: DeadlineItem) bool {
                if (a.deadline_ns != b.deadline_ns) {
                    return a.deadline_ns < b.deadline_ns;
                }
                return a.sequence < b.sequence;
            }
        };

        allocator: std.mem.Allocator,
        items: std.ArrayListUnmanaged(DeadlineItem),
        mutex: std.Thread.Mutex,
        sequence: u64,

        /// Initialize.
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .items = .{},
                .mutex = .{},
                .sequence = 0,
            };
        }

        /// Deinitialize.
        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        /// Push with deadline.
        pub fn push(self: *Self, item: T, deadline_ns: i128) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const deadline_item = DeadlineItem{
                .item = item,
                .deadline_ns = deadline_ns,
                .sequence = self.sequence,
            };
            self.sequence += 1;

            try self.items.append(self.allocator, deadline_item);
            self.heapSiftUp(self.items.items.len - 1);
        }

        /// Pop earliest deadline.
        pub fn pop(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.items.items.len == 0) return null;

            const result = self.items.items[0].item;

            if (self.items.items.len == 1) {
                self.items.clearRetainingCapacity();
            } else {
                self.items.items[0] = self.items.items[self.items.items.len - 1];
                self.items.items.len -= 1;
                self.heapSiftDown(0);
            }

            return result;
        }

        /// Peek at earliest deadline.
        pub fn peek(self: *Self) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.items.items.len == 0) return null;
            return self.items.items[0].item;
        }

        /// Get earliest deadline timestamp.
        pub fn peekDeadline(self: *Self) ?i128 {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.items.items.len == 0) return null;
            return self.items.items[0].deadline_ns;
        }

        /// Get queue length.
        pub fn len(self: *Self) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.items.items.len;
        }

        fn heapSiftUp(self: *Self, start_idx: usize) void {
            var idx = start_idx;
            while (idx > 0) {
                const parent = (idx - 1) / 2;
                if (DeadlineItem.lessThan({}, self.items.items[idx], self.items.items[parent])) {
                    std.mem.swap(DeadlineItem, &self.items.items[idx], &self.items.items[parent]);
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        fn heapSiftDown(self: *Self, start_idx: usize) void {
            var idx = start_idx;
            const items_len = self.items.items.len;

            while (true) {
                var smallest = idx;
                const left = 2 * idx + 1;
                const right = 2 * idx + 2;

                if (left < items_len and
                    DeadlineItem.lessThan({}, self.items.items[left], self.items.items[smallest]))
                {
                    smallest = left;
                }

                if (right < items_len and
                    DeadlineItem.lessThan({}, self.items.items[right], self.items.items[smallest]))
                {
                    smallest = right;
                }

                if (smallest == idx) break;

                std.mem.swap(DeadlineItem, &self.items.items[idx], &self.items.items[smallest]);
                idx = smallest;
            }
        }
    };
}

test "priority queue basic" {
    const allocator = std.testing.allocator;
    var queue = PriorityQueue(u32).init(allocator, .{});
    defer queue.deinit();

    try queue.push(1, .low);
    try queue.push(2, .critical);
    try queue.push(3, .normal);

    // Should come out in priority order
    try std.testing.expectEqual(@as(u32, 2), queue.pop().?); // critical
    try std.testing.expectEqual(@as(u32, 3), queue.pop().?); // normal
    try std.testing.expectEqual(@as(u32, 1), queue.pop().?); // low
    try std.testing.expect(queue.pop() == null);
}

test "priority queue fair scheduling" {
    const allocator = std.testing.allocator;
    var queue = PriorityQueue(u32).init(allocator, .{ .fair_scheduling = true });
    defer queue.deinit();

    // Add items with same priority
    try queue.push(1, .normal);
    try queue.push(2, .normal);
    try queue.push(3, .normal);

    // Should come out in FIFO order
    try std.testing.expectEqual(@as(u32, 1), queue.pop().?);
    try std.testing.expectEqual(@as(u32, 2), queue.pop().?);
    try std.testing.expectEqual(@as(u32, 3), queue.pop().?);
}

test "priority queue stats" {
    const allocator = std.testing.allocator;
    var queue = PriorityQueue(u32).init(allocator, .{});
    defer queue.deinit();

    try queue.push(1, .critical);
    try queue.push(2, .high);
    try queue.push(3, .normal);
    try queue.push(4, .low);
    try queue.push(5, .background);

    const stats = queue.getStats();
    try std.testing.expectEqual(@as(usize, 5), stats.size);
    try std.testing.expectEqual(@as(usize, 1), stats.critical_count);
    try std.testing.expectEqual(@as(usize, 1), stats.high_count);
    try std.testing.expectEqual(@as(usize, 1), stats.normal_count);
    try std.testing.expectEqual(@as(usize, 1), stats.low_count);
    try std.testing.expectEqual(@as(usize, 1), stats.background_count);
}

test "deadline queue" {
    const allocator = std.testing.allocator;
    var queue = DeadlineQueue(u32).init(allocator);
    defer queue.deinit();

    try queue.push(1, 300); // Later deadline
    try queue.push(2, 100); // Earliest deadline
    try queue.push(3, 200); // Middle deadline

    // Should come out in deadline order
    try std.testing.expectEqual(@as(u32, 2), queue.pop().?);
    try std.testing.expectEqual(@as(u32, 3), queue.pop().?);
    try std.testing.expectEqual(@as(u32, 1), queue.pop().?);
}

test "priority weights" {
    try std.testing.expectEqual(@as(u32, 16), Priority.critical.toWeight());
    try std.testing.expectEqual(@as(u32, 8), Priority.high.toWeight());
    try std.testing.expectEqual(@as(u32, 4), Priority.normal.toWeight());
    try std.testing.expectEqual(@as(u32, 2), Priority.low.toWeight());
    try std.testing.expectEqual(@as(u32, 1), Priority.background.toWeight());
}
