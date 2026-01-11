//! Lightweight concurrency helpers for queues and backoff.
const std = @import("std");

pub const lockfree = @import("lockfree.zig");
pub const LockFreeQueue = lockfree.LockFreeQueue;
pub const LockFreeStack = lockfree.LockFreeStack;
pub const ShardedMap = lockfree.ShardedMap;

pub const priority_queue = @import("priority_queue.zig");
pub const Priority = priority_queue.Priority;
pub const PriorityQueue = priority_queue.PriorityQueue;
pub const PriorityQueueConfig = priority_queue.PriorityQueueConfig;
pub const PrioritizedItem = priority_queue.PrioritizedItem;
pub const QueueStats = priority_queue.QueueStats;
pub const MultilevelQueue = priority_queue.MultilevelQueue;
pub const DeadlineQueue = priority_queue.DeadlineQueue;

pub fn WorkQueue(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        items: std.ArrayListUnmanaged(T),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .allocator = allocator,
                .items = std.ArrayListUnmanaged(T).empty,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn len(self: *@This()) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.items.items.len;
        }

        pub fn isEmpty(self: *@This()) bool {
            return self.len() == 0;
        }

        pub fn enqueue(self: *@This(), item: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.items.append(self.allocator, item);
        }

        pub fn dequeue(self: *@This()) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.items.items.len == 0) return null;
            return self.items.orderedRemove(0);
        }
    };
}

pub fn WorkStealingQueue(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        items: std.ArrayListUnmanaged(T),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .allocator = allocator,
                .items = std.ArrayListUnmanaged(T).empty,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.items.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn len(self: *@This()) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.items.items.len;
        }

        pub fn isEmpty(self: *@This()) bool {
            return self.len() == 0;
        }

        pub fn push(self: *@This(), item: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.items.append(self.allocator, item);
        }

        pub fn pop(self: *@This()) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.items.items.len == 0) {
                return null;
            }
            return self.items.pop();
        }

        pub fn steal(self: *@This()) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.items.items.len == 0) {
                return null;
            }
            return self.items.orderedRemove(0);
        }
    };
}

pub const Backoff = struct {
    spins: usize = 0,

    pub fn reset(self: *Backoff) void {
        self.spins = 0;
    }

    pub fn spin(self: *Backoff) void {
        self.spins += 1;
        if (self.spins <= 16) {
            std.atomic.spinLoopHint();
            return;
        }
        // Thread yield failure is non-critical; log at debug level and continue
        std.Thread.yield() catch |err| {
            std.log.debug("Thread yield failed during backoff spin: {t}", .{err});
        };
    }

    pub fn wait(self: *Backoff) void {
        self.spins += 1;
        const iterations = @min(self.spins, 64);
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            std.atomic.spinLoopHint();
        }
        if (self.spins > 32) {
            // Thread yield failure is non-critical; log at debug level and continue
            std.Thread.yield() catch |err| {
                std.log.debug("Thread yield failed during backoff wait: {t}", .{err});
            };
        }
    }
};

test "work queue is FIFO" {
    var queue = WorkQueue(u32).init(std.testing.allocator);
    defer queue.deinit();

    try queue.enqueue(1);
    try queue.enqueue(2);
    try queue.enqueue(3);

    try std.testing.expectEqual(@as(?u32, 1), queue.dequeue());
    try std.testing.expectEqual(@as(?u32, 2), queue.dequeue());
    try std.testing.expectEqual(@as(?u32, 3), queue.dequeue());
    try std.testing.expectEqual(@as(?u32, null), queue.dequeue());
}

test "work queue reports empty and length" {
    var queue = WorkQueue(u8).init(std.testing.allocator);
    defer queue.deinit();

    try std.testing.expect(queue.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), queue.len());

    try queue.enqueue(9);
    try std.testing.expect(!queue.isEmpty());
    try std.testing.expectEqual(@as(usize, 1), queue.len());
}

test "work stealing queue pops and steals" {
    var queue = WorkStealingQueue(u32).init(std.testing.allocator);
    defer queue.deinit();

    try queue.push(1);
    try queue.push(2);
    try std.testing.expectEqual(@as(?u32, 2), queue.pop());

    try queue.push(3);
    try std.testing.expectEqual(@as(?u32, 1), queue.steal());
    try std.testing.expectEqual(@as(?u32, 3), queue.pop());
    try std.testing.expectEqual(@as(?u32, null), queue.pop());
}
