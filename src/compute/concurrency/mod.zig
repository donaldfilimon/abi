//! Lightweight concurrency helpers for queues and backoff.
const std = @import("std");

pub const lockfree = @import("lockfree.zig");
pub const LockFreeQueue = lockfree.LockFreeQueue;
pub const LockFreeStack = lockfree.LockFreeStack;
pub const LockFreeMap = lockfree.LockFreeMap;

pub fn WorkQueue(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        items: std.ArrayList(T),

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{
                .allocator = allocator,
                .items = std.ArrayList(T).empty,
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
        _ = std.Thread.yield() catch {};
    }

    pub fn wait(self: *Backoff) void {
        self.spins += 1;
        const iterations = @min(self.spins, 64);
        var i: usize = 0;
        while (i < iterations) : (i += 1) {
            std.atomic.spinLoopHint();
        }
        if (self.spins > 32) {
            _ = std.Thread.yield() catch {};
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
