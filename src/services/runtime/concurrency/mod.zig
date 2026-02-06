//! Concurrency Primitives for Runtime
//!
//! This module provides lock-free and thread-safe data structures
//! for concurrent task execution.
//!
//! ## Available Types
//!
//! - `WorkStealingQueue` - Owner pushes/pops from back, thieves steal from front
//! - `WorkQueue` - Simple FIFO queue with mutex
//! - `LockFreeQueue` - CAS-based queue (SPSC)
//! - `LockFreeStack` - CAS-based stack (WARNING: ABA susceptible)
//! - `LockFreeStackEBR` - ABA-safe stack with epoch-based reclamation
//! - `ChaseLevDeque` - Lock-free work-stealing deque (Chase-Lev algorithm)
//! - `MpmcQueue` - Lock-free multi-producer multi-consumer queue
//! - `ShardedMap` - Partitioned map reducing contention
//! - `PriorityQueue` - Lock-free priority queue for task scheduling
//! - `Backoff` - Exponential backoff for spin-wait loops
//! - `EpochReclamation` - Safe memory reclamation for lock-free structures

const std = @import("std");

// Local imports (implementation files)
pub const lockfree = @import("lockfree.zig");
pub const priority_queue = @import("priority_queue.zig");
pub const epoch = @import("epoch.zig");
pub const chase_lev = @import("chase_lev.zig");
pub const mpmc = @import("mpmc_queue.zig");

// Zig 0.16 compatibility: Simple spinlock Mutex
const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

// Lock-free structures
pub const LockFreeQueue = lockfree.LockFreeQueue;
pub const LockFreeStack = lockfree.LockFreeStack;
pub const ShardedMap = lockfree.ShardedMap;

// Epoch-based reclamation (ABA-safe)
pub const EpochReclamation = epoch.EpochReclamation;
pub const LockFreeStackEBR = epoch.LockFreeStackEBR;

// Chase-Lev work-stealing deque
pub const ChaseLevDeque = chase_lev.ChaseLevDeque;
pub const WorkStealingScheduler = chase_lev.WorkStealingScheduler;

// MPMC queue
pub const MpmcQueue = mpmc.MpmcQueue;
pub const BlockingMpmcQueue = mpmc.BlockingMpmcQueue;

// Priority queue
pub const Priority = priority_queue.Priority;
pub const PriorityQueue = priority_queue.PriorityQueue;
pub const PriorityQueueConfig = priority_queue.PriorityQueueConfig;
pub const PrioritizedItem = priority_queue.PrioritizedItem;
pub const QueueStats = priority_queue.QueueStats;
pub const MultilevelQueue = priority_queue.MultilevelQueue;
pub const DeadlineQueue = priority_queue.DeadlineQueue;

// ============================================================================
// Work Queues (inline implementation)
// ============================================================================

pub fn WorkQueue(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        mutex: Mutex = .{},
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
        mutex: Mutex = .{},
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

// ============================================================================
// Backoff Utility
// ============================================================================

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
            std.Thread.yield() catch |err| {
                std.log.debug("Thread yield failed during backoff wait: {t}", .{err});
            };
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "WorkStealingQueue basic operations" {
    var queue = WorkStealingQueue(u32).init(std.testing.allocator);
    defer queue.deinit();

    try queue.push(1);
    try queue.push(2);

    // Pop returns from back (LIFO for owner)
    try std.testing.expectEqual(@as(?u32, 2), queue.pop());

    // Steal returns from front (FIFO for thieves)
    try queue.push(3);
    try std.testing.expectEqual(@as(?u32, 1), queue.steal());
}

test "Backoff spin loop hint" {
    var backoff = Backoff{};
    backoff.spin();
    backoff.spin();
    try std.testing.expect(backoff.spins == 2);
    backoff.reset();
    try std.testing.expect(backoff.spins == 0);
}

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
