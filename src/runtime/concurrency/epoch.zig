//! Epoch-Based Reclamation (EBR) for Safe Memory Reclamation
//!
//! Provides safe memory reclamation for lock-free data structures by deferring
//! deallocation until no thread can hold a reference to the memory.
//!
//! ## How it Works
//!
//! 1. Threads "pin" themselves to an epoch when accessing shared data
//! 2. Memory is placed in a retirement list when "freed"
//! 3. Memory is only truly freed when all threads have advanced past the epoch
//!    where the memory was retired
//!
//! ## Usage
//!
//! ```zig
//! var ebr = EpochReclamation.init(allocator);
//! defer ebr.deinit();
//!
//! // Pin current thread before accessing shared data
//! ebr.pin();
//! defer ebr.unpin();
//!
//! // When done with a node, retire it instead of freeing
//! ebr.retire(node);
//! ```
//!
//! ## Thread Safety
//!
//! This implementation is designed for multi-threaded use. Each thread should
//! have its own thread-local state, accessed via the `getThreadState()` function.

const std = @import("std");

/// Maximum number of threads supported
const MAX_THREADS = 256;

/// Number of epochs to keep (3 is sufficient for EBR)
const EPOCH_COUNT = 3;

/// Items to accumulate before attempting to reclaim
const RECLAIM_THRESHOLD = 64;

/// Thread-local epoch state
const ThreadState = struct {
    /// Current epoch this thread is pinned to (or null if not pinned)
    pinned_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Whether this thread is currently pinned
    is_pinned: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// Per-thread retirement lists for each epoch
    retired_lists: [EPOCH_COUNT]RetiredList = .{RetiredList{}} ** EPOCH_COUNT,
    /// Count of retired items for this thread
    retired_count: usize = 0,
};

/// A retired item waiting for reclamation
const RetiredItem = struct {
    ptr: *anyopaque,
    deinit_fn: *const fn (*anyopaque, std.mem.Allocator) void,
    next: ?*RetiredItem,
};

/// Per-epoch retirement list
const RetiredList = struct {
    head: ?*RetiredItem = null,
    count: usize = 0,
};

/// Epoch-Based Reclamation manager
pub const EpochReclamation = struct {
    /// Global epoch counter
    global_epoch: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Per-thread states
    thread_states: [MAX_THREADS]ThreadState = .{ThreadState{}} ** MAX_THREADS,
    /// Allocator for internal bookkeeping
    allocator: std.mem.Allocator,
    /// Active thread count
    active_threads: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn init(allocator: std.mem.Allocator) EpochReclamation {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *EpochReclamation) void {
        // Free all retired items
        for (&self.thread_states) |*state| {
            for (&state.retired_lists) |*list| {
                var current = list.head;
                while (current) |item| {
                    const next = item.next;
                    item.deinit_fn(item.ptr, self.allocator);
                    self.allocator.destroy(item);
                    current = next;
                }
                list.head = null;
                list.count = 0;
            }
        }
        self.* = undefined;
    }

    /// Get the thread-local state index for the current thread.
    fn getThreadIndex() usize {
        const thread_id = std.Thread.getCurrentId();
        // Simple hash to map thread ID to slot - use bitcast for type conversion
        const id_bits: usize = @intCast(thread_id);
        return id_bits % MAX_THREADS;
    }

    /// Pin the current thread to the global epoch.
    /// Must be called before accessing shared data.
    pub fn pin(self: *EpochReclamation) void {
        const idx = getThreadIndex();
        var state = &self.thread_states[idx];

        // Already pinned? Just return
        if (state.is_pinned.load(.acquire)) return;

        // Read the global epoch and mark ourselves as active
        const epoch = self.global_epoch.load(.acquire);
        state.pinned_epoch.store(epoch, .release);
        state.is_pinned.store(true, .seq_cst); // Use seq_cst as fence
    }

    /// Unpin the current thread, allowing epoch advancement.
    pub fn unpin(self: *EpochReclamation) void {
        const idx = getThreadIndex();
        var state = &self.thread_states[idx];

        state.is_pinned.store(false, .release);

        // Try to advance the epoch and reclaim memory
        self.tryAdvance();
    }

    /// Retire a node for later reclamation.
    /// The node will be freed when it's safe to do so.
    pub fn retire(self: *EpochReclamation, ptr: anytype) void {
        self.retireWithFn(ptr, struct {
            fn deinit(p: *anyopaque, alloc: std.mem.Allocator) void {
                const typed_ptr: *@TypeOf(ptr.*) = @ptrCast(@alignCast(p));
                alloc.destroy(typed_ptr);
            }
        }.deinit);
    }

    /// Retire with a custom deinitialization function.
    pub fn retireWithFn(
        self: *EpochReclamation,
        ptr: anytype,
        deinit_fn: *const fn (*anyopaque, std.mem.Allocator) void,
    ) void {
        const idx = getThreadIndex();
        var state = &self.thread_states[idx];

        // Get the current epoch's retirement list
        const epoch = self.global_epoch.load(.acquire);
        const list_idx = epoch % EPOCH_COUNT;
        var list = &state.retired_lists[list_idx];

        // Create retirement record
        const item = self.allocator.create(RetiredItem) catch return;
        item.* = .{
            .ptr = @ptrCast(ptr),
            .deinit_fn = deinit_fn,
            .next = list.head,
        };
        list.head = item;
        list.count += 1;
        state.retired_count += 1;

        // Try to reclaim if we've accumulated enough
        if (state.retired_count >= RECLAIM_THRESHOLD) {
            self.tryAdvance();
        }
    }

    /// Try to advance the global epoch.
    fn tryAdvance(self: *EpochReclamation) void {
        const current_epoch = self.global_epoch.load(.acquire);

        // Check if all pinned threads are at the current epoch
        for (&self.thread_states) |*state| {
            if (state.is_pinned.load(.acquire)) {
                const thread_epoch = state.pinned_epoch.load(.acquire);
                if (thread_epoch < current_epoch) {
                    // A thread is still in an older epoch, can't advance
                    return;
                }
            }
        }

        // Try to advance the epoch
        _ = self.global_epoch.cmpxchgWeak(
            current_epoch,
            current_epoch + 1,
            .acq_rel,
            .acquire,
        ) orelse {
            // Successfully advanced, now we can reclaim the oldest epoch
            self.reclaimEpoch((current_epoch + 1) % EPOCH_COUNT);
        };
    }

    /// Reclaim all items in the given epoch slot.
    fn reclaimEpoch(self: *EpochReclamation, epoch_slot: usize) void {
        for (&self.thread_states) |*state| {
            var list = &state.retired_lists[epoch_slot];
            var current = list.head;

            while (current) |item| {
                const next = item.next;
                item.deinit_fn(item.ptr, self.allocator);
                self.allocator.destroy(item);
                if (state.retired_count > 0) {
                    state.retired_count -= 1;
                }
                current = next;
            }

            list.head = null;
            list.count = 0;
        }
    }

    /// Get the current global epoch.
    pub fn currentEpoch(self: *const EpochReclamation) u64 {
        return self.global_epoch.load(.acquire);
    }
};

/// A lock-free stack using epoch-based reclamation (ABA-safe).
pub fn LockFreeStackEBR(comptime T: type) type {
    return struct {
        const Node = struct {
            value: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        head: std.atomic.Value(?*Node) = std.atomic.Value(?*Node).init(null),
        ebr: *EpochReclamation,

        pub fn init(allocator: std.mem.Allocator, ebr: *EpochReclamation) @This() {
            return .{
                .allocator = allocator,
                .ebr = ebr,
            };
        }

        pub fn deinit(self: *@This()) void {
            // Drain the stack
            while (self.popUnsafe()) |_| {}
            self.* = undefined;
        }

        /// Push a value onto the stack.
        pub fn push(self: *@This(), value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .value = value, .next = null };

            self.ebr.pin();
            defer self.ebr.unpin();

            while (true) {
                const current = self.head.load(.acquire);
                node.next = current;
                if (self.head.cmpxchgWeak(current, node, .acq_rel, .acquire) == null) {
                    break;
                }
            }
        }

        /// Pop a value from the stack (ABA-safe).
        pub fn pop(self: *@This()) ?T {
            self.ebr.pin();
            defer self.ebr.unpin();

            while (true) {
                const current = self.head.load(.acquire);
                if (current == null) return null;

                const next = current.?.next;
                if (self.head.cmpxchgWeak(current, next, .acq_rel, .acquire) == null) {
                    const value = current.?.value;
                    // Retire the node instead of immediate free
                    self.ebr.retire(current.?);
                    return value;
                }
            }
        }

        /// Pop without epoch protection (for cleanup only).
        fn popUnsafe(self: *@This()) ?T {
            while (true) {
                const current = self.head.load(.acquire);
                if (current == null) return null;

                const next = current.?.next;
                if (self.head.cmpxchgWeak(current, next, .acq_rel, .acquire) == null) {
                    const value = current.?.value;
                    self.allocator.destroy(current.?);
                    return value;
                }
            }
        }

        /// Check if the stack is empty.
        pub fn isEmpty(self: *const @This()) bool {
            return self.head.load(.acquire) == null;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "epoch reclamation basic" {
    var ebr = EpochReclamation.init(std.testing.allocator);
    defer ebr.deinit();

    // Test pin/unpin
    ebr.pin();
    const epoch1 = ebr.currentEpoch();
    ebr.unpin();

    // Epoch might advance after unpin
    try std.testing.expect(ebr.currentEpoch() >= epoch1);
}

test "lock-free stack EBR is LIFO" {
    var ebr = EpochReclamation.init(std.testing.allocator);
    defer ebr.deinit();

    var stack = LockFreeStackEBR(u32).init(std.testing.allocator, &ebr);
    defer stack.deinit();

    try stack.push(10);
    try stack.push(20);
    try stack.push(30);

    try std.testing.expectEqual(@as(?u32, 30), stack.pop());
    try std.testing.expectEqual(@as(?u32, 20), stack.pop());
    try std.testing.expectEqual(@as(?u32, 10), stack.pop());
    try std.testing.expectEqual(@as(?u32, null), stack.pop());
}

test "lock-free stack EBR concurrent" {
    var ebr = EpochReclamation.init(std.testing.allocator);
    defer ebr.deinit();

    var stack = LockFreeStackEBR(u32).init(std.testing.allocator, &ebr);
    defer stack.deinit();

    const thread_count = 4;
    const ops_per_thread = 50;

    var threads: [thread_count]std.Thread = undefined;
    var push_counts: [thread_count]std.atomic.Value(usize) = undefined;
    var pop_counts: [thread_count]std.atomic.Value(usize) = undefined;

    for (0..thread_count) |i| {
        push_counts[i] = std.atomic.Value(usize).init(0);
        pop_counts[i] = std.atomic.Value(usize).init(0);
    }

    for (&threads, 0..) |*t, tid| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn worker(
                s: *LockFreeStackEBR(u32),
                push_cnt: *std.atomic.Value(usize),
                pop_cnt: *std.atomic.Value(usize),
            ) !void {
                var pushed: usize = 0;
                var popped: usize = 0;

                for (0..ops_per_thread) |i| {
                    if (i % 2 == 0) {
                        try s.push(@intCast(i));
                        pushed += 1;
                    } else {
                        if (s.pop()) |_| {
                            popped += 1;
                        }
                    }
                }

                push_cnt.store(pushed, .release);
                pop_cnt.store(popped, .release);
            }
        }.worker, .{ &stack, &push_counts[tid], &pop_counts[tid] });
    }

    for (&threads) |*t| {
        t.join();
    }

    // Drain remaining items
    var remaining: usize = 0;
    while (stack.pop()) |_| {
        remaining += 1;
    }

    var total_pushed: usize = 0;
    var total_popped: usize = 0;
    for (0..thread_count) |i| {
        total_pushed += push_counts[i].load(.acquire);
        total_popped += pop_counts[i].load(.acquire);
    }

    // Total pushed should equal total popped + remaining
    try std.testing.expectEqual(total_pushed, total_popped + remaining);
}
