//! Synchronization primitives
//!
//! Provides cross-version compatible synchronization primitives for Zig 0.16.
//! This module works around the absence of std.Thread.Mutex in earlier 0.16 dev builds.

const std = @import("std");

/// Mutex is a synchronization primitive which enforces atomic access to a
/// shared region of code known as the "critical section".
///
/// This is a spinlock-based implementation that works across Zig 0.16 versions.
/// It blocks by busy-waiting rather than using futex/condvars, so use sparingly
/// for very short critical sections.
pub const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    /// Initialize a new unlocked mutex
    pub fn init() Mutex {
        return .{};
    }

    /// Acquires the mutex, blocking the caller's thread until it can.
    /// Once acquired, call `unlock()` to release it.
    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) {
            // Busy-wait spinloop - hint to CPU we're spinning
            std.atomic.spinLoopHint();
        }
    }

    /// Tries to acquire the mutex without blocking the caller's thread.
    /// Returns `true` if the lock was acquired, `false` otherwise.
    pub fn tryLock(self: *Mutex) bool {
        return !self.locked.swap(true, .acquire);
    }

    /// Releases the mutex which was previously acquired with `lock()` or `tryLock()`.
    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

test "mutex basic operations" {
    var mutex = Mutex.init();

    mutex.lock();
    try std.testing.expect(mutex.locked.load(.acquire));
    mutex.unlock();
    try std.testing.expect(!mutex.locked.load(.acquire));
}

test "mutex try lock" {
    var mutex = Mutex.init();

    try std.testing.expect(mutex.tryLock());
    try std.testing.expect(!mutex.tryLock());
    mutex.unlock();
    try std.testing.expect(mutex.tryLock());
    mutex.unlock();
}

/// Read-Write Lock allowing multiple readers or a single writer.
/// This is a simplified spinlock-based implementation.
pub const RwLock = struct {
    state: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),
    // state: 0 = unlocked, >0 = read locks, -1 = write lock

    pub fn init() RwLock {
        return .{};
    }

    pub fn lockShared(self: *RwLock) void {
        while (true) {
            const state = self.state.load(.acquire);
            if (state >= 0) {
                if (self.state.cmpxchgWeak(state, state + 1, .acquire, .monotonic) == null) {
                    return;
                }
            }
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlockShared(self: *RwLock) void {
        _ = self.state.fetchSub(1, .release);
    }

    pub fn lock(self: *RwLock) void {
        while (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(self: *RwLock) void {
        self.state.store(0, .release);
    }
};

/// Condition variable for thread coordination.
/// This is a minimal implementation using busy-waiting.
/// Note: This is NOT a proper condition variable and should be used sparingly.
pub const Condition = struct {
    pub fn init() Condition {
        return .{};
    }

    /// Signal is a no-op in this spinlock-based implementation.
    /// The waiting thread will detect changes through its predicate check.
    pub fn signal(self: *Condition) void {
        _ = self;
    }

    /// Broadcast is a no-op in this spinlock-based implementation.
    pub fn broadcast(self: *Condition) void {
        _ = self;
    }

    /// Wait is not properly implemented as it requires integration with mutex unlock/relock.
    /// Users should implement their own polling loop instead.
    pub fn wait(self: *Condition, mutex: *Mutex) void {
        _ = self;
        _ = mutex;
        // Cannot properly implement without OS-level futex/condvar support
        // Callers should use a polling pattern instead
    }
};

/// Wake event for signaling a sleeping thread.
/// This is a spinlock-based implementation that works without OS-level futex support.
/// Used to wake background monitor threads (e.g., auto-reindexer).
pub const Wake = struct {
    signaled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    /// Signal the wake event, causing any waiting thread to proceed.
    pub fn signal(self: *Wake) void {
        self.signaled.store(true, .release);
    }

    /// Wait until signaled or until the timeout (in nanoseconds) elapses.
    /// Returns `.timed_out` if the timeout expired, `.signaled` otherwise.
    pub fn timedWait(self: *Wake, timeout_ns: u64) TimedWaitResult {
        const time = @import("time.zig");
        const start = time.timestampNs();
        while (!self.signaled.load(.acquire)) {
            const elapsed: u64 = @intCast(time.timestampNs() - start);
            if (elapsed >= timeout_ns) return .timed_out;
            std.atomic.spinLoopHint();
        }
        self.signaled.store(false, .release);
        return .signaled;
    }

    /// No-op deinit for API compatibility.
    pub fn deinit(self: *Wake) void {
        _ = self;
    }

    pub const TimedWaitResult = enum {
        signaled,
        timed_out,
    };
};

test "rwlock basic" {
    var rwlock = RwLock.init();

    rwlock.lockShared();
    try std.testing.expectEqual(@as(i32, 1), rwlock.state.load(.acquire));
    rwlock.unlockShared();

    rwlock.lock();
    try std.testing.expectEqual(@as(i32, -1), rwlock.state.load(.acquire));
    rwlock.unlock();
}
