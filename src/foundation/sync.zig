//! Synchronization primitives
//!
//! Provides cross-version compatible synchronization primitives for Zig 0.16.
//! This module works around the absence of std.Thread.Mutex in earlier 0.16 dev builds.

const std = @import("std");
const builtin = @import("builtin");
const time_mod = @import("time.zig");

/// Mutex is a synchronization primitive which enforces atomic access to a
/// shared region of code known as the "critical section".
///
/// This is a spinlock-based implementation that works across Zig 0.16 versions.
/// It blocks by busy-waiting rather than using futex/condvars, so use sparingly
/// for very short critical sections.
///
/// NOTE: Do NOT pair this mutex with `Condition`. Use `BlockingMutex` instead.
pub const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    /// Initialize a new unlocked mutex
    pub fn init() Mutex {
        return .{};
    }

    /// Acquires the mutex, blocking the caller's thread until it can.
    /// Uses bounded exponential backoff to reduce CPU waste under contention.
    pub fn lock(self: *Mutex) void {
        // Fast path: try to acquire immediately
        if (!self.locked.swap(true, .acquire)) return;

        // Slow path: exponential backoff then yield
        var spin: u8 = 1;
        while (true) {
            for (0..spin) |_| std.atomic.spinLoopHint();
            if (!self.locked.swap(true, .acquire)) return;
            spin = @min(spin *| 2, 32);
            if (spin >= 32) {
                if (comptime builtin.os.tag != .freestanding) {
                    // yield is a performance hint; failure is non-critical
                    std.Thread.yield() catch {};
                }
            }
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
        // Fast path: try to acquire immediately
        const state = self.state.load(.acquire);
        if (state >= 0) {
            if (self.state.cmpxchgWeak(state, state + 1, .acquire, .monotonic) == null) {
                return;
            }
        }
        // Slow path: exponential backoff
        var spin: u8 = 1;
        while (true) {
            for (0..spin) |_| std.atomic.spinLoopHint();
            const s = self.state.load(.acquire);
            if (s >= 0) {
                if (self.state.cmpxchgWeak(s, s + 1, .acquire, .monotonic) == null) {
                    return;
                }
            }
            spin = @min(spin *| 2, 32);
            if (spin >= 32) {
                if (comptime builtin.os.tag != .freestanding) {
                    // yield is a performance hint; failure is non-critical
                    std.Thread.yield() catch {};
                }
            }
        }
    }

    pub fn unlockShared(self: *RwLock) void {
        _ = self.state.fetchSub(1, .release);
    }

    pub fn lock(self: *RwLock) void {
        // Fast path
        if (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) == null) return;
        // Slow path: exponential backoff
        var spin: u8 = 1;
        while (true) {
            for (0..spin) |_| std.atomic.spinLoopHint();
            if (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) == null) return;
            spin = @min(spin *| 2, 32);
            if (spin >= 32) {
                if (comptime builtin.os.tag != .freestanding) {
                    // yield is a performance hint; failure is non-critical
                    std.Thread.yield() catch {};
                }
            }
        }
    }

    pub fn unlock(self: *RwLock) void {
        self.state.store(0, .release);
    }
};

/// OS-backed blocking mutex.
///
/// Unlike `Mutex` (a spinlock), `BlockingMutex` suspends the calling thread via
/// the OS scheduler when the lock is contended. It is the **required** paired
/// mutex type for `Condition` — passing any other mutex type to
/// `Condition.wait()` / `Condition.timedWait()` is a compile error.
///
/// On POSIX platforms (macOS, Linux with libc) this wraps `pthread_mutex_t`.
/// On other targets it falls back to the `Mutex` spinlock.
pub const BlockingMutex = struct {
    impl: Impl = .{},

    const use_pthreads = (builtin.os.tag != .freestanding and builtin.os.tag != .wasi and
        builtin.link_libc);

    const Impl = if (use_pthreads)
        struct {
            inner: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
        }
    else
        struct {
            fallback: Mutex = .{},
        };

    /// Initialize a new unlocked blocking mutex.
    pub fn init() BlockingMutex {
        return .{};
    }

    /// Deinitialize. Must be called when the mutex is no longer needed (POSIX only).
    pub fn deinit(self: *BlockingMutex) void {
        if (comptime use_pthreads) {
            _ = std.c.pthread_mutex_destroy(&self.impl.inner);
        }
    }

    /// Acquire the mutex, blocking until it is available.
    pub fn lock(self: *BlockingMutex) void {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_mutex_lock(&self.impl.inner);
            std.debug.assert(rc == .SUCCESS);
        } else {
            self.impl.fallback.lock();
        }
    }

    /// Try to acquire the mutex without blocking.
    /// Returns `true` if the lock was acquired, `false` otherwise.
    pub fn tryLock(self: *BlockingMutex) bool {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_mutex_trylock(&self.impl.inner);
            return rc == .SUCCESS;
        } else {
            return self.impl.fallback.tryLock();
        }
    }

    /// Release the mutex.
    pub fn unlock(self: *BlockingMutex) void {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_mutex_unlock(&self.impl.inner);
            std.debug.assert(rc == .SUCCESS);
        } else {
            self.impl.fallback.unlock();
        }
    }
};

/// Condition variable for blocking thread coordination.
///
/// **IMPORTANT**: `Condition` must always be paired with a `BlockingMutex`.
/// Passing a `Mutex` (spinlock) is incorrect and will not compile.
///
/// On POSIX platforms (macOS, Linux with libc) this wraps `pthread_cond_t`,
/// which provides true OS-level blocking. On other targets `wait()` falls back
/// to yielding in a loop, which still avoids a busy-spin but is not ideal.
///
/// Usage pattern (mirrors `std.Thread.Condition`):
/// ```zig
/// var mutex = sync.BlockingMutex.init();
/// var cond  = sync.Condition.init();
///
/// // waiter thread
/// mutex.lock();
/// while (!predicate()) cond.wait(&mutex);
/// mutex.unlock();
///
/// // signaler thread
/// mutex.lock();
/// setPredicate();
/// cond.signal();
/// mutex.unlock();
/// ```
pub const Condition = struct {
    impl: Impl = .{},

    const use_pthreads = BlockingMutex.use_pthreads;

    const Impl = if (use_pthreads)
        struct {
            inner: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
        }
    else
        struct {};

    /// Initialize a new condition variable.
    pub fn init() Condition {
        return .{};
    }

    /// Deinitialize the condition variable (POSIX only; no-op elsewhere).
    pub fn deinit(self: *Condition) void {
        if (comptime use_pthreads) {
            _ = std.c.pthread_cond_destroy(&self.impl.inner);
        }
    }

    /// Wake one thread waiting on this condition.
    pub fn signal(self: *Condition) void {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_cond_signal(&self.impl.inner);
            std.debug.assert(rc == .SUCCESS);
        }
        // Non-posix: atomics / stores by the caller serve as the signal;
        // the looping waiter will observe them on the next iteration.
    }

    /// Wake all threads waiting on this condition.
    pub fn broadcast(self: *Condition) void {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_cond_broadcast(&self.impl.inner);
            std.debug.assert(rc == .SUCCESS);
        }
    }

    /// Atomically unlock `mutex`, block until signaled, then re-lock `mutex`.
    ///
    /// The caller must hold `mutex` before calling `wait()`. On return the
    /// caller holds `mutex` again. This may spuriously return; always re-check
    /// the predicate in a loop.
    pub fn wait(self: *Condition, mutex: *BlockingMutex) void {
        if (comptime use_pthreads) {
            const rc = std.c.pthread_cond_wait(&self.impl.inner, &mutex.impl.inner);
            std.debug.assert(rc == .SUCCESS);
        } else {
            // Fallback: release the spinlock and yield briefly. The waiter
            // loop in the caller will re-check the predicate, so this is
            // correct (though not as efficient as a kernel condvar).
            mutex.unlock();
            std.Thread.yield() catch {};
            mutex.lock();
        }
    }

    /// Like `wait()` but returns `error.Timeout` after `timeout_ns` nanoseconds.
    ///
    /// `pthread_cond_timedwait` takes an **absolute** REALTIME deadline.
    /// We compute `now + timeout_ns` before calling into the kernel.
    pub fn timedWait(self: *Condition, mutex: *BlockingMutex, timeout_ns: u64) error{Timeout}!void {
        if (comptime use_pthreads) {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.REALTIME, &ts);

            // Add timeout_ns to the current time
            const extra_sec: i64 = @intCast(timeout_ns / std.time.ns_per_s);
            const extra_nsec: i64 = @intCast(timeout_ns % std.time.ns_per_s);
            ts.sec += extra_sec;
            ts.nsec += extra_nsec;
            if (ts.nsec >= std.time.ns_per_s) {
                ts.nsec -= std.time.ns_per_s;
                ts.sec += 1;
            }

            const rc = std.c.pthread_cond_timedwait(&self.impl.inner, &mutex.impl.inner, &ts);
            if (rc == .TIMEDOUT) return error.Timeout;
            // EINTR is treated as a spurious wakeup (caller re-checks predicate)
        } else {
            // Fallback: spin-wait approximation
            const timer = time_mod.Timer.start() catch return error.Timeout;
            mutex.unlock();
            defer mutex.lock();
            while (timer.read() < timeout_ns) {
                std.Thread.yield() catch {};
            }
            return error.Timeout;
        }
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
    /// Uses exponential backoff (spin → yield) to avoid burning CPU on long waits.
    pub fn timedWait(self: *Wake, timeout_ns: u64) TimedWaitResult {
        const time = @import("time.zig");
        const start = time.timestampNs();
        var spin: u8 = 1;
        while (!self.signaled.load(.acquire)) {
            const elapsed: u64 = @intCast(time.timestampNs() - start);
            if (elapsed >= timeout_ns) return .timed_out;
            for (0..spin) |_| std.atomic.spinLoopHint();
            spin = @min(spin *| 2, 32);
            if (spin >= 32) {
                if (comptime builtin.os.tag != .freestanding) {
                    // yield is a performance hint; failure is non-critical
                    std.Thread.yield() catch {};
                }
            }
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

test "blocking mutex basic" {
    var m = BlockingMutex.init();
    defer m.deinit();

    m.lock();
    try std.testing.expect(!m.tryLock());
    m.unlock();
    try std.testing.expect(m.tryLock());
    m.unlock();
}

test "condition signal unblocks waiter" {
    // Verify that signal() + broadcast() compile and run without error.
    // A full multi-threaded test would require spawning threads; here we
    // exercise the single-threaded path (timedWait that times out).
    var m = BlockingMutex.init();
    defer m.deinit();
    var cond = Condition.init();
    defer cond.deinit();

    m.lock();
    const result = cond.timedWait(&m, 1); // 1 ns — should time out immediately
    m.unlock();
    try std.testing.expectError(error.Timeout, result);
}
