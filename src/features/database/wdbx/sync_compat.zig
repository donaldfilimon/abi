//! Local synchronization compatibility primitives for WDBX tests.
//!
//! Mirrors the `RwLock` behavior from `src/services/shared/sync.zig`, but stays
//! within this module path so direct `zig test src/features/database/wdbx/*.zig`
//! commands can compile without cross-module-path imports.

const std = @import("std");

pub const RwLock = struct {
    state: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),
    // state: 0 = unlocked, >0 = read locks, -1 = write lock

    pub fn init() RwLock {
        return .{};
    }

    pub fn lockShared(self: *RwLock) void {
        const state = self.state.load(.acquire);
        if (state >= 0) {
            if (self.state.cmpxchgWeak(state, state + 1, .acquire, .monotonic) == null) {
                return;
            }
        }

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
            if (spin >= 32) std.Thread.yield() catch {};
        }
    }

    pub fn unlockShared(self: *RwLock) void {
        _ = self.state.fetchSub(1, .release);
    }

    pub fn lock(self: *RwLock) void {
        if (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) == null) return;

        var spin: u8 = 1;
        while (true) {
            for (0..spin) |_| std.atomic.spinLoopHint();
            if (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) == null) return;
            spin = @min(spin *| 2, 32);
            if (spin >= 32) std.Thread.yield() catch {};
        }
    }

    pub fn unlock(self: *RwLock) void {
        self.state.store(0, .release);
    }
};

test "rwlock basic behavior" {
    var rwlock = RwLock.init();

    rwlock.lockShared();
    try std.testing.expectEqual(@as(i32, 1), rwlock.state.load(.acquire));
    rwlock.unlockShared();

    rwlock.lock();
    try std.testing.expectEqual(@as(i32, -1), rwlock.state.load(.acquire));
    rwlock.unlock();
}
