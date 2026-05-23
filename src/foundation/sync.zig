const std = @import("std");
const testing = std.testing;
const AtomicValue = std.atomic.Value;

pub const SpinLock = struct {
    state: AtomicValue(bool) = AtomicValue(bool).init(false),

    pub fn lock(self: *SpinLock) void {
        while (self.state.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(self: *SpinLock) void {
        self.state.store(false, .release);
    }

    pub fn tryLock(self: *SpinLock) bool {
        return self.state.cmpxchgWeak(false, true, .acquire, .monotonic) == null;
    }
};

pub const RwLock = struct {
    // State layout:
    // [1 bit: writer] [31 bits: readers]
    state: AtomicValue(u32) = AtomicValue(u32).init(0),

    const WRITER_BIT = 1 << 31;
    const READER_MASK = ~@as(u32, WRITER_BIT);

    pub fn lockRead(self: *RwLock) void {
        while (true) {
            const s = self.state.load(.acquire);
            if (s & WRITER_BIT != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (self.state.cmpxchgWeak(s, s + 1, .acquire, .monotonic) == null) {
                break;
            }
        }
    }

    pub fn unlockRead(self: *RwLock) void {
        _ = self.state.fetchSub(1, .release);
    }

    pub fn tryLockRead(self: *RwLock) bool {
        const s = self.state.load(.acquire);
        if (s & WRITER_BIT != 0) return false;
        return self.state.cmpxchgWeak(s, s + 1, .acquire, .monotonic) == null;
    }

    pub fn lockWrite(self: *RwLock) void {
        while (true) {
            if (self.state.cmpxchgWeak(0, WRITER_BIT, .acquire, .monotonic) == null) {
                break;
            }
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlockWrite(self: *RwLock) void {
        self.state.store(0, .release);
    }

    pub fn tryLockWrite(self: *RwLock) bool {
        return self.state.cmpxchgWeak(0, WRITER_BIT, .acquire, .monotonic) == null;
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "RwLock basic usage" {
    var lock = RwLock{};
    lock.lockRead();
    try testing.expect(lock.tryLockWrite() == false);
    lock.unlockRead();
    lock.lockWrite();
    try testing.expect(lock.tryLockRead() == false);
    lock.unlockWrite();
}

test "RwLock concurrent readers" {
    const reader_count = 4;
    var lock = RwLock{};
    var reader_ready = std.atomic.Value(usize).init(0);
    var can_exit = std.atomic.Value(bool).init(false);

    const Reader = struct {
        l: *RwLock,
        ready: *std.atomic.Value(usize),
        exit: *std.atomic.Value(bool),
        fn run(self: @This()) void {
            self.l.lockRead();
            defer self.l.unlockRead();
            _ = self.ready.fetchAdd(1, .release);
            while (!self.exit.load(.acquire)) {
                std.atomic.spinLoopHint();
            }
        }
    };

    var threads: [reader_count]std.Thread = undefined;
    for (0..reader_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, Reader.run, .{Reader{ .l = &lock, .ready = &reader_ready, .exit = &can_exit }});
    }

    while (reader_ready.load(.acquire) < reader_count) {
        std.atomic.spinLoopHint();
    }

    try testing.expect(lock.tryLockWrite() == false);
    can_exit.store(true, .release);

    for (threads) |t| {
        t.join();
    }
}

test "SpinLock basic usage" {
    var lock_obj = SpinLock{};
    lock_obj.lock();
    try testing.expect(lock_obj.tryLock() == false);
    lock_obj.unlock();
    try testing.expect(lock_obj.tryLock() == true);
    lock_obj.unlock();
}

test "SpinLock mutual exclusion" {
    const thread_count = 4;
    const iterations = 1000;

    var lock_obj = SpinLock{};
    var shared_counter: usize = 0;

    const Worker = struct {
        l: *SpinLock,
        c: *usize,
        fn run(self: @This()) void {
            for (0..iterations) |_| {
                self.l.lock();
                self.c.* += 1;
                self.l.unlock();
            }
        }
    };

    var threads: [thread_count]std.Thread = undefined;
    for (0..thread_count) |i| {
        threads[i] = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .l = &lock_obj, .c = &shared_counter }});
    }

    for (threads) |t| {
        t.join();
    }

    try testing.expectEqual(@as(usize, thread_count * iterations), shared_counter);
}
