// ============================================================================
// ABI Framework — Work-Stealing Thread Pool
// Adapted from abi-system-v2.0/thread_pool.zig
// ============================================================================
//
// Lightweight task scheduler with per-thread work queues and cooperative
// work-stealing for load balance. Self-contained — no external utility deps.
//
// Design:
//   - Fixed thread count (defaults to CPU core count)
//   - Each worker owns a local deque; idle workers steal from neighbors
//   - Tasks are closures captured into a fixed-size erased frame
//   - Graceful shutdown via atomic flag
//
// Performance: <1us task dispatch, <500ns steal latency
// ============================================================================

const std = @import("std");

// ─── SpinLock (inline) ──────────────────────────────────────────────────────

const SpinLock = struct {
    state: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    fn acquire(self: *SpinLock) void {
        var backoff: u32 = 1;
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            var i: u32 = 0;
            while (i < backoff) : (i += 1) std.atomic.spinLoopHint();
            backoff = @min(backoff * 2, 1024);
        }
    }

    fn release(self: *SpinLock) void {
        self.state.store(0, .release);
    }

    fn guard(self: *SpinLock) Guard {
        self.acquire();
        return .{ .lock = self };
    }

    const Guard = struct {
        lock: *SpinLock,
        fn deinit(self: Guard) void {
            self.lock.release();
        }
    };
};

// ─── Task Representation ─────────────────────────────────────────────────────

/// Type-erased task. Captures a function pointer and its arguments into
/// a fixed-size frame to avoid heap allocation per task.
pub const Task = struct {
    const max_capture_size = 128;

    frame: [max_capture_size]u8 = undefined,
    frame_len: u8 = 0,
    exec_fn: *const fn (*[max_capture_size]u8) void = undefined,

    /// Create a task from a function and its arguments
    pub fn init(comptime func: anytype, args: anytype) Task {
        const Args = @TypeOf(args);
        comptime {
            if (@sizeOf(Args) > max_capture_size) {
                @compileError("Task arguments exceed max capture size (" ++
                    std.fmt.comptimePrint("{d}", .{max_capture_size}) ++ " bytes)");
            }
        }

        var task = Task{};
        task.frame_len = @intCast(@sizeOf(Args));

        const dest: *Args = @ptrCast(@alignCast(&task.frame));
        dest.* = args;

        task.exec_fn = struct {
            fn exec(frame_ptr: *[max_capture_size]u8) void {
                const captured: *const Args = @ptrCast(@alignCast(frame_ptr));
                @call(.auto, func, captured.*);
            }
        }.exec;

        return task;
    }

    pub fn execute(self: *Task) void {
        self.exec_fn(&self.frame);
    }
};

// ─── Work Queue (Per-Worker) ─────────────────────────────────────────────────

fn WorkQueue(comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]Task = undefined,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,
        lock: SpinLock = .{},

        /// Owner pushes a task (LIFO for cache locality)
        pub fn push(self: *Self, task: Task) bool {
            const g = self.lock.guard();
            defer g.deinit();

            if (self.count >= capacity) return false;
            self.buffer[self.head] = task;
            self.head = (self.head + 1) % capacity;
            self.count += 1;
            return true;
        }

        /// Owner pops from head (LIFO — most recently pushed)
        pub fn pop(self: *Self) ?Task {
            const g = self.lock.guard();
            defer g.deinit();

            if (self.count == 0) return null;
            self.head = (self.head + capacity - 1) % capacity;
            self.count -= 1;
            return self.buffer[self.head];
        }

        /// Stealer takes from tail (FIFO — oldest task first)
        pub fn steal(self: *Self) ?Task {
            const g = self.lock.guard();
            defer g.deinit();

            if (self.count == 0) return null;
            const task = self.buffer[self.tail];
            self.tail = (self.tail + 1) % capacity;
            self.count -= 1;
            return task;
        }

        pub fn len(self: *const Self) usize {
            const g = @constCast(&self.lock).guard();
            defer g.deinit();
            return self.count;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len() == 0;
        }
    };
}

// ─── Thread Pool ─────────────────────────────────────────────────────────────

pub const ThreadPool = struct {
    const Self = @This();
    const queue_capacity = 4096;
    const max_workers = 64;

    pub const Config = struct {
        thread_count: u32 = 0,
        /// Reserved for API compatibility; current implementation uses
        /// fixed internal queue capacity.
        queue_size: u32 = queue_capacity,
        /// Reserved for future thread naming support.
        name: []const u8 = "abi-worker",
    };

    allocator: std.mem.Allocator,
    workers: []Worker,
    thread_count: u32,
    shutdown: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    tasks_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tasks_submitted: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    const Worker = struct {
        thread: ?std.Thread = null,
        queue: WorkQueue(queue_capacity) = .{},
        pool: *Self = undefined,
        id: u32 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !*Self {
        const cpu_count = std.Thread.getCpuCount() catch 4;
        const count: u32 = if (config.thread_count > 0)
            @min(config.thread_count, max_workers)
        else
            @intCast(@min(cpu_count, max_workers));

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .workers = try allocator.alloc(Worker, count),
            .thread_count = count,
        };

        for (self.workers, 0..) |*w, i| {
            w.* = Worker{
                .pool = self,
                .id = @intCast(i),
            };
        }

        for (self.workers) |*w| {
            w.thread = std.Thread.spawn(.{}, workerLoop, .{w}) catch null;
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.shutdown.store(true, .release);

        for (self.workers) |*w| {
            if (w.thread) |t| t.join();
        }

        self.allocator.free(self.workers);
        self.allocator.destroy(self);
    }

    /// Submit a task for execution. Returns false if all queues are full.
    pub fn schedule(self: *Self, comptime func: anytype, args: anytype) bool {
        const task = Task.init(func, args);
        return self.submitTask(task);
    }

    /// Submit a pre-built task
    pub fn submitTask(self: *Self, task: Task) bool {
        var min_load: usize = std.math.maxInt(usize);
        var target: u32 = 0;

        for (self.workers, 0..) |*w, i| {
            const load = w.queue.len();
            if (load < min_load) {
                min_load = load;
                target = @intCast(i);
            }
        }

        if (self.workers[target].queue.push(task)) {
            _ = self.tasks_submitted.fetchAdd(1, .monotonic);
            return true;
        }

        for (self.workers) |*w| {
            if (w.queue.push(task)) {
                _ = self.tasks_submitted.fetchAdd(1, .monotonic);
                return true;
            }
        }

        return false;
    }

    /// Spin-wait until all submitted tasks have completed.
    /// The calling thread helps drain queues to avoid deadlock.
    pub fn waitIdle(self: *Self) void {
        while (self.tasks_completed.load(.acquire) < self.tasks_submitted.load(.acquire)) {
            var found = false;
            for (self.workers) |*w| {
                if (w.queue.pop()) |task_copy| {
                    self.executeTaskAndRecord(task_copy);
                    found = true;
                    break;
                }
            }
            if (!found) std.atomic.spinLoopHint();
        }
    }

    pub fn isIdle(self: *const Self) bool {
        return self.tasks_completed.load(.acquire) >= self.tasks_submitted.load(.acquire);
    }

    pub fn stats(self: *const Self) Stats {
        var total_queued: usize = 0;
        for (self.workers) |*w| total_queued += w.queue.len();

        return .{
            .thread_count = self.thread_count,
            .tasks_submitted = self.tasks_submitted.load(.acquire),
            .tasks_completed = self.tasks_completed.load(.acquire),
            .tasks_queued = total_queued,
        };
    }

    pub const Stats = struct {
        thread_count: u32,
        tasks_submitted: u64,
        tasks_completed: u64,
        tasks_queued: usize,
    };

    fn executeTaskAndRecord(self: *Self, task_copy: Task) void {
        var task = task_copy;
        task.execute();
        _ = self.tasks_completed.fetchAdd(1, .monotonic);
    }

    // ── Worker Loop ──────────────────────────────────────────────────────

    fn workerLoop(worker: *Worker) void {
        const pool = worker.pool;

        while (!pool.shutdown.load(.acquire)) {
            if (worker.queue.pop()) |task_copy| {
                pool.executeTaskAndRecord(task_copy);
                continue;
            }

            if (pool.trySteal(worker.id)) |task_copy| {
                pool.executeTaskAndRecord(task_copy);
                continue;
            }

            for (0..32) |_| std.atomic.spinLoopHint();
            std.Thread.yield() catch {};
        }

        while (worker.queue.pop()) |task_copy| {
            pool.executeTaskAndRecord(task_copy);
        }
    }

    fn trySteal(self: *Self, caller_id: u32) ?Task {
        const start = (caller_id + 1) % self.thread_count;
        var i: u32 = 0;
        while (i < self.thread_count) : (i += 1) {
            const target = (start + i) % self.thread_count;
            if (target == caller_id) continue;
            if (self.workers[target].queue.steal()) |task| return task;
        }
        return null;
    }
};

// ─── Parallel For ────────────────────────────────────────────────────────────

/// Parallel iteration over a range, distributing chunks across the thread pool.
pub fn parallelFor(
    pool: *ThreadPool,
    total: usize,
    chunk_size: usize,
    comptime func: fn (usize, usize) void,
) void {
    var offset: usize = 0;
    while (offset < total) {
        const end = @min(offset + chunk_size, total);
        _ = pool.schedule(func, .{ offset, end });
        offset = end;
    }
    pool.waitIdle();
}

test "ThreadPool executes scheduled tasks and reaches idle state" {
    var pool = try ThreadPool.init(std.testing.allocator, .{ .thread_count = 2 });
    defer pool.deinit();

    var counter = std.atomic.Value(u64).init(0);
    const task_count: usize = 64;

    const increment = struct {
        fn run(counter_ptr: *std.atomic.Value(u64)) void {
            _ = counter_ptr.fetchAdd(1, .monotonic);
        }
    }.run;

    for (0..task_count) |_| {
        try std.testing.expect(pool.schedule(increment, .{&counter}));
    }

    pool.waitIdle();

    const stats = pool.stats();
    try std.testing.expect(pool.isIdle());
    try std.testing.expectEqual(@as(u64, task_count), counter.load(.acquire));
    try std.testing.expectEqual(stats.tasks_submitted, stats.tasks_completed);
}

test "ThreadPool stats reflect submissions" {
    var pool = try ThreadPool.init(std.testing.allocator, .{ .thread_count = 1 });
    defer pool.deinit();

    const noop = struct {
        fn run() void {}
    }.run;

    try std.testing.expect(pool.schedule(noop, .{}));
    try std.testing.expect(pool.schedule(noop, .{}));

    pool.waitIdle();
    const stats = pool.stats();
    try std.testing.expectEqual(@as(u64, 2), stats.tasks_submitted);
    try std.testing.expectEqual(@as(u64, 2), stats.tasks_completed);
}

test "ThreadPool isIdle starts true" {
    var pool = try ThreadPool.init(std.testing.allocator, .{ .thread_count = 1 });
    defer pool.deinit();

    // Pool should be idle initially (no tasks submitted)
    pool.waitIdle();
    try std.testing.expect(pool.isIdle());
}

test "Task type has correct max_capture_size" {
    // Verify the Task frame can hold reasonable closures
    try std.testing.expect(Task.max_capture_size >= 64);
    try std.testing.expect(@sizeOf(Task) > 0);
}
