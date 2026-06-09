//! Task-based Concurrency Scheduler (Heap-Optimized)
const std = @import("std");
const sync = @import("../foundation/sync.zig");
const time = @import("../foundation/time.zig");
const errors = @import("../foundation/errors.zig");

const build_options = @import("build_options");
const metrics = if (build_options.feat_metrics) @import("../features/metrics/mod.zig") else @import("../features/metrics/stub.zig");
const telemetry = if (build_options.feat_telemetry) @import("../features/telemetry/mod.zig") else @import("../features/telemetry/stub.zig");
const memory = @import("memory.zig");

pub const TaskStatus = enum(u8) {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

pub const TaskPriority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    critical = 3,
};

pub const TaskFn = *const fn (ctx: ?*anyopaque) anyerror!void;

pub const Task = struct {
    id: u64,
    name: []const u8,
    priority: TaskPriority,
    status: TaskStatus,
    fn_ptr: TaskFn,
    ctx: ?*anyopaque,
    created_at: i64,
    started_at: i64,
    completed_at: i64,
    error_msg: ?[]const u8,
};

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    heap: std.PriorityQueue(Task, void, compareTasks),
    tasks: std.ArrayListUnmanaged(Task),
    lock: sync.SpinLock,
    next_id: std.atomic.Value(u64),
    pending_count: std.atomic.Value(usize),
    running_count: std.atomic.Value(usize),
    completed_count: std.atomic.Value(usize),
    failed_count: std.atomic.Value(usize),
    cancelled_count: std.atomic.Value(usize),

    // Metrics instance (populated only when -Dfeat-metrics; used for task lifecycle counters).
    // Part of approved "Deeper src/ Integration & Observability Pass".
    metrics_instance: ?metrics.Metrics = null,

    // Optional MemoryTracker for the allocations performed by/through this scheduler
    // (e.g. task names, TaskCtx arenas in training paths). Wired in Phase 1 memory step.
    memory_tracker: ?*memory.MemoryTracker = null,

    pub fn init(allocator: std.mem.Allocator) Scheduler {
        var self = Scheduler{
            .allocator = allocator,
            .heap = std.PriorityQueue(Task, void, compareTasks).initContext({}),
            .tasks = std.ArrayListUnmanaged(Task).empty,
            .lock = sync.SpinLock{},
            .next_id = std.atomic.Value(u64).init(1),
            .pending_count = std.atomic.Value(usize).init(0),
            .running_count = std.atomic.Value(usize).init(0),
            .completed_count = std.atomic.Value(usize).init(0),
            .failed_count = std.atomic.Value(usize).init(0),
            .cancelled_count = std.atomic.Value(usize).init(0),
        };
        if (comptime build_options.feat_metrics) {
            self.metrics_instance = metrics.Metrics.init(allocator);
        }
        return self;
    }

    pub fn deinit(self: *Scheduler) void {
        self.heap.deinit(self.allocator);
        for (self.tasks.items) |task| {
            self.allocator.free(task.name);
            if (task.error_msg) |message| self.allocator.free(message);
        }
        self.tasks.deinit(self.allocator);
        if (self.metrics_instance) |*m| {
            m.deinit();
        }
    }

    fn compareTasks(_: void, a: Task, b: Task) std.math.Order {
        return std.math.order(@intFromEnum(b.priority), @intFromEnum(a.priority));
    }

    pub fn submit(self: *Scheduler, name: []const u8, priority: TaskPriority, fn_ptr: TaskFn, ctx: ?*anyopaque) !u64 {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const task_id = self.next_id.fetchAdd(1, .monotonic);

        const task = Task{
            .id = task_id,
            .name = name_copy,
            .priority = priority,
            .status = .pending,
            .fn_ptr = fn_ptr,
            .ctx = ctx,
            .created_at = time.unixMs(),
            .started_at = 0,
            .completed_at = 0,
            .error_msg = null,
        };

        self.lock.lock();
        defer self.lock.unlock();

        try self.heap.push(self.allocator, task);
        try self.tasks.append(self.allocator, task);
        _ = self.pending_count.fetchAdd(1, .monotonic);

        self.recordMetric("scheduler.tasks.submitted", 1);
        return task_id;
    }

    pub fn cancel(self: *Scheduler, task_id: u64) !void {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.tasks.items) |*task| {
            if (task.id == task_id) {
                if (task.status != .pending) return errors.AbiError.InvalidState;
                task.status = .cancelled;
                _ = self.pending_count.fetchSub(1, .monotonic);
                _ = self.cancelled_count.fetchAdd(1, .monotonic);
                self.recordMetric("scheduler.tasks.cancelled", 1);
                return;
            }
        }
        return errors.AbiError.NotFound;
    }

    pub fn runNext(self: *Scheduler) !?u64 {
        self.lock.lock();
        var task: Task = undefined;
        while (self.heap.pop()) |t| {
            var exists = false;
            for (self.tasks.items) |*internal_t| {
                if (internal_t.id == t.id) {
                    if (internal_t.status != .pending) continue;
                    exists = true;
                    internal_t.status = .running;
                    internal_t.started_at = time.unixMs();
                    task = internal_t.*;
                    _ = self.pending_count.fetchSub(1, .monotonic);
                    break;
                }
            }
            if (exists) break;
        } else {
            self.lock.unlock();
            return null;
        }

        _ = self.running_count.fetchAdd(1, .monotonic);
        const task_id = task.id;
        const fn_ptr = task.fn_ptr;
        const ctx = task.ctx;

        self.lock.unlock();

        fn_ptr(ctx) catch |err| {
            self.lock.lock();
            defer self.lock.unlock();
            for (self.tasks.items) |*t| {
                if (t.id == task_id) {
                    t.status = .failed;
                    t.completed_at = time.unixMs();
                    t.error_msg = self.allocator.dupe(u8, @errorName(err)) catch null;
                }
            }
            _ = self.running_count.fetchSub(1, .monotonic);
            _ = self.failed_count.fetchAdd(1, .monotonic);
            self.recordMetric("scheduler.tasks.failed", 1);
            return err;
        };

        self.lock.lock();
        defer self.lock.unlock();
        for (self.tasks.items) |*t| {
            if (t.id == task_id) {
                t.status = .completed;
                t.completed_at = time.unixMs();
            }
        }
        _ = self.running_count.fetchSub(1, .monotonic);
        _ = self.completed_count.fetchAdd(1, .monotonic);
        self.recordMetric("scheduler.tasks.completed", 1);

        return task_id;
    }

    pub fn runAll(self: *Scheduler) !void {
        while (true) {
            const result = self.runNext() catch |err| {
                if (err == errors.AbiError.InvalidState or err == errors.AbiError.NotFound) continue;
                return err;
            };
            if (result == null) break;
        }
    }

    pub fn getTask(self: *const Scheduler, task_id: u64) ?*const Task {
        const self_mut = @constCast(self);
        self_mut.lock.lock();
        defer self_mut.lock.unlock();
        for (self.tasks.items) |*task| {
            if (task.id == task_id) return task;
        }
        return null;
    }

    pub fn getPendingCount(self: *const Scheduler) usize {
        return self.pending_count.load(.monotonic);
    }

    pub fn getRunningCount(self: *const Scheduler) usize {
        return self.running_count.load(.monotonic);
    }

    pub fn getCompletedCount(self: *const Scheduler) usize {
        return self.completed_count.load(.monotonic);
    }

    pub fn getFailedCount(self: *const Scheduler) usize {
        return self.failed_count.load(.monotonic);
    }

    pub fn getCancelledCount(self: *const Scheduler) usize {
        return self.cancelled_count.load(.monotonic);
    }

    /// Total number of tasks currently tracked by this scheduler across all
    /// statuses (pending + running + completed + failed + cancelled). Useful
    /// for dashboards and metrics to size the active task set.
    pub fn getTaskCount(self: *const Scheduler) usize {
        return self.pending_count.load(.monotonic) +
            self.running_count.load(.monotonic) +
            self.completed_count.load(.monotonic) +
            self.failed_count.load(.monotonic) +
            self.cancelled_count.load(.monotonic);
    }

    /// Peek the highest-priority task without removing it from the heap.
    /// Returns null when the scheduler is empty.
    pub fn peek(self: *const Scheduler) ?Task {
        const self_mut = @constCast(self);
        self_mut.lock.lock();
        defer self_mut.lock.unlock();
        return self.heap.peek();
    }

    /// Returns true when the scheduler has no pending tasks.
    pub fn isEmpty(self: *const Scheduler) bool {
        return self.peek() == null;
    }

    pub fn getTasks(self: *const Scheduler, allocator: std.mem.Allocator) ![]Task {
        const self_mut = @constCast(self);
        self_mut.lock.lock();
        defer self_mut.lock.unlock();
        const result = try allocator.alloc(Task, self.tasks.items.len);
        errdefer allocator.free(result);
        @memcpy(result, self.tasks.items);
        return result;
    }

    pub const Stats = struct {
        running: usize,
        pending: usize,
        completed: usize,
        failed: usize,
        cancelled: usize,
        total_tasks: usize,
    };

    /// Aggregate view of the primary task counters. Useful for dashboard/MCP
    /// observability without multiple round-trips or lock acquisitions.
    pub fn stats(self: *const Scheduler) Stats {
        const running = self.getRunningCount();
        const pending = self.getPendingCount();
        const completed = self.getCompletedCount();
        const failed = self.getFailedCount();
        const cancelled = self.getCancelledCount();
        return .{
            .running = running,
            .pending = pending,
            .completed = completed,
            .failed = failed,
            .cancelled = cancelled,
            .total_tasks = running + pending + completed + failed + cancelled,
        };
    }

    /// Convenience wrapper around `submit` with the same signature. Provided
    /// for callers that prefer the action-oriented name (matches the roadmap
    /// Stream 4 API proposal).
    pub fn scheduleTask(self: *Scheduler, name: []const u8, priority: TaskPriority, fn_ptr: TaskFn, ctx: ?*anyopaque) !u64 {
        return self.submit(name, priority, fn_ptr, ctx);
    }

    // --- Metrics wiring (Phase 1 of approved deeper-integration plan) ---
    // Real counters when -Dfeat-metrics; metric failures are logged and remain non-fatal.
    fn recordMetric(self: *Scheduler, name: []const u8, delta: u64) void {
        if (self.metrics_instance) |*m| {
            if (m.increment(name, delta)) |_| {} else |err| {
                std.log.warn("scheduler metric increment failed: name={s} err={s}", .{ name, @errorName(err) });
            }
        }
        // Always-on lightweight telemetry (no-op when -Dfeat-telemetry=false), so
        // the on-by-default telemetry sink observes task lifecycle even when the
        // richer opt-in metrics instance is absent.
        telemetry.increment(name, delta);
    }

    /// Attach an external MemoryTracker so this scheduler can participate
    /// in cross-layer memory observability (training paths, task ctxs, etc.).
    pub fn setMemoryTracker(self: *Scheduler, tracker: *memory.MemoryTracker) void {
        self.memory_tracker = tracker;
    }

    pub fn getMemoryTracker(self: *const Scheduler) ?*memory.MemoryTracker {
        return self.memory_tracker;
    }
};

test {
    std.testing.refAllDecls(@This());
    _ = metrics; // metrics feature wired for real task lifecycle counters (Phase 1 of approved integration plan)
}

fn dummyTask(ctx: ?*anyopaque) anyerror!void {
    _ = ctx;
}

fn failingTask(ctx: ?*anyopaque) anyerror!void {
    _ = ctx;
    return error.TestFailure;
}

test "Scheduler init and deinit" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    try std.testing.expectEqual(@as(usize, 0), scheduler.tasks.items.len);
    try std.testing.expectEqual(@as(usize, 0), scheduler.getPendingCount());
}

test "Scheduler submit task" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    const id = try scheduler.submit("test_task", .normal, dummyTask, null);
    try std.testing.expect(id >= 1);
    try std.testing.expectEqual(@as(usize, 1), scheduler.getPendingCount());
}

test "Scheduler run task" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("runnable", .high, dummyTask, null);
    const result = try scheduler.runNext();
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 1), scheduler.getCompletedCount());
}

test "Scheduler priority ordering" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("low_task", .low, dummyTask, null);
    _ = try scheduler.submit("high_task", .high, dummyTask, null);
    _ = try scheduler.submit("normal_task", .normal, dummyTask, null);

    const first = try scheduler.runNext();
    try std.testing.expect(first != null);

    const task = scheduler.getTask(first.?);
    try std.testing.expect(task != null);
    try std.testing.expectEqual(.high, task.?.priority);
}

test "Scheduler cancel pending task" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    const id = try scheduler.submit("cancel_me", .normal, dummyTask, null);
    try scheduler.cancel(id);

    const task = scheduler.getTask(id);
    try std.testing.expect(task != null);
    try std.testing.expectEqual(.cancelled, task.?.status);
}

test "Scheduler runAll" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("task1", .normal, dummyTask, null);
    _ = try scheduler.submit("task2", .normal, dummyTask, null);
    _ = try scheduler.submit("task3", .normal, dummyTask, null);

    try scheduler.runAll();
    try std.testing.expectEqual(@as(usize, 0), scheduler.getPendingCount());
    try std.testing.expectEqual(@as(usize, 3), scheduler.getCompletedCount());
}

test "Scheduler failed task tracking" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("failing", .normal, failingTask, null);

    const err = scheduler.runNext() catch |e| e;
    try std.testing.expectEqual(error.TestFailure, err);

    try std.testing.expectEqual(@as(usize, 1), scheduler.getFailedCount());
}

test "Scheduler stats aggregates counters" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("t1", .normal, dummyTask, null);
    _ = try scheduler.submit("t2", .high, dummyTask, null);

    var s = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 0), s.running);
    try std.testing.expectEqual(@as(usize, 2), s.pending);
    try std.testing.expectEqual(@as(usize, 0), s.completed);
    try std.testing.expectEqual(@as(usize, 0), s.failed);
    try std.testing.expectEqual(@as(usize, 0), s.cancelled);
    try std.testing.expectEqual(@as(usize, 2), s.total_tasks);

    _ = try scheduler.runNext(); // runs one (high priority)

    s = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 0), s.running);
    try std.testing.expectEqual(@as(usize, 1), s.pending);
    try std.testing.expectEqual(@as(usize, 1), s.completed);
    try std.testing.expectEqual(@as(usize, 0), s.failed);
    try std.testing.expectEqual(@as(usize, 0), s.cancelled);
    try std.testing.expectEqual(@as(usize, 2), s.total_tasks);
}

test "Scheduler peek returns highest priority task without removing" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    try std.testing.expect(scheduler.isEmpty());
    try std.testing.expect(scheduler.peek() == null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.getTaskCount());

    _ = try scheduler.submit("low_task", .low, dummyTask, null);
    _ = try scheduler.submit("high_task", .high, dummyTask, null);
    _ = try scheduler.submit("normal_task", .normal, dummyTask, null);

    try std.testing.expect(!scheduler.isEmpty());
    const peeked = scheduler.peek();
    try std.testing.expect(peeked != null);
    try std.testing.expectEqualStrings("high_task", peeked.?.name);
    try std.testing.expectEqual(TaskPriority.high, peeked.?.priority);
    try std.testing.expectEqual(@as(usize, 3), scheduler.getTaskCount());
    try std.testing.expectEqual(@as(usize, 3), scheduler.getPendingCount());

    // Peek must not consume the task
    const peeked_again = scheduler.peek();
    try std.testing.expect(peeked_again != null);
    try std.testing.expectEqual(peeked.?.id, peeked_again.?.id);
}

test "Scheduler scheduleTask is an alias for submit" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    const id_a = try scheduler.submit("direct", .normal, dummyTask, null);
    const id_b = try scheduler.scheduleTask("aliased", .normal, dummyTask, null);
    try std.testing.expect(id_a >= 1);
    try std.testing.expect(id_b > id_a);
    try std.testing.expectEqual(@as(usize, 2), scheduler.getPendingCount());
}

test "Scheduler cancelled counter and total_tasks update correctly" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    const id_a = try scheduler.submit("a", .normal, dummyTask, null);
    _ = try scheduler.submit("b", .normal, dummyTask, null);
    _ = try scheduler.submit("c", .normal, dummyTask, null);

    try scheduler.cancel(id_a);
    try std.testing.expectEqual(@as(usize, 2), scheduler.getPendingCount());
    try std.testing.expectEqual(@as(usize, 1), scheduler.getCancelledCount());
    try std.testing.expectEqual(@as(usize, 3), scheduler.getTaskCount());

    const s = scheduler.stats();
    try std.testing.expectEqual(@as(usize, 0), s.running);
    try std.testing.expectEqual(@as(usize, 2), s.pending);
    try std.testing.expectEqual(@as(usize, 0), s.completed);
    try std.testing.expectEqual(@as(usize, 0), s.failed);
    try std.testing.expectEqual(@as(usize, 1), s.cancelled);
    try std.testing.expectEqual(@as(usize, 3), s.total_tasks);
}

test "Scheduler pending_count atomic decrements on runNext" {
    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("p1", .normal, dummyTask, null);
    _ = try scheduler.submit("p2", .normal, dummyTask, null);
    try std.testing.expectEqual(@as(usize, 2), scheduler.getPendingCount());

    _ = try scheduler.runNext();
    try std.testing.expectEqual(@as(usize, 1), scheduler.getPendingCount());
    try std.testing.expectEqual(@as(usize, 1), scheduler.getCompletedCount());
    try std.testing.expectEqual(@as(usize, 2), scheduler.getTaskCount());
}

test "Scheduler feeds telemetry counters across the task lifecycle" {
    if (!build_options.feat_telemetry) return;
    telemetry.reset();
    defer telemetry.reset();

    var scheduler = Scheduler.init(std.testing.allocator);
    defer scheduler.deinit();

    _ = try scheduler.submit("t1", .normal, dummyTask, null);
    const cancel_id = try scheduler.submit("t2", .low, dummyTask, null);
    try scheduler.cancel(cancel_id);
    try scheduler.runAll();

    // submitted x2, completed x1 (t1), cancelled x1 (t2) — all routed through
    // recordMetric into the on-by-default telemetry sink.
    try std.testing.expect(telemetry.counterValue("scheduler.tasks.submitted") >= 2);
    try std.testing.expect(telemetry.counterValue("scheduler.tasks.completed") >= 1);
    try std.testing.expect(telemetry.counterValue("scheduler.tasks.cancelled") >= 1);
}
