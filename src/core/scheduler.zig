//! Task-based Concurrency Scheduler (Heap-Optimized)
const std = @import("std");
const sync = @import("../foundation/sync.zig");
const time = @import("../foundation/time.zig");
const errors = @import("../foundation/errors.zig");

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
    running_count: std.atomic.Value(usize),
    completed_count: std.atomic.Value(usize),
    failed_count: std.atomic.Value(usize),

    pub fn init(allocator: std.mem.Allocator) Scheduler {
        return .{
            .allocator = allocator,
            .heap = std.PriorityQueue(Task, void, compareTasks).initContext(allocator, {}),
            .tasks = std.ArrayListUnmanaged(Task).empty,
            .lock = sync.SpinLock{},
            .next_id = std.atomic.Value(u64).init(1),
            .running_count = std.atomic.Value(usize).init(0),
            .completed_count = std.atomic.Value(usize).init(0),
            .failed_count = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *Scheduler) void {
        self.heap.deinit(self.allocator);
        self.tasks.deinit(self.allocator);
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

        try self.heap.add(self.allocator, task);
        try self.tasks.append(self.allocator, task);

        return task_id;
    }

    pub fn cancel(self: *Scheduler, task_id: u64) !void {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.tasks.items) |*task| {
            if (task.id == task_id) {
                if (task.status != .pending) return errors.AbiError.InvalidState;
                task.status = .cancelled;
                return;
            }
        }
        return errors.AbiError.NotFound;
    }

    pub fn runNext(self: *Scheduler) !?u64 {
        self.lock.lock();
        var task: Task = undefined;
        while (self.heap.removeOrNull()) |t| {
            var exists = false;
            for (self.tasks.items) |*internal_t| {
                if (internal_t.id == t.id) {
                    if (internal_t.status == .cancelled) continue;
                    exists = true;
                    internal_t.status = .running;
                    internal_t.started_at = time.unixMs();
                    task = internal_t.*;
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
        defer self.lock.lock();

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
            return err;
        };

        self.lock.lock();
        for (self.tasks.items) |*t| {
            if (t.id == task_id) {
                t.status = .completed;
                t.completed_at = time.unixMs();
            }
        }
        _ = self.running_count.fetchSub(1, .monotonic);
        _ = self.completed_count.fetchAdd(1, .monotonic);

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
        const self_mut = @constCast(self);
        self_mut.lock.lock();
        defer self_mut.lock.unlock();
        var count: usize = 0;
        for (self.tasks.items) |task| {
            if (task.status == .pending) count += 1;
        }
        return count;
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

    pub fn getTasks(self: *const Scheduler, allocator: std.mem.Allocator) ![]Task {
        const self_mut = @constCast(self);
        self_mut.lock.lock();
        defer self_mut.lock.unlock();
        const result = try allocator.alloc(Task, self.tasks.items.len);
        errdefer allocator.free(result);
        @memcpy(result, self.tasks.items);
        return result;
    }
};

test {
    std.testing.refAllDecls(@This());
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
