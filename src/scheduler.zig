const std = @import("std");
const sync = @import("foundation/sync.zig");
const time = @import("foundation/time.zig");
const errors = @import("foundation/errors.zig");

pub const TaskStatus = enum(u8) {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

pub const TaskPriority = enum(u8) {
    low,
    normal,
    high,
    critical,
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
    tasks: std.ArrayListUnmanaged(Task),
    lock: sync.SpinLock,
    next_id: std.atomic.Value(u64),
    running_count: std.atomic.Value(usize),
    completed_count: std.atomic.Value(usize),
    failed_count: std.atomic.Value(usize),

    pub fn init(allocator: std.mem.Allocator) Scheduler {
        return .{
            .allocator = allocator,
            .tasks = std.ArrayListUnmanaged(Task).empty,
            .lock = sync.SpinLock{},
            .next_id = std.atomic.Value(u64).init(1),
            .running_count = std.atomic.Value(usize).init(0),
            .completed_count = std.atomic.Value(usize).init(0),
            .failed_count = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *Scheduler) void {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.tasks.items) |*task| {
            if (task.error_msg) |msg| {
                self.allocator.free(msg);
            }
        }
        self.tasks.deinit(self.allocator);
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
        errdefer self.lock.unlock();

        try self.tasks.append(self.allocator, task);
        self.lock.unlock();

        return task_id;
    }

    pub fn cancel(self: *Scheduler, task_id: u64) !void {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.tasks.items) |*task| {
            if (task.id == task_id) {
                switch (task.status) {
                    .pending => {
                        task.status = .cancelled;
                        return;
                    },
                    .running, .completed, .failed, .cancelled => {
                        return errors.AbiError.InvalidState;
                    },
                }
            }
        }
        return errors.AbiError.NotFound;
    }

    pub fn runNext(self: *Scheduler) !?u64 {
        self.lock.lock();
        defer self.lock.unlock();

        var best_idx: ?usize = null;
        var best_priority: TaskPriority = .low;

        for (self.tasks.items, 0..) |*task, i| {
            if (task.status == .pending) {
                if (best_idx == null or @intFromEnum(task.priority) > @intFromEnum(best_priority)) {
                    best_idx = i;
                    best_priority = task.priority;
                }
            }
        }

        if (best_idx) |idx| {
            const task = &self.tasks.items[idx];
            task.status = .running;
            task.started_at = time.unixMs();
            self.running_count.fetchAdd(1, .monotonic);

            const task_id = task.id;
            const fn_ptr = task.fn_ptr;
            const ctx = task.ctx;

            self.lock.unlock();
            defer self.lock.lock();

            fn_ptr(ctx) catch |err| {
                self.lock.lock();
                defer self.lock.unlock();

                task.status = .failed;
                task.completed_at = time.unixMs();
                task.error_msg = self.allocator.dupe(u8, @errorName(err)) catch null;
                self.running_count.fetchSub(1, .monotonic);
                self.failed_count.fetchAdd(1, .monotonic);
                return err;
            };

            self.lock.lock();
            defer self.lock.unlock();

            task.status = .completed;
            task.completed_at = time.unixMs();
            self.running_count.fetchSub(1, .monotonic);
            self.completed_count.fetchAdd(1, .monotonic);

            return task_id;
        }

        return null;
    }

    pub fn runAll(self: *Scheduler) !void {
        while (true) {
            const result = self.runNext() catch |err| {
                if (err == errors.AbiError.InvalidState or err == errors.AbiError.NotFound) {
                    continue;
                }
                return err;
            };
            if (result == null) break;
        }
    }

    pub fn getTask(self: *const Scheduler, task_id: u64) ?*const Task {
        for (self.tasks.items) |*task| {
            if (task.id == task_id) return task;
        }
        return null;
    }

    pub fn getPendingCount(self: *const Scheduler) usize {
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

    pub fn getTasks(self: *const Scheduler) []const Task {
        return self.tasks.items;
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
