//! Task group for managing collections of related tasks.
//!
//! Provides structured concurrency primitives for spawning and
//! managing groups of tasks with collective wait operations.

const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const cancellation_mod = @import("cancellation.zig");
const future_mod = @import("future.zig");

const cancellation = cancellation_mod;
const future = future_mod;

const CancellationSource = cancellation.CancellationSource;
const CancellationToken = cancellation.CancellationToken;
const Future = future.Future;

/// Task identifier.
pub const TaskId = u64;

/// Common task execution errors
pub const TaskError = std.mem.Allocator.Error || cancellation.CancellationError || error{
    TaskFailed,
    ExecutionTimeout,
    InvalidState,
    ResourceExhausted,
};

/// Task state.
pub const TaskState = enum {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

/// Task completion result.
pub const TaskResult = union(enum) {
    success: void,
    failure: TaskError,
    cancelled: void,

    pub fn isSuccess(self: TaskResult) bool {
        return self == .success;
    }

    pub fn isFailure(self: TaskResult) bool {
        return self == .failure;
    }
};

/// Task metadata.
pub const TaskInfo = struct {
    id: TaskId,
    state: TaskState,
    name: ?[]const u8,
    submit_time_ns: i64,
    start_time_ns: ?i64,
    end_time_ns: ?i64,
    result: ?TaskResult,

    /// Get duration in nanoseconds (if completed).
    pub fn durationNs(self: TaskInfo) ?i64 {
        if (self.start_time_ns) |start| {
            if (self.end_time_ns) |end| {
                return end - start;
            }
        }
        return null;
    }
};

/// Task group configuration.
pub const TaskGroupConfig = struct {
    /// Maximum concurrent tasks (0 = unlimited).
    max_concurrent: usize = 0,
    /// Cancel all tasks on first failure.
    cancel_on_failure: bool = false,
    /// Group name for debugging.
    name: ?[]const u8 = null,
    /// Enable task timing.
    track_timing: bool = true,
};

/// Task function type.
pub const TaskFn = *const fn (*TaskContext) TaskError!void;

/// Context passed to task functions.
pub const TaskContext = struct {
    allocator: std.mem.Allocator,
    task_id: TaskId,
    cancellation_token: CancellationToken,
    user_data: ?*anyopaque,

    /// Check if cancellation was requested.
    pub fn isCancelled(self: *const TaskContext) bool {
        return self.cancellation_token.isCancelled();
    }

    /// Throw if cancellation was requested.
    pub fn checkCancellation(self: *const TaskContext) !void {
        return self.cancellation_token.throwIfCancellationRequested();
    }
};

/// Task group for managing collections of tasks.
pub const TaskGroup = struct {
    allocator: std.mem.Allocator,
    config: TaskGroupConfig,
    tasks: std.ArrayListUnmanaged(Task),
    mutex: sync.Mutex,
    condition: std.Thread.Condition,
    next_id: std.atomic.Value(u64),
    cancellation_source: CancellationSource,
    active_count: std.atomic.Value(usize),
    completed_count: std.atomic.Value(usize),
    failed_count: std.atomic.Value(usize),
    is_closed: bool,

    const Task = struct {
        id: TaskId,
        func: TaskFn,
        user_data: ?*anyopaque,
        name: ?[]const u8,
        state: TaskState,
        result: ?TaskResult,
        submit_time_ns: i64,
        start_time_ns: ?i64,
        end_time_ns: ?i64,
    };

    /// Initialize a new task group.
    pub fn init(allocator: std.mem.Allocator, config: TaskGroupConfig) TaskGroup {
        return .{
            .allocator = allocator,
            .config = config,
            .tasks = .{},
            .mutex = .{},
            .condition = .{},
            .next_id = std.atomic.Value(u64).init(1),
            .cancellation_source = CancellationSource.init(allocator),
            .active_count = std.atomic.Value(usize).init(0),
            .completed_count = std.atomic.Value(usize).init(0),
            .failed_count = std.atomic.Value(usize).init(0),
            .is_closed = false,
        };
    }

    /// Deinitialize the task group.
    pub fn deinit(self: *TaskGroup) void {
        self.cancellation_source.deinit();
        self.tasks.deinit(self.allocator);
        self.* = undefined;
    }

    /// Submit a task to the group.
    pub fn submit(self: *TaskGroup, func: TaskFn) !TaskId {
        return self.submitWithData(func, null, null);
    }

    /// Submit a task with name.
    pub fn submitNamed(self: *TaskGroup, func: TaskFn, name: []const u8) !TaskId {
        return self.submitWithData(func, null, name);
    }

    /// Submit a task with user data.
    pub fn submitWithData(self: *TaskGroup, func: TaskFn, user_data: ?*anyopaque, name: ?[]const u8) !TaskId {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.is_closed) {
            return error.GroupClosed;
        }

        // Check concurrency limit
        if (self.config.max_concurrent > 0) {
            while (self.active_count.load(.monotonic) >= self.config.max_concurrent) {
                self.condition.wait(&self.mutex);
            }
        }

        const id = self.next_id.fetchAdd(1, .monotonic);
        const now: i64 = @intCast(platform_time.nowNanoseconds());

        const task = Task{
            .id = id,
            .func = func,
            .user_data = user_data,
            .name = name,
            .state = .pending,
            .result = null,
            .submit_time_ns = now,
            .start_time_ns = null,
            .end_time_ns = null,
        };

        try self.tasks.append(self.allocator, task);
        return id;
    }

    /// Run a specific task (typically called by executor).
    pub fn runTask(self: *TaskGroup, task_id: TaskId) !void {
        const task_idx = self.findTaskIndex(task_id) orelse return error.TaskNotFound;

        self.mutex.lock();
        self.tasks.items[task_idx].state = .running;
        if (self.config.track_timing) {
            self.tasks.items[task_idx].start_time_ns = @intCast(platform_time.nowNanoseconds());
        }
        const func = self.tasks.items[task_idx].func;
        const user_data = self.tasks.items[task_idx].user_data;
        self.mutex.unlock();

        _ = self.active_count.fetchAdd(1, .monotonic);
        defer {
            _ = self.active_count.fetchSub(1, .monotonic);
            self.condition.broadcast();
        }

        var ctx = TaskContext{
            .allocator = self.allocator,
            .task_id = task_id,
            .cancellation_token = self.cancellation_source.getToken(),
            .user_data = user_data,
        };

        const result: TaskResult = if (self.cancellation_source.isCancelled())
            .cancelled
        else if (func(&ctx)) |_|
            .success
        else |err|
            .{ .failure = err };

        self.mutex.lock();
        defer self.mutex.unlock();

        if (task_idx < self.tasks.items.len and self.tasks.items[task_idx].id == task_id) {
            self.tasks.items[task_idx].result = result;
            self.tasks.items[task_idx].state = switch (result) {
                .success => .completed,
                .failure => .failed,
                .cancelled => .cancelled,
            };

            if (self.config.track_timing) {
                self.tasks.items[task_idx].end_time_ns = @intCast(platform_time.nowNanoseconds());
            }

            switch (result) {
                .success => _ = self.completed_count.fetchAdd(1, .monotonic),
                .failure => {
                    _ = self.failed_count.fetchAdd(1, .monotonic);
                    if (self.config.cancel_on_failure) {
                        self.cancellation_source.cancel();
                    }
                },
                .cancelled => {},
            }
        }
    }

    /// Run all pending tasks sequentially.
    pub fn runAll(self: *TaskGroup) !void {
        while (true) {
            const task_id = self.getNextPending() orelse break;
            try self.runTask(task_id);
        }
    }

    /// Wait for all tasks to complete.
    pub fn waitAll(self: *TaskGroup) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.hasPendingOrRunning()) {
            self.condition.wait(&self.mutex);
        }
    }

    /// Wait for all tasks with timeout.
    pub fn waitAllTimeout(self: *TaskGroup, timeout_ns: u64) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.hasPendingOrRunning()) return true;

        self.condition.timedWait(&self.mutex, timeout_ns) catch |err| {
            if (err == error.Timeout) return false;
            return err;
        };

        return !self.hasPendingOrRunning();
    }

    /// Wait for any task to complete.
    pub fn waitAny(self: *TaskGroup) !?TaskId {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (true) {
            for (self.tasks.items) |task| {
                if (task.state == .completed or task.state == .failed or task.state == .cancelled) {
                    return task.id;
                }
            }

            if (!self.hasPendingOrRunning()) return null;
            self.condition.wait(&self.mutex);
        }
    }

    /// Cancel all tasks in the group.
    pub fn cancel(self: *TaskGroup) void {
        self.cancellation_source.cancel();
    }

    /// Close the group (no more tasks can be submitted).
    pub fn close(self: *TaskGroup) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.is_closed = true;
    }

    /// Get task information.
    pub fn getTaskInfo(self: *TaskGroup, task_id: TaskId) ?TaskInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.tasks.items) |task| {
            if (task.id == task_id) {
                return TaskInfo{
                    .id = task.id,
                    .state = task.state,
                    .name = task.name,
                    .submit_time_ns = task.submit_time_ns,
                    .start_time_ns = task.start_time_ns,
                    .end_time_ns = task.end_time_ns,
                    .result = task.result,
                };
            }
        }
        return null;
    }

    /// Get group statistics.
    pub fn getStats(self: *TaskGroup) GroupStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var pending: usize = 0;
        var running: usize = 0;
        var completed: usize = 0;
        var failed: usize = 0;
        var cancelled: usize = 0;

        for (self.tasks.items) |task| {
            switch (task.state) {
                .pending => pending += 1,
                .running => running += 1,
                .completed => completed += 1,
                .failed => failed += 1,
                .cancelled => cancelled += 1,
            }
        }

        return .{
            .total = self.tasks.items.len,
            .pending = pending,
            .running = running,
            .completed = completed,
            .failed = failed,
            .cancelled = cancelled,
            .is_closed = self.is_closed,
        };
    }

    /// Get all task results.
    pub fn getResults(self: *TaskGroup) ![]TaskInfo {
        self.mutex.lock();
        defer self.mutex.unlock();

        var results = try self.allocator.alloc(TaskInfo, self.tasks.items.len);
        for (self.tasks.items, 0..) |task, i| {
            results[i] = TaskInfo{
                .id = task.id,
                .state = task.state,
                .name = task.name,
                .submit_time_ns = task.submit_time_ns,
                .start_time_ns = task.start_time_ns,
                .end_time_ns = task.end_time_ns,
                .result = task.result,
            };
        }
        return results;
    }

    // Internal helpers
    fn findTaskIndex(self: *TaskGroup, task_id: TaskId) ?usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.tasks.items, 0..) |task, i| {
            if (task.id == task_id) return i;
        }
        return null;
    }

    fn getNextPending(self: *TaskGroup) ?TaskId {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.tasks.items) |task| {
            if (task.state == .pending) return task.id;
        }
        return null;
    }

    fn hasPendingOrRunning(self: *const TaskGroup) bool {
        for (self.tasks.items) |task| {
            if (task.state == .pending or task.state == .running) {
                return true;
            }
        }
        return false;
    }
};

/// Group statistics.
pub const GroupStats = struct {
    total: usize,
    pending: usize,
    running: usize,
    completed: usize,
    failed: usize,
    cancelled: usize,
    is_closed: bool,

    pub fn successRate(self: GroupStats) f64 {
        const done = self.completed + self.failed + self.cancelled;
        if (done == 0) return 0;
        return @as(f64, @floatFromInt(self.completed)) / @as(f64, @floatFromInt(done));
    }
};

/// Builder for task groups with fluent API.
pub const TaskGroupBuilder = struct {
    allocator: std.mem.Allocator,
    config: TaskGroupConfig,

    pub fn init(allocator: std.mem.Allocator) TaskGroupBuilder {
        return .{
            .allocator = allocator,
            .config = .{},
        };
    }

    pub fn maxConcurrent(self: *TaskGroupBuilder, max: usize) *TaskGroupBuilder {
        self.config.max_concurrent = max;
        return self;
    }

    pub fn cancelOnFailure(self: *TaskGroupBuilder, cancel: bool) *TaskGroupBuilder {
        self.config.cancel_on_failure = cancel;
        return self;
    }

    pub fn name(self: *TaskGroupBuilder, n: []const u8) *TaskGroupBuilder {
        self.config.name = n;
        return self;
    }

    pub fn trackTiming(self: *TaskGroupBuilder, track: bool) *TaskGroupBuilder {
        self.config.track_timing = track;
        return self;
    }

    pub fn build(self: *TaskGroupBuilder) TaskGroup {
        return TaskGroup.init(self.allocator, self.config);
    }
};

/// Scoped task group that cancels on scope exit.
pub const ScopedTaskGroup = struct {
    group: TaskGroup,

    pub fn init(allocator: std.mem.Allocator, config: TaskGroupConfig) ScopedTaskGroup {
        return .{ .group = TaskGroup.init(allocator, config) };
    }

    pub fn deinit(self: *ScopedTaskGroup) void {
        self.group.cancel();
        self.group.deinit();
    }

    pub fn submit(self: *ScopedTaskGroup, func: TaskFn) !TaskId {
        return self.group.submit(func);
    }

    pub fn runAll(self: *ScopedTaskGroup) !void {
        return self.group.runAll();
    }
};

/// Parallel for-each over a slice.
pub fn parallelForEach(
    comptime T: type,
    allocator: std.mem.Allocator,
    items: []const T,
    func: *const fn (T) void,
) !void {
    var group = TaskGroup.init(allocator, .{});
    defer group.deinit();

    // Create wrapper that captures item
    // Note: The wrapper uses TaskError to match TaskFn signature.
    const Wrapper = struct {
        item: T,
        func_ptr: *const fn (T) void,

        fn run(ctx: *TaskContext) TaskError!void {
            const self: *@This() = @ptrCast(@alignCast(ctx.user_data.?));
            self.func_ptr(self.item);
        }
    };

    var wrappers = try allocator.alloc(Wrapper, items.len);
    defer allocator.free(wrappers);

    for (items, 0..) |item, i| {
        wrappers[i] = .{ .item = item, .func_ptr = func };
        _ = try group.submitWithData(Wrapper.run, &wrappers[i], null);
    }

    try group.runAll();
}

test "task group basic" {
    const allocator = std.testing.allocator;
    var group = TaskGroup.init(allocator, .{});
    defer group.deinit();

    // Test task functions use TaskError to stay aligned with production error handling.
    const task_fn = struct {
        fn run(_: *TaskContext) TaskError!void {
            // Simple task that does nothing
        }
    }.run;

    const id1 = try group.submit(task_fn);
    const id2 = try group.submit(task_fn);

    try std.testing.expect(id1 != id2);

    try group.runAll();

    const stats = group.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.completed);
}

test "task group cancellation" {
    const allocator = std.testing.allocator;
    var group = TaskGroup.init(allocator, .{});
    defer group.deinit();

    const task_fn = struct {
        fn run(ctx: *TaskContext) TaskError!void {
            try ctx.checkCancellation();
        }
    }.run;

    _ = try group.submit(task_fn);

    group.cancel();
    try group.runAll();

    const stats = group.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.cancelled);
}

test "task group stats" {
    const allocator = std.testing.allocator;
    var group = TaskGroup.init(allocator, .{});
    defer group.deinit();

    const success_fn = struct {
        fn run(_: *TaskContext) TaskError!void {}
    }.run;

    // Note: TaskFailed is part of TaskError and is used here to test failure handling.
    const fail_fn = struct {
        fn run(_: *TaskContext) TaskError!void {
            return error.TaskFailed;
        }
    }.run;

    _ = try group.submit(success_fn);
    _ = try group.submit(success_fn);
    _ = try group.submit(fail_fn);

    try group.runAll();

    const stats = group.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.total);
    try std.testing.expectEqual(@as(usize, 2), stats.completed);
    try std.testing.expectEqual(@as(usize, 1), stats.failed);
}

test "task group builder" {
    const allocator = std.testing.allocator;

    var builder = TaskGroupBuilder.init(allocator);
    var group = builder
        .maxConcurrent(4)
        .cancelOnFailure(true)
        .name("test-group")
        .build();
    defer group.deinit();

    try std.testing.expectEqual(@as(usize, 4), group.config.max_concurrent);
    try std.testing.expect(group.config.cancel_on_failure);
    try std.testing.expectEqualStrings("test-group", group.config.name.?);
}
