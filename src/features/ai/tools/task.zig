const std = @import("std");
const json = std.json;
const Tool = @import("tool.zig").Tool;
const ToolResult = @import("tool.zig").ToolResult;
const Context = @import("tool.zig").Context;
const ToolExecutionError = @import("tool.zig").ToolExecutionError;

/// Error set for subagent operations
pub const SubagentError = error{
    /// The requested subagent was not found
    SubagentNotFound,
    /// The task was not found
    TaskNotFound,
    /// Maximum retries exceeded
    MaxRetriesExceeded,
    /// Task execution timed out
    TaskTimeout,
    /// Input validation failed
    InvalidInput,
    /// Subagent is currently busy
    SubagentBusy,
    /// Handler returned an error
    HandlerFailed,
} || std.mem.Allocator.Error;

/// Function pointer type for subagent handlers.
pub const SubagentHandlerFn = *const fn ([]const u8, *Context) SubagentError!ToolResult;

pub const SubagentConfig = struct {
    timeout_ms: u64 = 30000,
    max_concurrent: usize = 4,
    memory_limit_mb: usize = 512,
    retry_count: u3 = 3,
    retry_delay_ms: u64 = 1000,
};

pub const SubagentState = enum {
    idle,
    running,
    completed,
    failed,
    cancelled,
};

pub const Subagent = struct {
    name: []const u8,
    description: []const u8,
    config: SubagentConfig,
    handler: SubagentHandlerFn,
    state: SubagentState = .idle,
    last_execution_time_ms: u64 = 0,
    execution_count: u64 = 0,
    error_count: u64 = 0,
};

pub const TaskStatus = enum {
    pending,
    running,
    completed,
    failed,
    cancelled,
    timeout,
};

pub const Task = struct {
    id: []const u8,
    subagent_name: []const u8,
    input: []const u8,
    status: TaskStatus,
    result: ?ToolResult = null,
    error_message: ?[]const u8 = null,
    created_at: i128,
    started_at: ?i128 = null,
    completed_at: ?i128 = null,
    timeout_ms: u64,
    retry_count: u3 = 0,
};

pub const TaskTool = struct {
    allocator: std.mem.Allocator,
    subagents: std.StringHashMap(Subagent),
    tasks: std.ArrayListUnmanaged(Task),
    task_ids: std.StringHashMap(usize),
    semaphore: std.Thread.Semaphore,
    next_task_id: u64 = 0,
    worker_running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    worker_thread: ?std.Thread = null,
    task_mutex: std.Thread.Mutex = .{},

    pub fn init(allocator: std.mem.Allocator) TaskTool {
        return TaskTool{
            .allocator = allocator,
            .subagents = std.StringHashMap(Subagent).init(allocator),
            .tasks = std.ArrayListUnmanaged(Task){},
            .task_ids = std.StringHashMap(usize).init(allocator),
            .semaphore = std.Thread.Semaphore.init(4),
        };
    }

    /// Start the background worker thread that processes async tasks
    pub fn startWorker(self: *TaskTool) !void {
        if (self.worker_running.load(.acquire)) return;

        self.worker_running.store(true, .release);
        self.worker_thread = std.Thread.spawn(.{}, workerLoop, .{self}) catch |err| {
            self.worker_running.store(false, .release);
            return err;
        };
    }

    /// Stop the background worker thread
    pub fn stopWorker(self: *TaskTool) void {
        self.worker_running.store(false, .release);
        if (self.worker_thread) |thread| {
            thread.join();
            self.worker_thread = null;
        }
    }

    /// Background worker loop that processes pending tasks
    fn workerLoop(self: *TaskTool) void {
        while (self.worker_running.load(.acquire)) {
            var task_to_execute: ?struct {
                index: usize,
                subagent_name: []const u8,
                input: []const u8,
            } = null;

            // Find a pending task
            {
                self.task_mutex.lock();
                defer self.task_mutex.unlock();

                for (self.tasks.items, 0..) |*task, i| {
                    if (task.status == .pending) {
                        task.status = .running;
                        task.started_at = std.time.nanoTimestamp();
                        task_to_execute = .{
                            .index = i,
                            .subagent_name = task.subagent_name,
                            .input = task.input,
                        };
                        break;
                    }
                }
            }

            // Execute the task if found
            if (task_to_execute) |task_info| {
                self.executeTask(task_info.index, task_info.subagent_name, task_info.input);
            } else {
                // No pending tasks, sleep briefly
                std.time.sleep(10 * std.time.ns_per_ms);
            }
        }
    }

    /// Execute a task and update its status
    fn executeTask(self: *TaskTool, task_index: usize, subagent_name: []const u8, input: []const u8) void {
        const subagent = self.subagents.get(subagent_name) orelse {
            self.task_mutex.lock();
            defer self.task_mutex.unlock();
            self.tasks.items[task_index].status = .failed;
            self.tasks.items[task_index].error_message = "Subagent not found";
            self.tasks.items[task_index].completed_at = std.time.nanoTimestamp();
            return;
        };

        // Acquire semaphore slot
        self.semaphore.wait();
        defer self.semaphore.post();

        var ctx = Context{
            .allocator = self.allocator,
            .working_directory = ".",
            .environment = null,
            .cancellation = null,
        };

        // Execute with retry logic
        var attempts: u3 = 0;
        const max_attempts = subagent.config.retry_count + 1;

        while (attempts < max_attempts) : (attempts += 1) {
            // Check for cancellation
            {
                self.task_mutex.lock();
                defer self.task_mutex.unlock();
                if (self.tasks.items[task_index].status == .cancelled) {
                    return;
                }
            }

            const result = subagent.handler(input, &ctx) catch |err| {
                if (attempts < max_attempts - 1) {
                    std.time.sleep(subagent.config.retry_delay_ms * std.time.ns_per_ms);
                    continue;
                }
                // Final attempt failed
                self.task_mutex.lock();
                defer self.task_mutex.unlock();
                self.tasks.items[task_index].status = .failed;
                self.tasks.items[task_index].error_message = std.fmt.allocPrint(
                    self.allocator,
                    "Execution failed: {}",
                    .{err},
                ) catch "Execution failed";
                self.tasks.items[task_index].completed_at = std.time.nanoTimestamp();

                // Update subagent error count
                if (self.subagents.getPtr(subagent_name)) |sa| {
                    sa.error_count += 1;
                }
                return;
            };

            // Success
            self.task_mutex.lock();
            defer self.task_mutex.unlock();
            self.tasks.items[task_index].status = .completed;
            self.tasks.items[task_index].result = result;
            self.tasks.items[task_index].completed_at = std.time.nanoTimestamp();

            // Update subagent stats
            if (self.subagents.getPtr(subagent_name)) |sa| {
                sa.execution_count += 1;
                if (self.tasks.items[task_index].started_at) |started| {
                    const duration = std.time.nanoTimestamp() - started;
                    sa.last_execution_time_ms = @intCast(@divTrunc(duration, std.time.ns_per_ms));
                }
            }
            return;
        }
    }

    pub fn deinit(self: *TaskTool) void {
        // Stop worker thread first
        self.stopWorker();

        for (self.tasks.items) |*task| {
            self.allocator.free(task.id);
            self.allocator.free(task.subagent_name);
            self.allocator.free(task.input);
            if (task.result) |*res| {
                res.deinit();
            }
            if (task.error_message) |err| {
                self.allocator.free(err);
            }
        }
        self.tasks.deinit(self.allocator);
        self.task_ids.deinit();
        self.subagents.deinit();
    }

    pub fn registerSubagent(self: *TaskTool, name: []const u8, description: []const u8, handler: *const fn ([]const u8, *Context) anyerror!ToolResult, config: SubagentConfig) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const description_copy = try self.allocator.dupe(u8, description);

        const subagent = Subagent{
            .name = name_copy,
            .description = description_copy,
            .config = config,
            .handler = handler,
        };

        try self.subagents.put(name_copy, subagent);
    }

    pub fn invoke(self: *TaskTool, subagent_name: []const u8, task_input: []const u8, _: ?u64) !ToolResult {
        const subagent = self.subagents.get(subagent_name) orelse {
            return ToolResult.fromError(self.allocator, "Subagent not found");
        };

        var ctx = Context{
            .allocator = self.allocator,
            .working_directory = ".",
            .environment = null,
            .cancellation = null,
        };

        var attempts: u3 = 0;
        const max_attempts = subagent.config.retry_count + 1;

        while (attempts < max_attempts) : (attempts += 1) {
            self.semaphore.wait();
            defer self.semaphore.post();

            var timer = std.time.Timer.start() catch {
                return ToolResult.fromError(self.allocator, "Failed to start timer");
            };

            const result = subagent.handler(task_input, &ctx) catch |err| {
                const err_msg = try std.fmt.allocPrint(self.allocator, "Execution failed: {}", .{err});
                defer self.allocator.free(err_msg);

                if (attempts < max_attempts - 1) {
                    std.time.sleep(subagent.config.retry_delay_ms * std.time.ns_per_ms);
                    continue;
                }
                return ToolResult.fromError(self.allocator, err_msg);
            };

            const elapsed_ns = timer.read();
            const duration_ms = @divTrunc(elapsed_ns, std.time.ns_per_ms);

            var mutable_subagent = self.subagents.getPtr(subagent_name).?;
            mutable_subagent.last_execution_time_ms = @as(u64, @intCast(duration_ms));
            mutable_subagent.execution_count += 1;

            return result;
        }

        return ToolResult.fromError(self.allocator, "Max retries exceeded");
    }

    pub fn invokeAsync(self: *TaskTool, subagent_name: []const u8, task_input: []const u8, timeout_ms: ?u64) ![]const u8 {
        const subagent = self.subagents.get(subagent_name) orelse {
            return error.SubagentNotFound;
        };

        const effective_timeout = timeout_ms orelse subagent.config.timeout_ms;

        // Auto-start worker if not running
        if (!self.worker_running.load(.acquire)) {
            try self.startWorker();
        }

        const task_id = try std.fmt.allocPrint(self.allocator, "task_{}", .{self.next_task_id});
        errdefer self.allocator.free(task_id);

        // Duplicate input strings so they're owned by the task
        const subagent_name_copy = try self.allocator.dupe(u8, subagent_name);
        errdefer self.allocator.free(subagent_name_copy);

        const input_copy = try self.allocator.dupe(u8, task_input);
        errdefer self.allocator.free(input_copy);

        self.next_task_id += 1;

        const task = Task{
            .id = task_id,
            .subagent_name = subagent_name_copy,
            .input = input_copy,
            .status = .pending,
            .created_at = std.time.nanoTimestamp(),
            .timeout_ms = effective_timeout,
        };

        self.task_mutex.lock();
        defer self.task_mutex.unlock();

        try self.tasks.append(self.allocator, task);
        try self.task_ids.put(task_id, self.tasks.items.len - 1);

        return task_id;
    }

    pub fn getTaskStatus(self: *TaskTool, task_id: []const u8) !TaskStatus {
        self.task_mutex.lock();
        defer self.task_mutex.unlock();

        const index = self.task_ids.get(task_id) orelse return error.TaskNotFound;
        return self.tasks.items[index].status;
    }

    pub fn waitForTask(self: *TaskTool, task_id: []const u8) !ToolResult {
        // Get initial values with lock
        const index: usize = blk: {
            self.task_mutex.lock();
            defer self.task_mutex.unlock();
            break :blk self.task_ids.get(task_id) orelse return error.TaskNotFound;
        };

        const timeout_ms: u64 = blk: {
            self.task_mutex.lock();
            defer self.task_mutex.unlock();
            break :blk self.tasks.items[index].timeout_ms;
        };

        var timer = std.time.Timer.start() catch {
            return ToolResult.fromError(self.allocator, "Failed to start timer");
        };
        const timeout_ns = timeout_ms * std.time.ns_per_ms;

        while (true) {
            // Check status with lock
            {
                self.task_mutex.lock();
                defer self.task_mutex.unlock();

                const current_status = self.tasks.items[index].status;
                if (current_status == .completed or current_status == .failed or current_status == .cancelled or current_status == .timeout) {
                    if (self.tasks.items[index].result) |res| {
                        return res;
                    }
                    if (self.tasks.items[index].error_message) |err| {
                        return ToolResult.fromError(self.allocator, err);
                    }
                    return ToolResult.fromError(self.allocator, "Task completed without result");
                }
            }

            if (timer.read() > timeout_ns) {
                self.task_mutex.lock();
                defer self.task_mutex.unlock();
                self.tasks.items[index].status = .timeout;
                return ToolResult.fromError(self.allocator, "Task timed out");
            }

            std.time.sleep(50 * std.time.ns_per_ms);
        }
    }

    pub fn cancelTask(self: *TaskTool, task_id: []const u8) !void {
        self.task_mutex.lock();
        defer self.task_mutex.unlock();

        const index = self.task_ids.get(task_id) orelse return error.TaskNotFound;
        self.tasks.items[index].status = .cancelled;
    }

    pub fn listSubagents(self: *TaskTool) ![]const []const u8 {
        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer names.deinit(self.allocator);
        var iterator = self.subagents.keyIterator();
        while (iterator.next()) |key| {
            try names.append(self.allocator, key.*);
        }
        return try names.toOwnedSlice(self.allocator);
    }

    pub fn getSubagentInfo(self: *TaskTool, name: []const u8) ?*const Subagent {
        return self.subagents.get(name);
    }

    pub fn getStatistics(self: *TaskTool) !json.Value {
        var stats = json.Object.init(self.allocator);
        errdefer stats.deinit();

        var subagent_stats = json.Array.init(self.allocator);
        errdefer subagent_stats.deinit();

        var iterator = self.subagents.valueIterator();
        while (iterator.next()) |subagent| {
            var sub_stat = json.Object.init(self.allocator);
            errdefer sub_stat.deinit();

            sub_stat.put("name", json.Value{ .string = subagent.name }) catch |err| {
                std.log.warn("Failed to add subagent name to statistics: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            sub_stat.put("state", json.Value{ .string = @tagName(subagent.state) }) catch |err| {
                std.log.warn("Failed to add subagent state to statistics: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            sub_stat.put("execution_count", json.Value{ .integer = @as(i64, @intCast(subagent.execution_count)) }) catch |err| {
                std.log.warn("Failed to add execution count to statistics: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            sub_stat.put("error_count", json.Value{ .integer = @as(i64, @intCast(subagent.error_count)) }) catch |err| {
                std.log.warn("Failed to add error count to statistics: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            sub_stat.put("last_execution_time_ms", json.Value{ .integer = @as(i64, @intCast(subagent.last_execution_time_ms)) }) catch |err| {
                std.log.warn("Failed to add execution time to statistics: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            subagent_stats.append(json.Value{ .object = sub_stat }) catch |err| {
                std.log.warn("Failed to append subagent stats: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
        }

        try stats.put("subagents", json.Value{ .array = subagent_stats });
        try stats.put("pending_tasks", json.Value{ .integer = @as(i64, @intCast(self.tasks.items.len)) });

        return json.Value{ .object = stats };
    }
};
