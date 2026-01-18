const std = @import("std");
const time = @import("../../../shared/time.zig");
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
    subagents: std.StringHashMapUnmanaged(Subagent),
    tasks: std.ArrayListUnmanaged(Task),
    task_ids: std.StringHashMapUnmanaged(usize),
    semaphore: std.Thread.Semaphore,
    next_task_id: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) TaskTool {
        return TaskTool{
            .allocator = allocator,
            .subagents = .{},
            .tasks = std.ArrayListUnmanaged(Task){},
            .task_ids = .{},
            .semaphore = std.Thread.Semaphore.init(4),
        };
    }

    pub fn deinit(self: *TaskTool) void {
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
        self.task_ids.deinit(self.allocator);
        self.subagents.deinit(self.allocator);
    }

    pub fn registerSubagent(self: *TaskTool, name: []const u8, description: []const u8, handler: SubagentHandlerFn, config: SubagentConfig) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const description_copy = try self.allocator.dupe(u8, description);

        const subagent = Subagent{
            .name = name_copy,
            .description = description_copy,
            .config = config,
            .handler = handler,
        };

        try self.subagents.put(self.allocator, name_copy, subagent);
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

            const start_time = time.nowNanoseconds();

            const result = subagent.handler(task_input, &ctx) catch |err| {
                const err_msg = try std.fmt.allocPrint(self.allocator, "Execution failed: {}", .{err});
                defer self.allocator.free(err_msg);

                if (attempts < max_attempts - 1) {
                    time.sleepMs(subagent.config.retry_delay_ms);
                    continue;
                }
                return ToolResult.fromError(self.allocator, err_msg);
            };

            const end_time = time.nowNanoseconds();
            const duration_ms = @divTrunc(@as(i128, end_time - start_time), std.time.ns_per_ms);

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

        const task_id = try std.fmt.allocPrint(self.allocator, "task_{}", .{self.next_task_id});
        errdefer self.allocator.free(task_id);

        self.next_task_id += 1;

        const task = Task{
            .id = task_id,
            .subagent_name = subagent_name,
            .input = task_input,
            .status = .pending,
            .created_at = time.nowNanoseconds(),
            .timeout_ms = effective_timeout,
        };

        try self.tasks.append(self.allocator, task);
        try self.task_ids.put(self.allocator, task_id, self.tasks.items.len - 1);

        return task_id;
    }

    pub fn getTaskStatus(self: *TaskTool, task_id: []const u8) !TaskStatus {
        const index = self.task_ids.get(task_id) orelse return error.TaskNotFound;
        return self.tasks.items[index].status;
    }

    pub fn waitForTask(self: *TaskTool, task_id: []const u8) !ToolResult {
        const index = self.task_ids.get(task_id) orelse return error.TaskNotFound;
        const timeout_ms = self.tasks.items[index].timeout_ms;

        const start_time = time.nowNanoseconds();
        const deadline = start_time + (timeout_ms * std.time.ns_per_ms);

        while (true) {
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

            if (time.nowNanoseconds() > deadline) {
                self.tasks.items[index].status = .timeout;
                return ToolResult.fromError(self.allocator, "Task timed out");
            }

            time.sleepMs(100);
        }
    }

    pub fn cancelTask(self: *TaskTool, task_id: []const u8) !void {
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

            // Use {t} format specifier instead of @tagName()
            const state_str = std.fmt.allocPrint(self.allocator, "{t}", .{subagent.state}) catch |err| {
                std.log.warn("Failed to format subagent state: {t}", .{err});
                sub_stat.deinit();
                continue;
            };
            defer self.allocator.free(state_str);

            sub_stat.put("state", json.Value{ .string = state_str }) catch |err| {
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
