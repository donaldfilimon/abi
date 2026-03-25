const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const retry = @import("../../../../foundation/utils/retry.zig");
const aggregation = @import("../aggregation.zig");
const messaging = @import("../messaging.zig");
const types = @import("../types.zig");

const Error = types.Error;
const AgentHealth = types.AgentHealth;
const AgentResult = types.AgentResult;
const CoordinatorStats = types.CoordinatorStats;

pub fn deinit(self: anytype) void {
    for (self.results.items) |result| {
        self.allocator.free(result.response);
    }
    self.results.deinit(self.allocator);
    for (self.mailboxes.items) |*mailbox| mailbox.deinit();
    self.mailboxes.deinit(self.allocator);
    self.health.deinit(self.allocator);
    self.agents.deinit(self.allocator);
    if (self.event_bus) |*bus| bus.deinit();
    self.* = undefined;
}

pub fn register(self: anytype, agent_ptr: anytype) Error!void {
    if (self.agents.items.len >= self.config.max_agents) {
        return Error.MaxAgentsReached;
    }
    self.agents.append(self.allocator, agent_ptr) catch return Error.ExecutionFailed;
    self.health.append(self.allocator, .{
        .failure_threshold = self.config.circuit_breaker_threshold,
    }) catch return Error.ExecutionFailed;
    self.mailboxes.append(self.allocator, messaging.AgentMailbox.init(self.allocator)) catch
        return Error.ExecutionFailed;
}

pub fn getAgentHealth(self: anytype, index: usize) ?AgentHealth {
    if (index >= self.health.items.len) return null;
    return self.health.items[index];
}

pub fn sendMessage(self: anytype, msg: messaging.AgentMessage) Error!void {
    if (msg.to_agent >= self.mailboxes.items.len) return Error.AgentNotFound;
    self.mailboxes.items[msg.to_agent].send(msg) catch return Error.ExecutionFailed;
}

pub fn pendingMessages(self: anytype, agent_index: usize) ?usize {
    if (agent_index >= self.mailboxes.items.len) return null;
    return self.mailboxes.items[agent_index].pendingCount();
}

pub fn agentCount(self: anytype) usize {
    return self.agents.items.len;
}

pub fn onEvent(self: anytype, event_type: messaging.EventType, callback: messaging.EventCallback) !void {
    if (self.event_bus) |*bus| {
        try bus.subscribe(event_type, callback);
    }
}

pub fn runTask(self: anytype, task: []const u8) Error![]u8 {
    if (self.agents.items.len == 0) return Error.NoAgents;

    const task_id = messaging.taskId(task);
    if (self.event_bus) |*bus| bus.taskStarted(task_id);

    var task_timer = time.Timer.start() catch null;

    for (self.results.items) |result| {
        self.allocator.free(result.response);
    }
    self.results.clearRetainingCapacity();

    switch (self.config.execution_strategy) {
        .sequential => try executeSequential(self, task),
        .parallel => try executeParallel(self, task),
        .pipeline => try executePipeline(self, task),
        .adaptive => try executeAdaptive(self, task),
    }

    const aggregated = aggregateResults(self) catch |err| {
        if (self.event_bus) |*bus| bus.taskFailed(task_id, "aggregation failed");
        return err;
    };

    const duration = if (task_timer) |*timer| timer.read() else 0;
    if (self.event_bus) |*bus| bus.taskCompleted(task_id, duration);

    return aggregated;
}

pub fn getStats(self: anytype) CoordinatorStats {
    var stats = CoordinatorStats{
        .agent_count = self.agents.items.len,
        .result_count = self.results.items.len,
    };

    var total_duration: u64 = 0;
    for (self.results.items) |result| {
        if (result.success) {
            stats.success_count += 1;
        }
        total_duration += result.duration_ns;
    }

    if (self.results.items.len > 0) {
        stats.avg_duration_ns = total_duration / self.results.items.len;
    }

    return stats;
}

fn executeSequential(self: anytype, task: []const u8) Error!void {
    const task_id = messaging.taskId(task);
    const timeout_ns: u64 = self.config.agent_timeout_ms * std.time.ns_per_ms;

    for (self.agents.items, 0..) |agent, i| {
        if (self.config.circuit_breaker_threshold > 0 and
            i < self.health.items.len and
            !self.health.items[i].canAttempt())
        {
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = self.allocator.dupe(u8, "[Skipped: circuit open]") catch return Error.ExecutionFailed,
                .success = false,
                .duration_ns = 0,
            }) catch return Error.ExecutionFailed;
            continue;
        }

        if (self.event_bus) |*bus| bus.agentStarted(task_id, i);
        var timer = time.Timer.start() catch null;

        const response = processWithRetry(agent, task, self.allocator, self.config.retry_config) catch {
            const duration = if (timer) |*current_timer| current_timer.read() else 0;
            if (self.event_bus) |*bus| bus.agentFinished(task_id, i, false, duration);
            if (i < self.health.items.len) self.health.items[i].recordFailure();
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = self.allocator.dupe(u8, "[Error: execution failed]") catch return Error.ExecutionFailed,
                .success = false,
                .duration_ns = duration,
            }) catch return Error.ExecutionFailed;
            continue;
        };

        const duration = if (timer) |*current_timer| current_timer.read() else 0;

        if (timeout_ns > 0 and duration > timeout_ns) {
            self.allocator.free(response);
            if (i < self.health.items.len) self.health.items[i].recordFailure();
            if (self.event_bus) |*bus| bus.agentFinished(task_id, i, false, duration);
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = self.allocator.dupe(u8, "[Error: agent timed out]") catch return Error.ExecutionFailed,
                .success = false,
                .duration_ns = duration,
                .timed_out = true,
            }) catch return Error.ExecutionFailed;
            continue;
        }

        if (i < self.health.items.len) self.health.items[i].recordSuccess();
        if (self.event_bus) |*bus| bus.agentFinished(task_id, i, true, duration);
        self.results.append(self.allocator, .{
            .agent_index = i,
            .response = response,
            .success = true,
            .duration_ns = duration,
        }) catch {
            self.allocator.free(response);
            return Error.ExecutionFailed;
        };
    }
}

fn executeParallel(self: anytype, task: []const u8) Error!void {
    if (!self.config.enable_parallel or self.agents.items.len <= 1) {
        return executeSequential(self, task);
    }

    const agent_count = self.agents.items.len;
    const timeout_ns: u64 = self.config.agent_timeout_ms * std.time.ns_per_ms;

    const thread_results = self.allocator.alloc(ThreadResult, agent_count) catch
        return Error.ExecutionFailed;
    defer self.allocator.free(thread_results);

    const thread_spawned = self.allocator.alloc(bool, agent_count) catch
        return Error.ExecutionFailed;
    defer self.allocator.free(thread_spawned);
    @memset(thread_spawned, false);

    for (thread_results) |*thread_result| {
        thread_result.* = .{};
    }

    const threads = self.allocator.alloc(std.Thread, agent_count) catch
        return Error.ExecutionFailed;
    defer self.allocator.free(threads);

    for (self.agents.items, 0..) |agent, i| {
        if (self.config.circuit_breaker_threshold > 0 and
            i < self.health.items.len and
            !self.health.items[i].canAttempt())
        {
            thread_results[i] = .{
                .response = self.allocator.dupe(u8, "[Skipped: circuit open]") catch null,
                .success = false,
                .duration_ns = 0,
            };
            continue;
        }

        threads[i] = std.Thread.spawn(.{}, runAgentThread, .{
            agent,
            task,
            self.allocator,
            &thread_results[i],
            timeout_ns,
            self.config.retry_config,
        }) catch {
            thread_results[i] = .{
                .response = self.allocator.dupe(u8, "[Error: thread spawn failed]") catch null,
                .success = false,
                .duration_ns = 0,
            };
            continue;
        };
        thread_spawned[i] = true;
    }

    for (0..agent_count) |i| {
        if (thread_spawned[i]) {
            threads[i].join();
        }
    }

    for (thread_results, 0..) |thread_result, i| {
        if (i < self.health.items.len) {
            if (thread_result.success) {
                self.health.items[i].recordSuccess();
            } else {
                self.health.items[i].recordFailure();
            }
        }
        self.results.append(self.allocator, .{
            .agent_index = i,
            .response = thread_result.response orelse (self.allocator.dupe(u8, "[Error: no response]") catch return Error.ExecutionFailed),
            .success = thread_result.success,
            .duration_ns = thread_result.duration_ns,
            .timed_out = thread_result.timed_out,
        }) catch return Error.ExecutionFailed;
    }
}

fn executePipeline(self: anytype, task: []const u8) Error!void {
    var current_input = task;
    var owned_input: ?[]u8 = null;
    defer if (owned_input) |input| self.allocator.free(input);

    for (self.agents.items, 0..) |agent, i| {
        var timer = time.Timer.start() catch null;

        const response = agent.process(current_input, self.allocator) catch {
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = self.allocator.dupe(u8, "[Error: pipeline stage failed]") catch return Error.ExecutionFailed,
                .success = false,
                .duration_ns = if (timer) |*current_timer| current_timer.read() else 0,
            }) catch return Error.ExecutionFailed;
            return Error.ExecutionFailed;
        };

        self.results.append(self.allocator, .{
            .agent_index = i,
            .response = response,
            .success = true,
            .duration_ns = if (timer) |*current_timer| current_timer.read() else 0,
        }) catch {
            self.allocator.free(response);
            return Error.ExecutionFailed;
        };

        if (owned_input) |input| self.allocator.free(input);
        owned_input = self.allocator.dupe(u8, response) catch return Error.ExecutionFailed;
        current_input = owned_input.?;
    }
}

fn executeAdaptive(self: anytype, task: []const u8) Error!void {
    if (self.agents.items.len > 2 and task.len < 1000) {
        return executeParallel(self, task);
    }
    return executeSequential(self, task);
}

fn aggregateResults(self: anytype) Error![]u8 {
    if (self.results.items.len == 0) return Error.NoAgents;

    return switch (self.config.aggregation_strategy) {
        .concatenate => aggregateConcatenate(self),
        .vote => aggregateVote(self),
        .select_best => aggregateSelectBest(self),
        .merge => aggregateMerge(self),
        .first_success => aggregateFirstSuccess(self),
    };
}

fn aggregateConcatenate(self: anytype) Error![]u8 {
    var builder = std.ArrayListUnmanaged(u8).empty;
    errdefer builder.deinit(self.allocator);

    for (self.results.items, 0..) |result, i| {
        if (i > 0) {
            builder.appendSlice(self.allocator, "\n---\n") catch return Error.AggregationFailed;
        }
        builder.appendSlice(self.allocator, result.response) catch return Error.AggregationFailed;
    }

    return builder.toOwnedSlice(self.allocator) catch return Error.AggregationFailed;
}

fn aggregateVote(self: anytype) Error![]u8 {
    const outputs = self.allocator.alloc(aggregation.AgentOutput, self.results.items.len) catch
        return Error.AggregationFailed;
    defer self.allocator.free(outputs);

    for (self.results.items, 0..) |result, i| {
        outputs[i] = .{
            .response = result.response,
            .success = result.success,
            .agent_index = result.agent_index,
            .duration_ns = result.duration_ns,
        };
    }

    return aggregation.hashVote(self.allocator, outputs) catch return Error.AggregationFailed;
}

fn aggregateSelectBest(self: anytype) Error![]u8 {
    const outputs = self.allocator.alloc(aggregation.AgentOutput, self.results.items.len) catch
        return Error.AggregationFailed;
    defer self.allocator.free(outputs);

    for (self.results.items, 0..) |result, i| {
        outputs[i] = .{
            .response = result.response,
            .success = result.success,
            .agent_index = result.agent_index,
            .duration_ns = result.duration_ns,
        };
    }

    return aggregation.weightedSelect(self.allocator, outputs) catch return Error.AggregationFailed;
}

fn aggregateMerge(self: anytype) Error![]u8 {
    const outputs = self.allocator.alloc(aggregation.AgentOutput, self.results.items.len) catch
        return Error.AggregationFailed;
    defer self.allocator.free(outputs);

    for (self.results.items, 0..) |result, i| {
        outputs[i] = .{
            .response = result.response,
            .success = result.success,
            .agent_index = result.agent_index,
            .duration_ns = result.duration_ns,
        };
    }

    return aggregation.sectionMerge(self.allocator, outputs) catch return Error.AggregationFailed;
}

fn aggregateFirstSuccess(self: anytype) Error![]u8 {
    for (self.results.items) |result| {
        if (result.success) {
            return self.allocator.dupe(u8, result.response) catch return Error.AggregationFailed;
        }
    }
    return Error.AggregationFailed;
}

fn processWithRetry(agent: anytype, task: []const u8, allocator: std.mem.Allocator, cfg: retry.RetryConfig) ![]u8 {
    var attempt: u32 = 0;
    while (true) {
        const result = agent.process(task, allocator) catch |err| {
            if (attempt >= cfg.max_retries) return err;
            attempt += 1;
            const delay_ms = retry.calculateDelay(cfg, attempt);
            time.sleepMs(delay_ms);
            continue;
        };
        return result;
    }
}

const ThreadResult = struct {
    response: ?[]u8 = null,
    success: bool = false,
    duration_ns: u64 = 0,
    timed_out: bool = false,
};

fn runAgentThread(
    agent: anytype,
    task: []const u8,
    allocator: std.mem.Allocator,
    result: *ThreadResult,
    timeout_ns: u64,
    retry_cfg: retry.RetryConfig,
) void {
    var timer = time.Timer.start() catch null;

    const response = processWithRetry(agent, task, allocator, retry_cfg) catch {
        result.* = .{
            .response = allocator.dupe(u8, "[Error: execution failed]") catch null,
            .success = false,
            .duration_ns = if (timer) |*current_timer| current_timer.read() else 0,
        };
        return;
    };

    const duration = if (timer) |*current_timer| current_timer.read() else 0;

    if (timeout_ns > 0 and duration > timeout_ns) {
        allocator.free(response);
        result.* = .{
            .response = allocator.dupe(u8, "[Error: agent timed out]") catch null,
            .success = false,
            .duration_ns = duration,
            .timed_out = true,
        };
        return;
    }

    result.* = .{
        .response = response,
        .success = true,
        .duration_ns = duration,
    };
}
