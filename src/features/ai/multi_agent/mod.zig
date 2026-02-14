//! Multi‑Agent Coordination Module
//!
//! Provides a sophisticated orchestrator that can run a collection of
//! `agents.Agent` instances on a given task. Supports both sequential
//! and parallel execution with multiple aggregation strategies.
//!
//! ## Features
//!
//! - **Execution Strategies**: Sequential, parallel, pipeline, adaptive
//! - **Aggregation Strategies**: Concatenate, vote, select best, merge
//! - **Message Passing**: Agents can communicate via event bus
//! - **Health Monitoring**: Track agent status and handle failures
//! - **Real Parallel Execution**: Uses `std.Thread` for concurrent agent processing
//!
//! ## Example
//!
//! ```zig
//! const multi_agent = @import("multi_agent/mod.zig");
//!
//! var coord = multi_agent.Coordinator.init(allocator, .{
//!     .execution_strategy = .parallel,
//!     .aggregation_strategy = .vote,
//! });
//! defer coord.deinit();
//!
//! try coord.registerAgent(agent1);
//! try coord.registerAgent(agent2);
//!
//! const result = try coord.runTask("Analyze this code");
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const agents = @import("../agents/mod.zig");
const build_options = @import("build_options");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");

pub const Error = error{
    AgentDisabled, // Underlying agents module disabled
    NoAgents, // No agents registered in the coordinator
    MaxAgentsReached, // Cannot register more agents
    AgentNotFound, // Agent ID not found
    ExecutionFailed, // Task execution failed
    AggregationFailed, // Result aggregation failed
    Timeout, // Operation timed out
};

/// How to execute tasks across agents.
pub const ExecutionStrategy = enum {
    /// Run agents one at a time in order.
    sequential,
    /// Run all agents simultaneously.
    parallel,
    /// Each agent's output becomes next agent's input.
    pipeline,
    /// Choose strategy based on task characteristics.
    adaptive,

    pub fn toString(self: ExecutionStrategy) []const u8 {
        return @tagName(self);
    }
};

/// How to combine results from multiple agents.
pub const AggregationStrategy = enum {
    /// Concatenate all responses with separators.
    concatenate,
    /// Return the most common response (majority vote via hash).
    vote,
    /// Return the response from the healthiest/best agent.
    select_best,
    /// Merge responses by deduplicating markdown sections.
    merge,
    /// Return first successful response.
    first_success,

    pub fn toString(self: AggregationStrategy) []const u8 {
        return @tagName(self);
    }
};

/// Agent execution result.
pub const AgentResult = struct {
    agent_index: usize,
    response: []u8,
    success: bool,
    duration_ns: u64,
};

/// Configuration for the coordinator.
pub const CoordinatorConfig = struct {
    /// How to execute tasks.
    execution_strategy: ExecutionStrategy = .sequential,
    /// How to aggregate results.
    aggregation_strategy: AggregationStrategy = .concatenate,
    /// Maximum number of agents.
    max_agents: u32 = 100,
    /// Timeout per agent in milliseconds.
    agent_timeout_ms: u64 = 30_000,
    /// Enable parallel execution (requires threading).
    enable_parallel: bool = true,
    /// Enable event bus for lifecycle events.
    enable_events: bool = false,
    /// Maximum threads for parallel execution (0 = auto-detect).
    max_threads: u32 = 0,

    pub fn defaults() CoordinatorConfig {
        return .{};
    }
};

/// Coordinator holds a list of agents and orchestrates task execution.
pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    config: CoordinatorConfig,
    agents: std.ArrayListUnmanaged(*agents.Agent) = .{},
    results: std.ArrayListUnmanaged(AgentResult) = .{},
    mutex: sync.Mutex = .{},
    event_bus: ?messaging.EventBus = null,

    /// Initialise the coordinator with an allocator and default config.
    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return initWithConfig(allocator, .{});
    }

    /// Initialise the coordinator with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: CoordinatorConfig) Coordinator {
        return .{
            .allocator = allocator,
            .config = config,
            .agents = .{},
            .results = .{},
            .mutex = .{},
            .event_bus = if (config.enable_events) messaging.EventBus.init(allocator) else null,
        };
    }

    /// Deinitialise and free resources.
    pub fn deinit(self: *Coordinator) void {
        // Free any stored results
        for (self.results.items) |result| {
            self.allocator.free(result.response);
        }
        self.results.deinit(self.allocator);
        self.agents.deinit(self.allocator);
        if (self.event_bus) |*bus| bus.deinit();
        self.* = undefined;
    }

    /// Register an existing agent instance.
    pub fn register(self: *Coordinator, agent_ptr: *agents.Agent) Error!void {
        if (self.agents.items.len >= self.config.max_agents) {
            return Error.MaxAgentsReached;
        }
        self.agents.append(self.allocator, agent_ptr) catch return Error.ExecutionFailed;
    }

    /// Get the number of registered agents.
    pub fn agentCount(self: *const Coordinator) usize {
        return self.agents.items.len;
    }

    /// Subscribe to coordinator lifecycle events.
    pub fn onEvent(self: *Coordinator, event_type: messaging.EventType, callback: messaging.EventCallback) !void {
        if (self.event_bus) |*bus| {
            try bus.subscribe(event_type, callback);
        }
    }

    /// Run a textual task across all registered agents.
    /// Returns the aggregated output based on the configured strategy.
    pub fn runTask(self: *Coordinator, task: []const u8) Error![]u8 {
        if (self.agents.items.len == 0) return Error.NoAgents;

        const task_id = messaging.taskId(task);
        if (self.event_bus) |*bus| bus.taskStarted(task_id);

        var task_timer = time.Timer.start() catch null;

        // Clear previous results
        for (self.results.items) |result| {
            self.allocator.free(result.response);
        }
        self.results.clearRetainingCapacity();

        // Execute based on strategy
        switch (self.config.execution_strategy) {
            .sequential => try self.executeSequential(task),
            .parallel => try self.executeParallel(task),
            .pipeline => try self.executePipeline(task),
            .adaptive => try self.executeAdaptive(task),
        }

        // Aggregate results
        const aggregated = self.aggregateResults() catch |err| {
            if (self.event_bus) |*bus| bus.taskFailed(task_id, "aggregation failed");
            return err;
        };

        const duration = if (task_timer) |*t| t.read() else 0;
        if (self.event_bus) |*bus| bus.taskCompleted(task_id, duration);

        return aggregated;
    }

    /// Execute agents sequentially.
    fn executeSequential(self: *Coordinator, task: []const u8) Error!void {
        const task_id = messaging.taskId(task);

        for (self.agents.items, 0..) |ag, i| {
            if (self.event_bus) |*bus| bus.agentStarted(task_id, i);
            var timer = time.Timer.start() catch null;

            const response = ag.process(task, self.allocator) catch {
                const dur = if (timer) |*t| t.read() else 0;
                if (self.event_bus) |*bus| bus.agentFinished(task_id, i, false, dur);
                self.results.append(self.allocator, .{
                    .agent_index = i,
                    .response = self.allocator.dupe(u8, "[Error: execution failed]") catch return Error.ExecutionFailed,
                    .success = false,
                    .duration_ns = dur,
                }) catch return Error.ExecutionFailed;
                continue;
            };

            const dur = if (timer) |*t| t.read() else 0;
            if (self.event_bus) |*bus| bus.agentFinished(task_id, i, true, dur);
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = response,
                .success = true,
                .duration_ns = dur,
            }) catch {
                self.allocator.free(response);
                return Error.ExecutionFailed;
            };
        }
    }

    /// Execute agents in parallel using std.Thread.
    fn executeParallel(self: *Coordinator, task: []const u8) Error!void {
        if (!self.config.enable_parallel or self.agents.items.len <= 1) {
            return self.executeSequential(task);
        }

        const agent_count = self.agents.items.len;

        // Allocate per-thread result slots
        const thread_results = self.allocator.alloc(ThreadResult, agent_count) catch
            return Error.ExecutionFailed;
        defer self.allocator.free(thread_results);

        // Initialize all results
        for (thread_results) |*tr| {
            tr.* = .{};
        }

        // Spawn threads for each agent
        const threads = self.allocator.alloc(std.Thread, agent_count) catch
            return Error.ExecutionFailed;
        defer self.allocator.free(threads);

        var spawned: usize = 0;
        for (self.agents.items, 0..) |ag, i| {
            threads[i] = std.Thread.spawn(.{}, runAgentThread, .{
                ag,
                task,
                self.allocator,
                &thread_results[i],
            }) catch {
                // If spawn fails, mark as failed
                thread_results[i] = .{
                    .response = self.allocator.dupe(u8, "[Error: thread spawn failed]") catch null,
                    .success = false,
                    .duration_ns = 0,
                };
                continue;
            };
            spawned += 1;
        }

        // Join all spawned threads
        for (0..agent_count) |i| {
            if (thread_results[i].response != null or thread_results[i].success) {
                // Thread was spawned, join it
                if (i < spawned) {
                    threads[i].join();
                }
            }
        }

        // Collect results
        for (thread_results, 0..) |tr, i| {
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = tr.response orelse (self.allocator.dupe(u8, "[Error: no response]") catch return Error.ExecutionFailed),
                .success = tr.success,
                .duration_ns = tr.duration_ns,
            }) catch return Error.ExecutionFailed;
        }
    }

    /// Execute agents as a pipeline (output of one becomes input of next).
    fn executePipeline(self: *Coordinator, task: []const u8) Error!void {
        var current_input = task;
        var owned_input: ?[]u8 = null;
        defer if (owned_input) |o| self.allocator.free(o);

        for (self.agents.items, 0..) |ag, i| {
            var timer = time.Timer.start() catch null;

            const response = ag.process(current_input, self.allocator) catch {
                self.results.append(self.allocator, .{
                    .agent_index = i,
                    .response = self.allocator.dupe(u8, "[Error: pipeline stage failed]") catch return Error.ExecutionFailed,
                    .success = false,
                    .duration_ns = if (timer) |*t| t.read() else 0,
                }) catch return Error.ExecutionFailed;
                return Error.ExecutionFailed; // Pipeline broken
            };

            // Store result
            self.results.append(self.allocator, .{
                .agent_index = i,
                .response = response,
                .success = true,
                .duration_ns = if (timer) |*t| t.read() else 0,
            }) catch {
                self.allocator.free(response);
                return Error.ExecutionFailed;
            };

            // Update input for next stage
            if (owned_input) |o| self.allocator.free(o);
            owned_input = self.allocator.dupe(u8, response) catch return Error.ExecutionFailed;
            current_input = owned_input.?;
        }
    }

    /// Adaptively choose execution strategy based on task/agent characteristics.
    fn executeAdaptive(self: *Coordinator, task: []const u8) Error!void {
        // Use parallel for multiple agents with short tasks,
        // sequential for long tasks or few agents
        if (self.agents.items.len > 2 and task.len < 1000) {
            return self.executeParallel(task);
        }
        return self.executeSequential(task);
    }

    /// Aggregate results based on configured strategy.
    fn aggregateResults(self: *Coordinator) Error![]u8 {
        if (self.results.items.len == 0) return Error.NoAgents;

        return switch (self.config.aggregation_strategy) {
            .concatenate => self.aggregateConcatenate(),
            .vote => self.aggregateVote(),
            .select_best => self.aggregateSelectBest(),
            .merge => self.aggregateMerge(),
            .first_success => self.aggregateFirstSuccess(),
        };
    }

    /// Concatenate all responses with separators.
    fn aggregateConcatenate(self: *Coordinator) Error![]u8 {
        var builder = std.ArrayListUnmanaged(u8){};
        errdefer builder.deinit(self.allocator);

        for (self.results.items, 0..) |result, i| {
            if (i > 0) {
                builder.appendSlice(self.allocator, "\n---\n") catch return Error.AggregationFailed;
            }
            builder.appendSlice(self.allocator, result.response) catch return Error.AggregationFailed;
        }

        return builder.toOwnedSlice(self.allocator) catch return Error.AggregationFailed;
    }

    /// Return most common response using hash-based majority vote.
    fn aggregateVote(self: *Coordinator) Error![]u8 {
        // Convert results to AgentOutput for the aggregation module
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

    /// Return response from best performing agent using quality heuristics.
    fn aggregateSelectBest(self: *Coordinator) Error![]u8 {
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

    /// Merge responses by deduplicating markdown sections.
    fn aggregateMerge(self: *Coordinator) Error![]u8 {
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

    /// Return first successful response.
    fn aggregateFirstSuccess(self: *Coordinator) Error![]u8 {
        for (self.results.items) |result| {
            if (result.success) {
                return self.allocator.dupe(u8, result.response) catch return Error.AggregationFailed;
            }
        }
        return Error.AggregationFailed;
    }

    /// Get execution statistics.
    pub fn getStats(self: *const Coordinator) CoordinatorStats {
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
};

/// Thread result slot for parallel execution.
const ThreadResult = struct {
    response: ?[]u8 = null,
    success: bool = false,
    duration_ns: u64 = 0,
};

/// Thread function for parallel agent execution.
fn runAgentThread(
    ag: *agents.Agent,
    task: []const u8,
    allocator: std.mem.Allocator,
    result: *ThreadResult,
) void {
    var timer = time.Timer.start() catch null;

    const response = ag.process(task, allocator) catch {
        result.* = .{
            .response = allocator.dupe(u8, "[Error: execution failed]") catch null,
            .success = false,
            .duration_ns = if (timer) |*t| t.read() else 0,
        };
        return;
    };

    result.* = .{
        .response = response,
        .success = true,
        .duration_ns = if (timer) |*t| t.read() else 0,
    };
}

/// Statistics about coordinator execution.
pub const CoordinatorStats = struct {
    agent_count: usize = 0,
    result_count: usize = 0,
    success_count: usize = 0,
    avg_duration_ns: u64 = 0,

    pub fn successRate(self: CoordinatorStats) f64 {
        if (self.result_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.success_count)) /
            @as(f64, @floatFromInt(self.result_count));
    }
};

/// Check if multi-agent is enabled.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

// ============================================================================
// Tests
// ============================================================================

test "coordinator init and deinit" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Coordinator starts with no agents
    try std.testing.expectEqual(@as(usize, 0), coord.agentCount());
}

test "coordinator with config" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .vote,
        .max_agents = 10,
    });
    defer coord.deinit();

    try std.testing.expectEqual(ExecutionStrategy.parallel, coord.config.execution_strategy);
    try std.testing.expectEqual(AggregationStrategy.vote, coord.config.aggregation_strategy);
    try std.testing.expectEqual(@as(u32, 10), coord.config.max_agents);
}

test "coordinator with event bus" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .enable_events = true,
    });
    defer coord.deinit();

    try std.testing.expect(coord.event_bus != null);
}

test "coordinator runTask with no agents returns error" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // Running task with no agents should return NoAgents error
    const result = coord.runTask("test task");
    try std.testing.expectError(Error.NoAgents, result);
}

test "execution strategy toString" {
    try std.testing.expectEqualStrings("sequential", ExecutionStrategy.sequential.toString());
    try std.testing.expectEqualStrings("parallel", ExecutionStrategy.parallel.toString());
    try std.testing.expectEqualStrings("pipeline", ExecutionStrategy.pipeline.toString());
}

test "aggregation strategy toString" {
    try std.testing.expectEqualStrings("concatenate", AggregationStrategy.concatenate.toString());
    try std.testing.expectEqualStrings("vote", AggregationStrategy.vote.toString());
    try std.testing.expectEqualStrings("select_best", AggregationStrategy.select_best.toString());
    try std.testing.expectEqualStrings("merge", AggregationStrategy.merge.toString());
}

test "coordinator stats" {
    const stats = CoordinatorStats{
        .agent_count = 5,
        .result_count = 10,
        .success_count = 8,
        .avg_duration_ns = 1000,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 0.8), stats.successRate(), 0.001);
}

test "coordinator config defaults" {
    const config = CoordinatorConfig.defaults();
    try std.testing.expectEqual(ExecutionStrategy.sequential, config.execution_strategy);
    try std.testing.expectEqual(AggregationStrategy.concatenate, config.aggregation_strategy);
    try std.testing.expectEqual(@as(u32, 100), config.max_agents);
    try std.testing.expect(!config.enable_events);
}

// Bring in submodule tests
test {
    _ = aggregation;
    _ = messaging;
}
