//! Supervisor Pattern for Multi-Agent Systems
//!
//! Monitors agent execution, detects failures, and applies recovery strategies.
//! Inspired by Erlang/OTP supervisor trees but adapted for LLM agent workflows.
//!
//! Features:
//! - **Restart strategies**: one_for_one, one_for_all, rest_for_one
//! - **Failure tracking**: Per-agent failure counts with time windows
//! - **Escalation**: After max retries, escalate to parent or fail workflow
//! - **Health checks**: Periodic liveness/readiness probes
//! - **Event integration**: Publishes supervisor events to the messaging EventBus

const std = @import("std");
const messaging = @import("messaging.zig");
const time = @import("../../../services/shared/time.zig");

// ============================================================================
// Types
// ============================================================================

/// Strategy for handling agent failures.
pub const RestartStrategy = enum {
    /// Restart only the failed agent.
    one_for_one,
    /// Restart all agents if any one fails.
    one_for_all,
    /// Restart the failed agent and all agents registered after it.
    rest_for_one,
};

/// Action the supervisor takes in response to a failure.
pub const SupervisorAction = enum {
    /// Retry the same agent on the same task.
    retry,
    /// Reassign the task to a different agent.
    reassign,
    /// Skip the failed task and continue.
    skip,
    /// Escalate — halt the workflow and report failure.
    escalate,
    /// Restart the agent with fresh state.
    restart,
};

/// Severity of a failure event.
pub const FailureSeverity = enum {
    /// Transient error (timeout, rate limit) — likely recoverable.
    transient,
    /// Persistent error (bad config, auth failure) — retry unlikely to help.
    persistent,
    /// Fatal error (OOM, panic) — agent is dead.
    fatal,

    /// Suggest the default action for this severity level.
    pub fn suggestedAction(self: FailureSeverity) SupervisorAction {
        return switch (self) {
            .transient => .retry,
            .persistent => .reassign,
            .fatal => .escalate,
        };
    }
};

/// A failure event reported to the supervisor.
pub const FailureEvent = struct {
    /// ID of the agent that failed.
    agent_id: []const u8,
    /// ID of the task/step that failed.
    task_id: []const u8,
    /// Severity classification.
    severity: FailureSeverity,
    /// Human-readable error description.
    error_message: []const u8,
    /// Timestamp of the failure (monotonic nanoseconds).
    timestamp_ns: u64,
    /// Number of times this agent has failed on this task.
    attempt_number: u32,
};

/// Scope of agents affected by a supervisor decision.
pub const AffectedScope = enum {
    /// Only the failed agent is affected (one_for_one).
    single,
    /// All agents in the group are affected (one_for_all).
    all,
    /// The failed agent and all agents registered after it (rest_for_one).
    dependents,
};

/// Decision made by the supervisor in response to a failure.
pub const SupervisorDecision = struct {
    action: SupervisorAction,
    /// If reassign, the suggested replacement agent ID (empty = auto-select).
    replacement_agent: []const u8,
    /// Reason for the decision.
    reason: []const u8,
    /// Delay before executing the action (nanoseconds).
    delay_ns: u64,
    /// Scope of agents affected by this decision.
    affected_scope: AffectedScope = .single,
};

/// Configuration for the supervisor.
pub const SupervisorConfig = struct {
    /// How to handle failures across the agent group.
    restart_strategy: RestartStrategy = .one_for_one,
    /// Maximum retry attempts per agent per task before escalating.
    max_retries: u32 = 3,
    /// Time window for failure counting (nanoseconds). Failures older than this are forgotten.
    failure_window_ns: u64 = 60 * std.time.ns_per_s,
    /// Maximum total failures across all agents before halting the workflow.
    max_total_failures: u32 = 10,
    /// Base delay between retries (nanoseconds). Doubles on each retry (exponential backoff).
    base_retry_delay_ns: u64 = 1 * std.time.ns_per_s,
    /// Maximum retry delay cap (nanoseconds).
    max_retry_delay_ns: u64 = 30 * std.time.ns_per_s,
    /// Whether to publish events to the messaging EventBus.
    publish_events: bool = true,
};

// ============================================================================
// Supervisor
// ============================================================================

/// Monitors agents and applies recovery strategies on failure.
pub const Supervisor = struct {
    allocator: std.mem.Allocator,
    config: SupervisorConfig,
    /// Per-agent failure tracking: agent_id -> list of failure timestamps.
    agent_failures: std.StringHashMapUnmanaged(FailureHistory),
    /// Per-task attempt counters: "agent_id:task_id" -> attempt count.
    attempt_counts: std.StringHashMapUnmanaged(u32),
    /// Total failure count across all agents in the current window.
    total_failures: u32,
    /// Decision log for auditing.
    decision_log: std.ArrayListUnmanaged(LogEntry),
    /// Optional event bus for publishing supervisor events.
    event_bus: ?*messaging.EventBus,

    const FailureHistory = struct {
        timestamps: std.ArrayListUnmanaged(u64),

        fn init() FailureHistory {
            return .{ .timestamps = .empty };
        }

        fn deinit(self: *FailureHistory, allocator: std.mem.Allocator) void {
            self.timestamps.deinit(allocator);
        }

        /// Count failures within a time window.
        fn countRecent(self: *const FailureHistory, window_ns: u64) u32 {
            const now = time.timestampNs();
            var cnt: u32 = 0;
            for (self.timestamps.items) |ts| {
                if (now >= ts and now - ts <= window_ns) {
                    cnt += 1;
                }
            }
            return cnt;
        }
    };

    const LogEntry = struct {
        agent_id: []const u8,
        task_id: []const u8,
        action: SupervisorAction,
        reason: []const u8,
        timestamp_ns: u64,
    };

    pub fn init(allocator: std.mem.Allocator, config: SupervisorConfig) Supervisor {
        return .{
            .allocator = allocator,
            .config = config,
            .agent_failures = .empty,
            .attempt_counts = .empty,
            .total_failures = 0,
            .decision_log = .empty,
            .event_bus = null,
        };
    }

    pub fn deinit(self: *Supervisor) void {
        var iter = self.agent_failures.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.agent_failures.deinit(self.allocator);

        // Free owned compound keys
        var key_iter = self.attempt_counts.iterator();
        while (key_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.attempt_counts.deinit(self.allocator);
        self.decision_log.deinit(self.allocator);
    }

    /// Attach an event bus for publishing supervisor events.
    pub fn attachEventBus(self: *Supervisor, bus: *messaging.EventBus) void {
        self.event_bus = bus;
    }

    /// Report a failure and get a decision on what to do.
    pub fn reportFailure(self: *Supervisor, event: FailureEvent) !SupervisorDecision {
        self.total_failures += 1;

        // Track per-agent failure
        const gop = try self.agent_failures.getOrPut(self.allocator, event.agent_id);
        if (!gop.found_existing) {
            gop.value_ptr.* = FailureHistory.init();
        }
        try gop.value_ptr.timestamps.append(self.allocator, event.timestamp_ns);

        // Track per-task attempt count — heap-allocate the compound key
        const compound_key = try std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ event.agent_id, event.task_id });
        const attempt_gop = try self.attempt_counts.getOrPut(self.allocator, compound_key);
        if (!attempt_gop.found_existing) {
            attempt_gop.value_ptr.* = 0;
        } else {
            // Key already existed, free the duplicate we just allocated
            self.allocator.free(compound_key);
        }
        attempt_gop.value_ptr.* += 1;
        const attempts = attempt_gop.value_ptr.*;

        // Make decision
        const decision = self.decide(event, attempts);

        // Log decision
        try self.decision_log.append(self.allocator, .{
            .agent_id = event.agent_id,
            .task_id = event.task_id,
            .action = decision.action,
            .reason = decision.reason,
            .timestamp_ns = time.timestampNs(),
        });

        // Publish event if bus attached
        if (self.event_bus) |bus| {
            if (self.config.publish_events) {
                bus.publish(.{
                    .event_type = .task_failed,
                    .task_id = 0,
                    .success = false,
                    .detail = event.error_message,
                });
            }
        }

        return decision;
    }

    /// Check if the workflow should be halted due to too many failures.
    pub fn shouldHalt(self: *const Supervisor) bool {
        return self.total_failures >= self.config.max_total_failures;
    }

    /// Reset failure counts for an agent (e.g., after successful task completion).
    pub fn resetAgent(self: *Supervisor, agent_id: []const u8) void {
        if (self.agent_failures.getPtr(agent_id)) |history| {
            history.timestamps.clearRetainingCapacity();
        }
    }

    /// Get the number of recent failures for an agent.
    pub fn recentFailures(self: *const Supervisor, agent_id: []const u8) u32 {
        const history = self.agent_failures.get(agent_id) orelse return 0;
        return history.countRecent(self.config.failure_window_ns);
    }

    /// Calculate retry delay with exponential backoff.
    pub fn retryDelay(self: *const Supervisor, attempt: u32) u64 {
        if (attempt == 0) return self.config.base_retry_delay_ns;
        const shift: u6 = @intCast(@min(attempt - 1, 30));
        const multiplier: u64 = @as(u64, 1) << shift;
        const delay = self.config.base_retry_delay_ns *| multiplier;
        return @min(delay, self.config.max_retry_delay_ns);
    }

    /// Number of decisions made so far.
    pub fn decisionCount(self: *const Supervisor) usize {
        return self.decision_log.items.len;
    }

    /// Total failures observed.
    pub fn totalFailures(self: *const Supervisor) u32 {
        return self.total_failures;
    }

    fn decide(self: *const Supervisor, event: FailureEvent, attempts: u32) SupervisorDecision {
        // Determine scope from restart strategy
        const scope: AffectedScope = switch (self.config.restart_strategy) {
            .one_for_one => .single,
            .one_for_all => .all,
            .rest_for_one => .dependents,
        };

        if (self.total_failures >= self.config.max_total_failures) {
            return .{
                .action = .escalate,
                .replacement_agent = "",
                .reason = "max total failures exceeded",
                .delay_ns = 0,
                .affected_scope = scope,
            };
        }

        if (attempts >= self.config.max_retries) {
            return switch (event.severity) {
                .transient => .{
                    .action = .reassign,
                    .replacement_agent = "",
                    .reason = "max retries exhausted for transient error",
                    .delay_ns = 0,
                    .affected_scope = scope,
                },
                .persistent, .fatal => .{
                    .action = .escalate,
                    .replacement_agent = "",
                    .reason = "max retries exhausted for non-transient error",
                    .delay_ns = 0,
                    .affected_scope = scope,
                },
            };
        }

        const action = event.severity.suggestedAction();
        const delay = if (action == .retry) self.retryDelay(attempts) else 0;

        // For restart actions with one_for_all/rest_for_one, use .restart to
        // signal the broader scope; for one_for_one keep the suggested action.
        const final_action = if (scope != .single and action == .retry)
            SupervisorAction.restart
        else
            action;

        return .{
            .action = final_action,
            .replacement_agent = "",
            .reason = switch (event.severity) {
                .transient => if (scope != .single)
                    "transient error, restarting affected agents"
                else
                    "transient error, retrying with backoff",
                .persistent => "persistent error, reassigning to different agent",
                .fatal => "fatal error, escalating",
            },
            .delay_ns = delay,
            .affected_scope = scope,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "supervisor transient failure retry" {
    var sup = Supervisor.init(std.testing.allocator, .{});
    defer sup.deinit();

    const decision = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "timeout",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(SupervisorAction.retry, decision.action);
    try std.testing.expect(decision.delay_ns > 0);
}

test "supervisor persistent failure reassign" {
    var sup = Supervisor.init(std.testing.allocator, .{});
    defer sup.deinit();

    const decision = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .persistent,
        .error_message = "auth failed",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(SupervisorAction.reassign, decision.action);
}

test "supervisor fatal failure escalate" {
    var sup = Supervisor.init(std.testing.allocator, .{});
    defer sup.deinit();

    const decision = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .fatal,
        .error_message = "out of memory",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(SupervisorAction.escalate, decision.action);
}

test "supervisor max retries exhausted" {
    var sup = Supervisor.init(std.testing.allocator, .{ .max_retries = 2 });
    defer sup.deinit();

    const now = time.timestampNs();

    const d1 = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "timeout",
        .timestamp_ns = now,
        .attempt_number = 1,
    });
    try std.testing.expectEqual(SupervisorAction.retry, d1.action);

    const d2 = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "timeout again",
        .timestamp_ns = now,
        .attempt_number = 2,
    });
    try std.testing.expectEqual(SupervisorAction.reassign, d2.action);
}

test "supervisor global halt" {
    var sup = Supervisor.init(std.testing.allocator, .{ .max_total_failures = 2 });
    defer sup.deinit();

    const now = time.timestampNs();

    _ = try sup.reportFailure(.{
        .agent_id = "a1",
        .task_id = "t1",
        .severity = .transient,
        .error_message = "err",
        .timestamp_ns = now,
        .attempt_number = 1,
    });
    try std.testing.expect(!sup.shouldHalt());

    _ = try sup.reportFailure(.{
        .agent_id = "a2",
        .task_id = "t2",
        .severity = .transient,
        .error_message = "err",
        .timestamp_ns = now,
        .attempt_number = 1,
    });
    try std.testing.expect(sup.shouldHalt());
}

test "supervisor exponential backoff" {
    const sup = Supervisor.init(std.testing.allocator, .{
        .base_retry_delay_ns = 1_000_000_000,
        .max_retry_delay_ns = 16_000_000_000,
    });

    try std.testing.expectEqual(@as(u64, 1_000_000_000), sup.retryDelay(0));
    try std.testing.expectEqual(@as(u64, 1_000_000_000), sup.retryDelay(1));
    try std.testing.expectEqual(@as(u64, 2_000_000_000), sup.retryDelay(2));
    try std.testing.expectEqual(@as(u64, 4_000_000_000), sup.retryDelay(3));
    try std.testing.expectEqual(@as(u64, 8_000_000_000), sup.retryDelay(4));
    try std.testing.expectEqual(@as(u64, 16_000_000_000), sup.retryDelay(5));
    try std.testing.expectEqual(@as(u64, 16_000_000_000), sup.retryDelay(6));
}

test "supervisor reset agent" {
    var sup = Supervisor.init(std.testing.allocator, .{});
    defer sup.deinit();

    _ = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "timeout",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(@as(u32, 1), sup.recentFailures("agent-1"));

    sup.resetAgent("agent-1");
    try std.testing.expectEqual(@as(u32, 0), sup.recentFailures("agent-1"));
}

test "supervisor decision logging" {
    var sup = Supervisor.init(std.testing.allocator, .{});
    defer sup.deinit();

    _ = try sup.reportFailure(.{
        .agent_id = "agent-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "err",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), sup.decisionCount());
    try std.testing.expectEqual(@as(u32, 1), sup.totalFailures());
}

test {
    std.testing.refAllDecls(@This());
}
