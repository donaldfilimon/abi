//! Workflow Runner — DAG Execution Engine
//!
//! Connects all multi-agent sub-modules into a cohesive execution engine.
//! Runs workflow DAGs by:
//!   1. Validating the workflow graph
//!   2. Computing topological layers
//!   3. Assigning profiles to steps via capability matching
//!   4. Executing steps layer-by-layer with blackboard I/O
//!   5. Handling failures via the supervisor pattern
//!   6. Publishing lifecycle events
//!
//! ## Example
//!
//! ```zig
//! var runner = WorkflowRunner.init(allocator, .{});
//! defer runner.deinit();
//!
//! try runner.registerAgent("coder", &coder_agent);
//! const result = try runner.run(&workflow.preset_workflows.code_review);
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const workflow_mod = @import("workflow.zig");
const blackboard_mod = @import("blackboard.zig");
const roles = @import("roles.zig");
const supervisor_mod = @import("supervisor.zig");
const messaging = @import("messaging.zig");
const protocol = @import("protocol.zig");
const types = @import("types.zig");
const prepared_step = @import("workflow_runner/prepare.zig");
const run_session = @import("workflow_runner/session.zig");
const step_commit = @import("workflow_runner/commit.zig");
const step_execute = @import("workflow_runner/execute.zig");
const run_finalize = @import("workflow_runner/finalize.zig");
const build_options = @import("build_options");
const agents_mod = @import("../agents/mod.zig");
const time = @import("../../../foundation/mod.zig").time;
const reasoning = @import("../reasoning/engine.zig");
const training = if (build_options.feat_training)
    @import("../training/mod.zig")
else
    @import("../training/stub.zig");
const SelfLearningSystem = training.SelfLearningSystem;

pub const RunnerConfig = types.RunnerConfig;
pub const WorkflowResult = types.WorkflowResult;
pub const StepResult = types.StepResult;
pub const WorkflowStats = types.WorkflowStats;
pub const RunError = types.RunError;

// ============================================================================
// WorkflowRunner
// ============================================================================

pub const WorkflowRunner = struct {
    allocator: Allocator,
    config: RunnerConfig,
    blackboard: blackboard_mod.Blackboard,
    profile_registry: roles.ProfileRegistry,
    supervisor: supervisor_mod.Supervisor,
    event_bus: messaging.EventBus,
    conversation_manager: protocol.ConversationManager,
    agent_map: std.StringHashMapUnmanaged(*agents_mod.Agent),
    learning_system: ?*SelfLearningSystem = null,

    pub fn init(allocator: Allocator, config: RunnerConfig) WorkflowRunner {
        return .{
            .allocator = allocator,
            .config = config,
            .blackboard = blackboard_mod.Blackboard.init(allocator, config.max_history),
            .profile_registry = roles.ProfileRegistry.init(allocator),
            .supervisor = supervisor_mod.Supervisor.init(allocator, .{
                .restart_strategy = config.restart_strategy,
                .max_retries = config.max_retries,
            }),
            .event_bus = messaging.EventBus.init(allocator),
            .conversation_manager = protocol.ConversationManager.init(allocator, 20),
            .agent_map = .{},
        };
    }

    pub fn deinit(self: *WorkflowRunner) void {
        self.agent_map.deinit(self.allocator);
        self.conversation_manager.deinit();
        self.event_bus.deinit();
        self.supervisor.deinit();
        self.profile_registry.deinit();
        self.blackboard.deinit();
    }

    /// Register a named agent for use in workflows.
    pub fn registerAgent(self: *WorkflowRunner, name: []const u8, agent: *agents_mod.Agent) !void {
        try self.agent_map.put(self.allocator, name, agent);
    }

    /// Connect a self-learning system to receive workflow outcome feedback.
    pub fn setLearningSystem(self: *WorkflowRunner, sys: *SelfLearningSystem) void {
        self.learning_system = sys;
    }

    /// Execute a complete workflow DAG.
    pub fn run(self: *WorkflowRunner, def: *const workflow_mod.WorkflowDef) !WorkflowResult {
        var session = try run_session.RunSession.bootstrap(self, def);
        defer session.deinit();

        for (session.layers) |layer| {
            for (layer) |step_id| {
                const step = def.getStep(step_id) orelse continue;

                const current_status = session.tracker.getStepStatus(step_id) orelse continue;
                if (current_status == .skipped or current_status == .failed or current_status == .completed) {
                    if (current_status == .skipped) {
                        session.stats.skipped_steps += 1;
                    }
                    continue;
                }

                session.tracker.markRunning(step_id);
                self.event_bus.publish(.{
                    .event_type = .agent_started,
                    .task_id = session.task_id,
                });

                const prepared = prepared_step.PreparedStep.prepare(self, &step);
                defer prepared.deinit(self.allocator);

                const outcome = step_execute.executeStepAttempts(
                    self,
                    &session.tracker,
                    step_id,
                    prepared.prompt,
                    prepared.profile_name,
                );

                step_commit.commitStepOutcome(
                    self,
                    &session,
                    step_id,
                    &step,
                    &prepared,
                    outcome,
                );

                // Select agent: try profile name first, then first available
                const agent = self.selectAgent(prepared.profile_name);

                var step_timer = time.Timer.start() catch null;
                var attempts: u32 = 0;
                var step_output: ?[]const u8 = null;
                var step_status: workflow_mod.StepStatus = .failed;
                var escalated = false;

                // Execution loop with retry
                while (attempts <= self.config.max_retries) : (attempts += 1) {
                    if (agent) |ag| {
                        const result = ag.process(prepared.prompt, self.allocator) catch |err| {
                            // Handle failure
                            const action = self.handleFailure(step_id, err, &session.tracker);
                            switch (action) {
                                .retry => {
                                    session.stats.total_retries += 1;
                                    continue;
                                },
                                .reassign => {
                                    session.stats.total_retries += 1;
                                    continue;
                                },
                                .skip => {
                                    step_status = .failed;
                                    break;
                                },
                                .escalate => {
                                    escalated = true;
                                    step_status = .failed;
                                    break;
                                },
                                .restart => {
                                    session.stats.total_retries += 1;
                                    continue;
                                },
                            }
                        };

                        // Success
                        step_output = result;

                        // --- Reactive Orchestration: Evaluate Confidence ---
                        var chain = reasoning.ReasoningChain.init(self.allocator, prepared.prompt);
                        defer chain.deinit();

                        // Mock evaluation heuristic
                        const output_str = step_output orelse "";
                        var conf_score: f32 = 0.8; // Default high confidence
                        if (std.mem.indexOf(u8, output_str, "error") != null or
                            std.mem.indexOf(u8, output_str, "unknown") != null or
                            std.mem.indexOf(u8, output_str, "research needed") != null)
                        {
                            conf_score = 0.3;
                        }

                        chain.addStep(.assessment, "Evaluating step output confidence", .{
                            .level = reasoning.ConfidenceLevel.fromScore(conf_score),
                            .score = conf_score,
                            .reasoning = "Mocked evaluation based on output keywords",
                        }) catch {};

                        chain.finalize() catch {};
                        const final_conf = chain.getOverallConfidence();

                        if (final_conf.score < 0.5) {
                            if (attempts < self.config.max_retries) {
                                chain.addStep(.research, "Confidence below threshold, initiating research", .{
                                    .level = reasoning.ConfidenceLevel.fromScore(0.8),
                                    .score = 0.8,
                                    .reasoning = "Fallback triggered",
                                }) catch {};

                                if (step_output) |o| self.allocator.free(o);
                                step_output = null;

                                const action = self.handleFailure(step_id, RunError.ExecutionFailed, &session.tracker);
                                switch (action) {
                                    .retry, .reassign, .restart => {
                                        session.stats.total_retries += 1;
                                        continue;
                                    },
                                    .skip => {
                                        step_status = .failed;
                                        break;
                                    },
                                    .escalate => {
                                        escalated = true;
                                        step_status = .failed;
                                        break;
                                    },
                                }
                            } else {
                                step_status = .failed;
                                break;
                            }
                        }

                        step_status = .completed;
                        break;
                    } else {
                        // No agent available
                        step_status = .failed;
                        break;
                    }
                }

                // Duration
                const duration_ms: u64 = if (step_timer) |*t|
                    t.read() / std.time.ns_per_ms
                else
                    0;

                // Record result
                if (step_status == .completed) {
                    // Store output in blackboard
                    if (step_output) |output| {
                        self.blackboard.put(step.output_key, output, prepared.profile_name) catch |err| {
                            std.log.warn("Failed to update blackboard: {t}", .{err});
                        };
                    }

                    const step_result_entry = workflow_mod.StepResult{
                        .step_id = step_id,
                        .status = .completed,
                        .output = step_output orelse "",
                        .error_message = "",
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                        .assigned_profile = prepared.profile_name,
                    };
                    session.tracker.markCompleted(step_id, step_result_entry) catch |err| {
                        std.log.warn("Failed to mark step completed: {t}", .{err});
                    };
                    session.stats.completed_steps += 1;

                    // Store in our step_results map (dupe the output for ownership)
                    const output_copy = if (step_output) |o|
                        self.allocator.dupe(u8, o) catch null
                    else
                        null;
                    session.step_results.put(self.allocator, step_id, .{
                        .step_id = step_id,
                        .output = output_copy,
                        .status = .completed,
                        .assigned_profile = prepared.profile_name,
                        .attempts = attempts,
                        .duration_ms = duration_ms,
                    }) catch |err| {
                        std.log.warn("Failed to generate step log: {t}", .{err});
                    };

                    // Free the process() result (we duped it for step_results)
                    if (step_output) |o| self.allocator.free(o);

                    self.event_bus.publish(.{
                        .event_type = .agent_finished,
                        .task_id = session.task_id,
                        .success = true,
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                    });

                    // Reset supervisor for this agent on success
                    self.supervisor.resetAgent(prepared.profile_name);
                } else {
                    const step_result_entry = workflow_mod.StepResult{
                        .step_id = step_id,
                        .status = .failed,
                        .output = "",
                        .error_message = "step execution failed",
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                        .assigned_profile = prepared.profile_name,
                    };
                    session.tracker.markFailed(step_id, step_result_entry) catch |err| {
                        std.log.warn("Failed to mark step failed: {t}", .{err});
                    };
                    session.stats.failed_steps += 1;

                    session.step_results.put(self.allocator, step_id, .{
                        .step_id = step_id,
                        .output = null,
                        .status = .failed,
                        .assigned_profile = prepared.profile_name,
                        .attempts = attempts,
                        .duration_ms = duration_ms,
                    }) catch |err| {
                        std.log.warn("Failed to generate step log: {t}", .{err});
                    };

                    // Free any partial output on failure
                    if (step_output) |o| self.allocator.free(o);

                    self.event_bus.publish(.{
                        .event_type = .agent_finished,
                        .task_id = session.task_id,
                        .success = false,
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                    });

                    if (escalated) {
                        // Stop workflow on escalation
                        const total_ms: u64 = if (session.overall_timer) |*t|
                            t.read() / std.time.ns_per_ms
                        else
                            0;
                        session.stats.total_duration_ms = total_ms;

                        self.event_bus.taskFailed(session.task_id, "step escalated");

                        return WorkflowResult{
                            .success = false,
                            .step_results = session.step_results,
                            .final_output = null,
                            .stats = session.stats,
                            .allocator = self.allocator,
                        };
                    }
                }
            }
        }

        return run_finalize.finalizeRun(self, def, &session);
    }
};

test {
    std.testing.refAllDecls(@This());
}
