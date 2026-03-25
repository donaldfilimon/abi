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

                var prepared = prepared_step.PreparedStep.prepare(self, &step);
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

                if (outcome.escalated) {
                    return session.fail(self, "step escalated");
                }
            }
        }

        return run_finalize.finalizeRun(self, def, &session);
    }
};
