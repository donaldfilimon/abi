//! Workflow Runner — DAG Execution Engine
//!
//! Connects all multi-agent sub-modules into a cohesive execution engine.
//! Runs workflow DAGs by:
//!   1. Validating the workflow graph
//!   2. Computing topological layers
//!   3. Assigning personas to steps via capability matching
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
const agents_mod = @import("../agents/mod.zig");
const time = @import("../../../services/shared/time.zig");

// ============================================================================
// WorkflowRunner
// ============================================================================

pub const WorkflowRunner = struct {
    allocator: Allocator,
    config: RunnerConfig,
    blackboard: blackboard_mod.Blackboard,
    persona_registry: roles.PersonaRegistry,
    supervisor: supervisor_mod.Supervisor,
    event_bus: messaging.EventBus,
    conversation_manager: protocol.ConversationManager,
    agent_map: std.StringHashMapUnmanaged(*agents_mod.Agent),

    pub const RunnerConfig = struct {
        max_retries: u32 = 3,
        step_timeout_ms: u64 = 30_000,
        enable_negotiation: bool = false,
        restart_strategy: supervisor_mod.RestartStrategy = .one_for_one,
        max_history: u32 = 100,
    };

    pub const WorkflowResult = struct {
        success: bool,
        step_results: std.StringHashMapUnmanaged(StepResult),
        final_output: ?[]const u8,
        stats: WorkflowStats,
        allocator: Allocator,

        pub fn deinit(self: *WorkflowResult) void {
            // Free owned step result output strings
            var iter = self.step_results.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.output) |o| {
                    self.allocator.free(o);
                }
            }
            self.step_results.deinit(self.allocator);
            if (self.final_output) |fo| {
                self.allocator.free(fo);
            }
        }
    };

    pub const StepResult = struct {
        step_id: []const u8,
        output: ?[]const u8,
        status: workflow_mod.StepStatus,
        assigned_persona: ?[]const u8,
        attempts: u32,
        duration_ms: u64,
    };

    pub const WorkflowStats = struct {
        total_steps: u32 = 0,
        completed_steps: u32 = 0,
        failed_steps: u32 = 0,
        skipped_steps: u32 = 0,
        total_retries: u32 = 0,
        total_duration_ms: u64 = 0,
    };

    pub const RunError = error{
        InvalidWorkflow,
        NoAgents,
        ExecutionFailed,
        Escalated,
        OutOfMemory,
    };

    pub fn init(allocator: Allocator, config: RunnerConfig) WorkflowRunner {
        return .{
            .allocator = allocator,
            .config = config,
            .blackboard = blackboard_mod.Blackboard.init(allocator, config.max_history),
            .persona_registry = roles.PersonaRegistry.init(allocator),
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
        self.persona_registry.deinit();
        self.blackboard.deinit();
    }

    /// Register a named agent for use in workflows.
    pub fn registerAgent(self: *WorkflowRunner, name: []const u8, agent: *agents_mod.Agent) !void {
        try self.agent_map.put(self.allocator, name, agent);
    }

    /// Execute a complete workflow DAG.
    pub fn run(self: *WorkflowRunner, def: *const workflow_mod.WorkflowDef) !WorkflowResult {
        // 1. Validate
        const validation = def.validate();
        if (!validation.valid) {
            return RunError.InvalidWorkflow;
        }

        // 2. Check we have at least one agent
        if (self.agent_map.count() == 0) {
            return RunError.NoAgents;
        }

        // 3. Compute execution layers
        const layers = def.computeLayers(self.allocator) catch return RunError.OutOfMemory;
        defer {
            for (layers) |layer| self.allocator.free(layer);
            self.allocator.free(layers);
        }

        // 4. Create execution tracker
        var tracker = workflow_mod.ExecutionTracker.init(self.allocator, def.*) catch
            return RunError.OutOfMemory;
        defer tracker.deinit();

        // 5. Load persona presets
        self.persona_registry.loadPresets() catch {};

        // 6. Publish task_started
        const task_id = messaging.taskId(def.id);
        self.event_bus.taskStarted(task_id);

        var overall_timer = time.Timer.start() catch null;

        // 7. Track step results and stats
        var step_results = std.StringHashMapUnmanaged(StepResult).empty;
        errdefer {
            var iter = step_results.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.output) |o| {
                    self.allocator.free(o);
                }
            }
            step_results.deinit(self.allocator);
        }
        var stats = WorkflowStats{};
        stats.total_steps = @intCast(def.steps.len);

        // 8. Execute layer by layer
        for (layers) |layer| {
            for (layer) |step_id| {
                const step = def.getStep(step_id) orelse continue;

                // Check if step is still eligible (not skipped by failure propagation)
                const current_status = tracker.getStepStatus(step_id) orelse continue;
                if (current_status == .skipped or current_status == .failed or current_status == .completed) {
                    if (current_status == .skipped) stats.skipped_steps += 1;
                    continue;
                }

                tracker.markRunning(step_id);
                self.event_bus.publish(.{
                    .event_type = .agent_started,
                    .task_id = task_id,
                });

                // Assign persona
                const persona = self.assignPersona(&step);
                const persona_name: []const u8 = if (persona) |p| p.id else "default";

                // Gather inputs from blackboard
                const inputs = self.gatherInputs(&step) catch "";

                // Build prompt
                const prompt = self.buildPrompt(step.prompt_template, inputs) catch step.prompt_template;
                const prompt_owned = !std.mem.eql(u8, prompt, step.prompt_template);

                defer {
                    if (inputs.len > 0) self.allocator.free(inputs);
                    if (prompt_owned) self.allocator.free(prompt);
                }

                // Select agent: try persona name first, then first available
                const agent = self.selectAgent(persona_name);

                var step_timer = time.Timer.start() catch null;
                var attempts: u32 = 0;
                var step_output: ?[]const u8 = null;
                var step_status: workflow_mod.StepStatus = .failed;
                var escalated = false;

                // Execution loop with retry
                while (attempts <= self.config.max_retries) : (attempts += 1) {
                    if (agent) |ag| {
                        const result = ag.process(prompt, self.allocator) catch |err| {
                            // Handle failure
                            const action = self.handleFailure(step_id, err, &tracker);
                            switch (action) {
                                .retry => {
                                    stats.total_retries += 1;
                                    continue;
                                },
                                .reassign => {
                                    stats.total_retries += 1;
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
                                    stats.total_retries += 1;
                                    continue;
                                },
                            }
                        };

                        // Success
                        step_output = result;
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
                        self.blackboard.put(step.output_key, output, persona_name) catch {};
                    }

                    const step_result_entry = workflow_mod.StepResult{
                        .step_id = step_id,
                        .status = .completed,
                        .output = step_output orelse "",
                        .error_message = "",
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                        .assigned_persona = persona_name,
                    };
                    tracker.markCompleted(step_id, step_result_entry) catch {};
                    stats.completed_steps += 1;

                    // Store in our step_results map (dupe the output for ownership)
                    const output_copy = if (step_output) |o|
                        self.allocator.dupe(u8, o) catch null
                    else
                        null;
                    step_results.put(self.allocator, step_id, .{
                        .step_id = step_id,
                        .output = output_copy,
                        .status = .completed,
                        .assigned_persona = persona_name,
                        .attempts = attempts,
                        .duration_ms = duration_ms,
                    }) catch {};

                    // Free the process() result (we duped it for step_results)
                    if (step_output) |o| self.allocator.free(o);

                    self.event_bus.publish(.{
                        .event_type = .agent_finished,
                        .task_id = task_id,
                        .success = true,
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                    });

                    // Reset supervisor for this agent on success
                    self.supervisor.resetAgent(persona_name);
                } else {
                    const step_result_entry = workflow_mod.StepResult{
                        .step_id = step_id,
                        .status = .failed,
                        .output = "",
                        .error_message = "step execution failed",
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                        .assigned_persona = persona_name,
                    };
                    tracker.markFailed(step_id, step_result_entry) catch {};
                    stats.failed_steps += 1;

                    step_results.put(self.allocator, step_id, .{
                        .step_id = step_id,
                        .output = null,
                        .status = .failed,
                        .assigned_persona = persona_name,
                        .attempts = attempts,
                        .duration_ms = duration_ms,
                    }) catch {};

                    // Free any partial output on failure
                    if (step_output) |o| self.allocator.free(o);

                    self.event_bus.publish(.{
                        .event_type = .agent_finished,
                        .task_id = task_id,
                        .success = false,
                        .duration_ns = duration_ms * std.time.ns_per_ms,
                    });

                    if (escalated) {
                        // Stop workflow on escalation
                        const total_ms: u64 = if (overall_timer) |*t|
                            t.read() / std.time.ns_per_ms
                        else
                            0;
                        stats.total_duration_ms = total_ms;

                        self.event_bus.taskFailed(task_id, "step escalated");

                        return WorkflowResult{
                            .success = false,
                            .step_results = step_results,
                            .final_output = null,
                            .stats = stats,
                            .allocator = self.allocator,
                        };
                    }
                }
            }
        }

        // 9. Compute final stats
        const total_ms: u64 = if (overall_timer) |*t|
            t.read() / std.time.ns_per_ms
        else
            0;
        stats.total_duration_ms = total_ms;

        // Count skipped steps that were propagated by failure
        const prog = tracker.progress();
        stats.skipped_steps = @intCast(prog.total - prog.completed - prog.failed - prog.running);

        // 10. Build final output from the last step's blackboard entry
        var final_output: ?[]const u8 = null;
        if (def.steps.len > 0) {
            // Try to get output from the last step in the workflow definition
            const last_step = def.steps[def.steps.len - 1];
            if (self.blackboard.get(last_step.output_key)) |entry| {
                final_output = self.allocator.dupe(u8, entry.value) catch null;
            }
        }

        // 11. Determine overall success
        const success = stats.failed_steps == 0 and stats.completed_steps > 0;

        // 12. Publish completion
        if (success) {
            self.event_bus.taskCompleted(task_id, total_ms * std.time.ns_per_ms);
        } else {
            self.event_bus.taskFailed(task_id, "workflow had failures");
        }

        return WorkflowResult{
            .success = success,
            .step_results = step_results,
            .final_output = final_output,
            .stats = stats,
            .allocator = self.allocator,
        };
    }

    // -- Internal methods --

    fn assignPersona(self: *WorkflowRunner, step: *const workflow_mod.Step) ?roles.Persona {
        // 1. If step has an explicit persona assignment, use it
        if (step.assigned_persona.len > 0) {
            return self.persona_registry.get(step.assigned_persona);
        }
        // 2. If step has required capabilities, find the best match
        if (step.required_capabilities.len > 0) {
            return self.persona_registry.findBestMatch(step.required_capabilities);
        }
        // 3. No constraint — use default
        return null;
    }

    fn gatherInputs(self: *WorkflowRunner, step: *const workflow_mod.Step) ![]const u8 {
        if (step.input_keys.len == 0) return "";

        var parts = std.ArrayListUnmanaged(u8).empty;
        errdefer parts.deinit(self.allocator);

        for (step.input_keys, 0..) |key, i| {
            if (self.blackboard.get(key)) |entry| {
                if (i > 0) {
                    try parts.appendSlice(self.allocator, "\n");
                }
                try parts.appendSlice(self.allocator, key);
                try parts.appendSlice(self.allocator, ": ");
                try parts.appendSlice(self.allocator, entry.value);
            }
        }

        if (parts.items.len == 0) return "";

        return parts.toOwnedSlice(self.allocator);
    }

    fn buildPrompt(self: *WorkflowRunner, template: []const u8, inputs: []const u8) ![]const u8 {
        if (inputs.len == 0) return template;

        // Replace {input} and {context} with the gathered inputs
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < template.len) {
            if (i + 7 <= template.len and std.mem.eql(u8, template[i .. i + 7], "{input}")) {
                try result.appendSlice(self.allocator, inputs);
                i += 7;
            } else if (i + 9 <= template.len and std.mem.eql(u8, template[i .. i + 9], "{context}")) {
                try result.appendSlice(self.allocator, inputs);
                i += 9;
            } else {
                try result.append(self.allocator, template[i]);
                i += 1;
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    fn selectAgent(self: *WorkflowRunner, persona_name: []const u8) ?*agents_mod.Agent {
        // Try to find agent by persona name
        if (self.agent_map.get(persona_name)) |ag| return ag;

        // Fallback: return first available agent
        var iter = self.agent_map.iterator();
        if (iter.next()) |entry| {
            return entry.value_ptr.*;
        }
        return null;
    }

    fn handleFailure(
        self: *WorkflowRunner,
        step_id: []const u8,
        _: anyerror,
        tracker: *workflow_mod.ExecutionTracker,
    ) supervisor_mod.SupervisorAction {

        // Determine severity based on whether the step is critical
        const step = tracker.workflow.getStep(step_id);
        const severity: supervisor_mod.FailureSeverity = if (step) |s|
            (if (s.is_critical) supervisor_mod.FailureSeverity.persistent else supervisor_mod.FailureSeverity.transient)
        else
            supervisor_mod.FailureSeverity.transient;

        const decision = self.supervisor.reportFailure(.{
            .agent_id = step_id,
            .task_id = step_id,
            .severity = severity,
            .error_message = "step execution failed",
            .timestamp_ns = time.timestampNs(),
            .attempt_number = 0,
        }) catch {
            return .skip;
        };

        return decision.action;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "runner - init and deinit" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    try std.testing.expectEqual(@as(u32, 3), runner.config.max_retries);
    try std.testing.expectEqual(@as(u64, 30_000), runner.config.step_timeout_ms);
    try std.testing.expectEqual(@as(usize, 0), runner.agent_map.count());
    try std.testing.expectEqual(@as(usize, 0), runner.blackboard.count());
}

test "runner - register agent" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "echo-agent",
        .backend = .echo,
    });
    defer agent.deinit();

    try runner.registerAgent("echo-agent", &agent);
    try std.testing.expectEqual(@as(usize, 1), runner.agent_map.count());

    const retrieved = runner.agent_map.get("echo-agent");
    try std.testing.expect(retrieved != null);
}

test "runner - simple workflow execution" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "echo-agent",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();

    try runner.registerAgent("echo-agent", &agent);

    // Simple single-step workflow
    const steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Echo test",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{},
            .output_key = "step1:output",
            .prompt_template = "Hello world",
        },
    };

    const wf = workflow_mod.WorkflowDef{
        .id = "test-wf",
        .name = "Test Workflow",
        .description = "Single step test",
        .steps = &steps,
    };

    var result = try runner.run(&wf);
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 1), result.stats.total_steps);
    try std.testing.expectEqual(@as(u32, 1), result.stats.completed_steps);
    try std.testing.expectEqual(@as(u32, 0), result.stats.failed_steps);

    // Check blackboard has the output
    const bb_entry = runner.blackboard.get("step1:output");
    try std.testing.expect(bb_entry != null);
    // Echo backend returns "Echo: <input>"
    try std.testing.expect(std.mem.startsWith(u8, bb_entry.?.value, "Echo:"));

    // Check final output is set
    try std.testing.expect(result.final_output != null);
}

test "runner - multi-step DAG execution" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent1 = try agents_mod.Agent.init(allocator, .{
        .name = "agent1",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent1.deinit();

    try runner.registerAgent("agent1", &agent1);

    // Seed blackboard with initial input
    try runner.blackboard.put("task:input", "analyze this code", "system");

    // 3-step DAG: step1 -> step2 -> step3
    const steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Gather info",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{"task:input"},
            .output_key = "step1:output",
            .prompt_template = "Gather: {input}",
        },
        .{
            .id = "step2",
            .description = "Analyze",
            .depends_on = &.{"step1"},
            .required_capabilities = &.{},
            .input_keys = &.{"step1:output"},
            .output_key = "step2:output",
            .prompt_template = "Analyze: {input}",
        },
        .{
            .id = "step3",
            .description = "Report",
            .depends_on = &.{"step2"},
            .required_capabilities = &.{},
            .input_keys = &.{"step2:output"},
            .output_key = "step3:output",
            .prompt_template = "Report: {input}",
        },
    };

    const wf = workflow_mod.WorkflowDef{
        .id = "dag-test",
        .name = "DAG Test",
        .description = "Three step pipeline",
        .steps = &steps,
    };

    var result = try runner.run(&wf);
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 3), result.stats.total_steps);
    try std.testing.expectEqual(@as(u32, 3), result.stats.completed_steps);
    try std.testing.expectEqual(@as(u32, 0), result.stats.failed_steps);

    // All three steps should have output in blackboard
    try std.testing.expect(runner.blackboard.get("step1:output") != null);
    try std.testing.expect(runner.blackboard.get("step2:output") != null);
    try std.testing.expect(runner.blackboard.get("step3:output") != null);

    // Final output should be from step3
    try std.testing.expect(result.final_output != null);
}

test "runner - persona auto-assignment" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    try runner.persona_registry.loadPresets();

    // Step with code review capabilities should match code-reviewer persona
    const step_with_caps = workflow_mod.Step{
        .id = "review",
        .description = "Review code",
        .depends_on = &.{},
        .required_capabilities = &.{ .code_review, .critique },
        .input_keys = &.{},
        .output_key = "review:output",
        .prompt_template = "Review this",
    };

    const persona = runner.assignPersona(&step_with_caps);
    try std.testing.expect(persona != null);
    try std.testing.expectEqualStrings("code-reviewer", persona.?.id);

    // Step with explicit persona assignment
    const step_explicit = workflow_mod.Step{
        .id = "impl",
        .description = "Implement",
        .depends_on = &.{},
        .required_capabilities = &.{},
        .input_keys = &.{},
        .output_key = "impl:output",
        .prompt_template = "Implement this",
        .assigned_persona = "architect",
    };

    const persona2 = runner.assignPersona(&step_explicit);
    try std.testing.expect(persona2 != null);
    try std.testing.expectEqualStrings("architect", persona2.?.id);

    // Step with no capabilities — returns null
    const step_none = workflow_mod.Step{
        .id = "generic",
        .description = "Generic step",
        .depends_on = &.{},
        .required_capabilities = &.{},
        .input_keys = &.{},
        .output_key = "generic:output",
        .prompt_template = "Do something",
    };

    const persona3 = runner.assignPersona(&step_none);
    try std.testing.expect(persona3 == null);
}

test "runner - failure handling with retry" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{ .max_retries = 2 });
    defer runner.deinit();

    // Supervisor should suggest retry for transient failures
    const decision = try runner.supervisor.reportFailure(.{
        .agent_id = "step-1",
        .task_id = "step-1",
        .severity = .transient,
        .error_message = "timeout",
        .timestamp_ns = time.timestampNs(),
        .attempt_number = 1,
    });

    try std.testing.expectEqual(supervisor_mod.SupervisorAction.retry, decision.action);
}

test "runner - gather inputs from blackboard" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    // Put data into blackboard
    try runner.blackboard.put("analyze:output", "Found 3 bugs", "reviewer");
    try runner.blackboard.put("security:output", "No vulnerabilities", "security");

    const step = workflow_mod.Step{
        .id = "synthesize",
        .description = "Combine findings",
        .depends_on = &.{},
        .required_capabilities = &.{},
        .input_keys = &.{ "analyze:output", "security:output" },
        .output_key = "final:output",
        .prompt_template = "Synthesize: {input}",
    };

    const inputs = try runner.gatherInputs(&step);
    defer allocator.free(inputs);

    try std.testing.expect(inputs.len > 0);
    // Should contain both keys and values
    try std.testing.expect(std.mem.indexOf(u8, inputs, "Found 3 bugs") != null);
    try std.testing.expect(std.mem.indexOf(u8, inputs, "No vulnerabilities") != null);
}

test "runner - build prompt with template" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    // Test {input} substitution
    const prompt1 = try runner.buildPrompt("Analyze: {input}", "some data");
    defer allocator.free(prompt1);
    try std.testing.expectEqualStrings("Analyze: some data", prompt1);

    // Test {context} substitution
    const prompt2 = try runner.buildPrompt("Context: {context}", "other data");
    defer allocator.free(prompt2);
    try std.testing.expectEqualStrings("Context: other data", prompt2);

    // Test no substitution needed (empty inputs returns template directly)
    const prompt3 = try runner.buildPrompt("Plain template", "");
    // buildPrompt returns the template directly when inputs is empty, so no free needed
    try std.testing.expectEqualStrings("Plain template", prompt3);
}

test "runner - run with no agents returns error" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    const steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Test",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{},
            .output_key = "step1:output",
            .prompt_template = "Test",
        },
    };

    const wf = workflow_mod.WorkflowDef{
        .id = "test",
        .name = "Test",
        .description = "Test",
        .steps = &steps,
    };

    const result = runner.run(&wf);
    try std.testing.expectError(WorkflowRunner.RunError.NoAgents, result);
}

test "runner - invalid workflow returns error" {
    const allocator = std.testing.allocator;
    var runner = WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "agent",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();
    try runner.registerAgent("agent", &agent);

    // Empty workflow is invalid
    const wf = workflow_mod.WorkflowDef{
        .id = "empty",
        .name = "Empty",
        .description = "No steps",
        .steps = &.{},
    };

    const result = runner.run(&wf);
    try std.testing.expectError(WorkflowRunner.RunError.InvalidWorkflow, result);
}

test {
    std.testing.refAllDecls(@This());
}
