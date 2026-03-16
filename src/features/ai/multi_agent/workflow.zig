//! Workflow DAG Engine
//!
//! Defines and executes directed acyclic graphs (DAGs) of agent tasks with
//! dependency tracking. Each step in the workflow specifies its dependencies,
//! required capabilities, and blackboard I/O keys.
//!
//! Features:
//! - **DAG validation**: Cycle detection and dependency resolution
//! - **Topological execution**: Steps run only after all dependencies complete
//! - **Parallel layers**: Independent steps within a layer execute concurrently
//! - **Blackboard integration**: Steps read inputs and write outputs to shared state
//! - **Capability matching**: Steps declare what capabilities the assigned agent needs

const std = @import("std");
const roles = @import("roles.zig");

// ============================================================================
// Types
// ============================================================================

/// Status of a workflow step.
pub const StepStatus = enum {
    /// Not yet started; waiting for dependencies.
    pending,
    /// All dependencies met; ready to execute.
    ready,
    /// Currently executing.
    running,
    /// Completed successfully.
    completed,
    /// Failed (error or timeout).
    failed,
    /// Skipped (dependency failed and step is not required).
    skipped,

    pub fn isTerminal(self: StepStatus) bool {
        return self == .completed or self == .failed or self == .skipped;
    }
};

/// A single step in a workflow DAG.
pub const Step = struct {
    /// Unique step identifier within the workflow.
    id: []const u8,
    /// Human-readable description of what this step does.
    description: []const u8,
    /// IDs of steps that must complete before this one can run.
    depends_on: []const []const u8,
    /// Capabilities required by the agent assigned to this step.
    required_capabilities: []const roles.Capability,
    /// Blackboard keys this step reads as input.
    input_keys: []const []const u8,
    /// Blackboard key where this step writes its output.
    output_key: []const u8,
    /// The prompt template for this step. May contain {input} placeholders.
    prompt_template: []const u8,
    /// Whether failure of this step should halt the entire workflow.
    is_critical: bool = true,
    /// Maximum execution time in milliseconds (0 = no limit).
    timeout_ms: u64 = 0,
    /// Assigned persona ID (empty = auto-assign based on capabilities).
    assigned_persona: []const u8 = "",
};

/// Result of executing a single workflow step.
pub const StepResult = struct {
    step_id: []const u8,
    status: StepStatus,
    output: []const u8,
    error_message: []const u8,
    duration_ns: u64,
    assigned_persona: []const u8,
};

/// Overall workflow status.
pub const WorkflowStatus = enum {
    /// Not yet started.
    created,
    /// Currently executing.
    running,
    /// All steps completed successfully.
    completed,
    /// One or more critical steps failed.
    failed,
    /// Execution was cancelled.
    cancelled,
};

// ============================================================================
// Workflow Definition
// ============================================================================

/// A complete workflow definition — a DAG of steps.
pub const WorkflowDef = struct {
    /// Unique workflow identifier.
    id: []const u8,
    /// Human-readable name.
    name: []const u8,
    /// Description of what this workflow accomplishes.
    description: []const u8,
    /// The steps in this workflow (order doesn't matter; dependencies define execution order).
    steps: []const Step,

    /// Validate the workflow DAG: check for missing dependencies and cycles.
    pub fn validate(self: WorkflowDef) ValidationResult {
        // Check for missing dependency references
        for (self.steps) |step| {
            for (step.depends_on) |dep_id| {
                if (!self.hasStep(dep_id)) {
                    return .{ .valid = false, .error_message = "missing dependency reference" };
                }
            }
        }

        // Check for cycles using DFS with coloring
        if (self.hasCycle()) {
            return .{ .valid = false, .error_message = "workflow contains a cycle" };
        }

        // Check for empty steps
        if (self.steps.len == 0) {
            return .{ .valid = false, .error_message = "workflow has no steps" };
        }

        return .{ .valid = true, .error_message = "" };
    }

    /// Check if a step with the given ID exists.
    pub fn hasStep(self: WorkflowDef, id: []const u8) bool {
        for (self.steps) |step| {
            if (std.mem.eql(u8, step.id, id)) return true;
        }
        return false;
    }

    /// Get a step by ID.
    pub fn getStep(self: WorkflowDef, id: []const u8) ?Step {
        for (self.steps) |step| {
            if (std.mem.eql(u8, step.id, id)) return step;
        }
        return null;
    }

    /// Compute execution layers — groups of steps that can run in parallel.
    /// Returns slices of step IDs grouped by execution layer (layer 0 first).
    pub fn computeLayers(self: WorkflowDef, allocator: std.mem.Allocator) ![]const []const []const u8 {
        var step_to_layer = std.StringHashMapUnmanaged(usize).empty;
        defer step_to_layer.deinit(allocator);

        // Assign layers: a step's layer = max(dependency layers) + 1
        // Steps with no dependencies are layer 0
        var changed = true;
        // Initialize all steps to layer 0
        for (self.steps) |step| {
            try step_to_layer.put(allocator, step.id, 0);
        }

        // Iterate until stable
        while (changed) {
            changed = false;
            for (self.steps) |step| {
                var max_dep_layer: usize = 0;
                for (step.depends_on) |dep_id| {
                    if (step_to_layer.get(dep_id)) |dep_layer| {
                        max_dep_layer = @max(max_dep_layer, dep_layer + 1);
                    }
                }
                const current = step_to_layer.get(step.id) orelse 0;
                if (max_dep_layer > current) {
                    try step_to_layer.put(allocator, step.id, max_dep_layer);
                    changed = true;
                }
            }
        }

        // Find max layer
        var max_layer: usize = 0;
        var iter = step_to_layer.iterator();
        while (iter.next()) |entry| {
            max_layer = @max(max_layer, entry.value_ptr.*);
        }

        // Group steps by layer
        var layers: std.ArrayListUnmanaged([]const []const u8) = .empty;
        errdefer {
            for (layers.items) |layer| allocator.free(layer);
            layers.deinit(allocator);
        }

        for (0..max_layer + 1) |layer_idx| {
            var layer_steps: std.ArrayListUnmanaged([]const u8) = .empty;
            errdefer layer_steps.deinit(allocator);

            for (self.steps) |step| {
                if ((step_to_layer.get(step.id) orelse 0) == layer_idx) {
                    try layer_steps.append(allocator, step.id);
                }
            }

            try layers.append(allocator, try layer_steps.toOwnedSlice(allocator));
        }

        return layers.toOwnedSlice(allocator);
    }

    // -- internal cycle detection --

    fn hasCycle(self: WorkflowDef) bool {
        // DFS coloring: 0=white, 1=gray (in progress), 2=black (done)
        var colors: [64]u8 = [_]u8{0} ** 64;
        if (self.steps.len > 64) return false; // safety limit

        for (0..self.steps.len) |i| {
            if (colors[i] == 0) {
                if (self.dfsHasCycle(i, &colors)) return true;
            }
        }
        return false;
    }

    fn dfsHasCycle(self: WorkflowDef, idx: usize, colors: *[64]u8) bool {
        colors[idx] = 1; // gray
        const step = self.steps[idx];
        for (step.depends_on) |dep_id| {
            for (self.steps, 0..) |s, j| {
                if (std.mem.eql(u8, s.id, dep_id)) {
                    if (colors[j] == 1) return true; // back edge = cycle
                    if (colors[j] == 0 and self.dfsHasCycle(j, colors)) return true;
                    break;
                }
            }
        }
        colors[idx] = 2; // black
        return false;
    }
};

/// Result of workflow validation.
pub const ValidationResult = struct {
    valid: bool,
    error_message: []const u8,
};

// ============================================================================
// Workflow Execution Tracker
// ============================================================================

/// Tracks the runtime state of a workflow execution.
pub const ExecutionTracker = struct {
    allocator: std.mem.Allocator,
    workflow: WorkflowDef,
    status: WorkflowStatus,
    step_statuses: std.StringHashMapUnmanaged(StepStatus),
    step_results: std.StringHashMapUnmanaged(StepResult),

    pub fn init(allocator: std.mem.Allocator, workflow: WorkflowDef) !ExecutionTracker {
        var step_statuses = std.StringHashMapUnmanaged(StepStatus).empty;
        errdefer step_statuses.deinit(allocator);

        for (workflow.steps) |step| {
            const initial: StepStatus = if (step.depends_on.len == 0) .ready else .pending;
            try step_statuses.put(allocator, step.id, initial);
        }

        return .{
            .allocator = allocator,
            .workflow = workflow,
            .status = .created,
            .step_statuses = step_statuses,
            .step_results = .{},
        };
    }

    pub fn deinit(self: *ExecutionTracker) void {
        self.step_statuses.deinit(self.allocator);
        self.step_results.deinit(self.allocator);
    }

    /// Get the status of a specific step.
    pub fn getStepStatus(self: *const ExecutionTracker, step_id: []const u8) ?StepStatus {
        return self.step_statuses.get(step_id);
    }

    /// Mark a step as running.
    pub fn markRunning(self: *ExecutionTracker, step_id: []const u8) void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .running;
        }
        self.status = .running;
    }

    /// Mark a step as completed and update dependent steps' readiness.
    pub fn markCompleted(self: *ExecutionTracker, step_id: []const u8, result: StepResult) !void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .completed;
        }
        try self.step_results.put(self.allocator, step_id, result);
        self.updateReadiness();
        self.checkOverallStatus();
    }

    /// Mark a step as failed. If critical, propagate failure.
    pub fn markFailed(self: *ExecutionTracker, step_id: []const u8, result: StepResult) !void {
        if (self.step_statuses.getPtr(step_id)) |ptr| {
            ptr.* = .failed;
        }
        try self.step_results.put(self.allocator, step_id, result);

        // If critical, skip all dependent steps
        if (self.workflow.getStep(step_id)) |step| {
            if (step.is_critical) {
                self.skipDependents(step_id);
            }
        }

        self.updateReadiness();
        self.checkOverallStatus();
    }

    /// Get all steps that are currently ready to execute.
    pub fn readySteps(self: *const ExecutionTracker, allocator: std.mem.Allocator) ![]const []const u8 {
        var result: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .ready) {
                try result.append(allocator, entry.key_ptr.*);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Check if the workflow is complete (all steps terminal).
    pub fn isComplete(self: *const ExecutionTracker) bool {
        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.isTerminal()) return false;
        }
        return true;
    }

    /// Count completed, failed, and total steps.
    pub fn progress(self: *const ExecutionTracker) Progress {
        var completed: usize = 0;
        var failed: usize = 0;
        var running: usize = 0;
        var total: usize = 0;

        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            total += 1;
            switch (entry.value_ptr.*) {
                .completed => completed += 1,
                .failed => failed += 1,
                .running => running += 1,
                else => {},
            }
        }

        return .{
            .completed = completed,
            .failed = failed,
            .running = running,
            .total = total,
        };
    }

    pub const Progress = struct {
        completed: usize,
        failed: usize,
        running: usize,
        total: usize,
    };

    // -- internal --

    fn updateReadiness(self: *ExecutionTracker) void {
        for (self.workflow.steps) |step| {
            const current = self.step_statuses.get(step.id) orelse continue;
            if (current != .pending) continue;

            // Check if all dependencies are completed
            var all_met = true;
            for (step.depends_on) |dep_id| {
                const dep_status = self.step_statuses.get(dep_id) orelse .pending;
                if (dep_status != .completed) {
                    all_met = false;
                    break;
                }
            }

            if (all_met) {
                if (self.step_statuses.getPtr(step.id)) |ptr| {
                    ptr.* = .ready;
                }
            }
        }
    }

    fn skipDependents(self: *ExecutionTracker, failed_step_id: []const u8) void {
        for (self.workflow.steps) |step| {
            for (step.depends_on) |dep_id| {
                if (std.mem.eql(u8, dep_id, failed_step_id)) {
                    if (self.step_statuses.getPtr(step.id)) |ptr| {
                        if (!ptr.*.isTerminal()) {
                            ptr.* = .skipped;
                        }
                    }
                    // Recursively skip dependents of this step too
                    self.skipDependents(step.id);
                    break;
                }
            }
        }
    }

    fn checkOverallStatus(self: *ExecutionTracker) void {
        if (!self.isComplete()) return;

        var any_failed = false;
        var iter = self.step_statuses.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* == .failed) {
                any_failed = true;
                break;
            }
        }

        self.status = if (any_failed) .failed else .completed;
    }
};

// ============================================================================
// Preset Workflows
// ============================================================================

/// Built-in workflow templates.
pub const preset_workflows = struct {
    pub const code_review = WorkflowDef{
        .id = "code-review",
        .name = "Multi-Perspective Code Review",
        .description = "Analyzes code from multiple angles then synthesizes findings",
        .steps = &.{
            .{
                .id = "analyze",
                .description = "Analyze code structure and identify areas of concern",
                .depends_on = &.{},
                .required_capabilities = &.{.code_review},
                .input_keys = &.{"task:input"},
                .output_key = "analyze:output",
                .prompt_template = "Analyze the following code and identify structural concerns:\n\n{input}",
            },
            .{
                .id = "security",
                .description = "Check for security vulnerabilities",
                .depends_on = &.{},
                .required_capabilities = &.{.security_audit},
                .input_keys = &.{"task:input"},
                .output_key = "security:output",
                .prompt_template = "Review the following code for security vulnerabilities:\n\n{input}",
            },
            .{
                .id = "quality",
                .description = "Evaluate code quality and suggest improvements",
                .depends_on = &.{},
                .required_capabilities = &.{ .critique, .code_review },
                .input_keys = &.{"task:input"},
                .output_key = "quality:output",
                .prompt_template = "Evaluate the quality of this code and suggest improvements:\n\n{input}",
            },
            .{
                .id = "synthesize",
                .description = "Combine all review findings into a final report",
                .depends_on = &.{ "analyze", "security", "quality" },
                .required_capabilities = &.{.synthesis},
                .input_keys = &.{ "analyze:output", "security:output", "quality:output" },
                .output_key = "review:final",
                .prompt_template = "Synthesize these code review findings into a final report:\n\n" ++
                    "Structure Analysis:\n{input}\n\n" ++
                    "Security Review:\n{input}\n\n" ++
                    "Quality Assessment:\n{input}",
            },
        },
    };

    pub const research = WorkflowDef{
        .id = "research",
        .name = "Research and Analysis",
        .description = "Researches a topic, analyzes findings, and produces a report",
        .steps = &.{
            .{
                .id = "gather",
                .description = "Gather information about the topic",
                .depends_on = &.{},
                .required_capabilities = &.{.summarization},
                .input_keys = &.{"task:input"},
                .output_key = "gather:output",
                .prompt_template = "Research and gather key information about:\n\n{input}",
            },
            .{
                .id = "analyze",
                .description = "Analyze the gathered information",
                .depends_on = &.{"gather"},
                .required_capabilities = &.{.data_analysis},
                .input_keys = &.{"gather:output"},
                .output_key = "analyze:output",
                .prompt_template = "Analyze the following research findings:\n\n{input}",
            },
            .{
                .id = "report",
                .description = "Write a clear report from the analysis",
                .depends_on = &.{"analyze"},
                .required_capabilities = &.{.doc_writing},
                .input_keys = &.{"analyze:output"},
                .output_key = "report:final",
                .prompt_template = "Write a clear, well-structured report based on this analysis:\n\n{input}",
            },
        },
    };

    pub const implement_feature = WorkflowDef{
        .id = "implement-feature",
        .name = "Feature Implementation",
        .description = "Plans, implements, tests, and reviews a feature",
        .steps = &.{
            .{
                .id = "plan",
                .description = "Design the implementation plan",
                .depends_on = &.{},
                .required_capabilities = &.{ .problem_decomposition, .planning },
                .input_keys = &.{"task:input"},
                .output_key = "plan:output",
                .prompt_template = "Design an implementation plan for:\n\n{input}",
            },
            .{
                .id = "implement",
                .description = "Write the implementation code",
                .depends_on = &.{"plan"},
                .required_capabilities = &.{.code_generation},
                .input_keys = &.{ "task:input", "plan:output" },
                .output_key = "implement:output",
                .prompt_template = "Implement the following based on the plan:\n\n{input}",
            },
            .{
                .id = "test",
                .description = "Write tests for the implementation",
                .depends_on = &.{"implement"},
                .required_capabilities = &.{.test_writing},
                .input_keys = &.{"implement:output"},
                .output_key = "test:output",
                .prompt_template = "Write comprehensive tests for:\n\n{input}",
            },
            .{
                .id = "review",
                .description = "Review the implementation and tests",
                .depends_on = &.{ "implement", "test" },
                .required_capabilities = &.{ .code_review, .critique },
                .input_keys = &.{ "implement:output", "test:output" },
                .output_key = "review:final",
                .prompt_template = "Review the implementation and tests for quality:\n\n{input}",
            },
        },
    };
};

// ============================================================================
// Tests
// ============================================================================

test "workflow validation - valid DAG" {
    const wf = preset_workflows.code_review;
    const result = wf.validate();
    try std.testing.expect(result.valid);
}

test "workflow validation - missing dependency" {
    const wf = WorkflowDef{
        .id = "bad",
        .name = "Bad Workflow",
        .description = "Has a missing dependency",
        .steps = &.{
            .{
                .id = "step1",
                .description = "Does something",
                .depends_on = &.{"nonexistent"},
                .required_capabilities = &.{},
                .input_keys = &.{},
                .output_key = "step1:out",
                .prompt_template = "do it",
            },
        },
    };
    const result = wf.validate();
    try std.testing.expect(!result.valid);
    try std.testing.expectEqualStrings("missing dependency reference", result.error_message);
}

test "workflow validation - empty" {
    const wf = WorkflowDef{
        .id = "empty",
        .name = "Empty",
        .description = "No steps",
        .steps = &.{},
    };
    const result = wf.validate();
    try std.testing.expect(!result.valid);
}

test "workflow validation - cycle detection" {
    const wf = WorkflowDef{
        .id = "cyclic",
        .name = "Cyclic",
        .description = "Has a cycle",
        .steps = &.{
            .{
                .id = "a",
                .description = "Step A",
                .depends_on = &.{"b"},
                .required_capabilities = &.{},
                .input_keys = &.{},
                .output_key = "a:out",
                .prompt_template = "a",
            },
            .{
                .id = "b",
                .description = "Step B",
                .depends_on = &.{"a"},
                .required_capabilities = &.{},
                .input_keys = &.{},
                .output_key = "b:out",
                .prompt_template = "b",
            },
        },
    };
    const result = wf.validate();
    try std.testing.expect(!result.valid);
    try std.testing.expectEqualStrings("workflow contains a cycle", result.error_message);
}

test "workflow compute layers" {
    const wf = preset_workflows.code_review;
    const layers = try wf.computeLayers(std.testing.allocator);
    defer {
        for (layers) |layer| std.testing.allocator.free(layer);
        std.testing.allocator.free(layers);
    }

    // code_review: analyze, security, quality are layer 0; synthesize is layer 1
    try std.testing.expectEqual(@as(usize, 2), layers.len);
    try std.testing.expectEqual(@as(usize, 3), layers[0].len); // parallel steps
    try std.testing.expectEqual(@as(usize, 1), layers[1].len); // synthesize
}

test "execution tracker initial state" {
    const wf = preset_workflows.code_review;
    var tracker = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tracker.deinit();

    try std.testing.expectEqual(WorkflowStatus.created, tracker.status);

    // Steps with no dependencies should be ready
    try std.testing.expectEqual(StepStatus.ready, tracker.getStepStatus("analyze").?);
    try std.testing.expectEqual(StepStatus.ready, tracker.getStepStatus("security").?);

    // Steps with dependencies should be pending
    try std.testing.expectEqual(StepStatus.pending, tracker.getStepStatus("synthesize").?);
}

test "execution tracker step progression" {
    const wf = preset_workflows.research;
    var tracker = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tracker.deinit();

    // Initially only "gather" is ready
    try std.testing.expectEqual(StepStatus.ready, tracker.getStepStatus("gather").?);
    try std.testing.expectEqual(StepStatus.pending, tracker.getStepStatus("analyze").?);

    // Mark gather as completed
    tracker.markRunning("gather");
    try std.testing.expectEqual(StepStatus.running, tracker.getStepStatus("gather").?);

    try tracker.markCompleted("gather", .{
        .step_id = "gather",
        .status = .completed,
        .output = "findings",
        .error_message = "",
        .duration_ns = 1000,
        .assigned_persona = "researcher",
    });

    // Now "analyze" should be ready
    try std.testing.expectEqual(StepStatus.ready, tracker.getStepStatus("analyze").?);
}

test "execution tracker failure propagation" {
    const wf = preset_workflows.research;
    var tracker = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tracker.deinit();

    // Fail the gather step (which is critical by default)
    try tracker.markFailed("gather", .{
        .step_id = "gather",
        .status = .failed,
        .output = "",
        .error_message = "timeout",
        .duration_ns = 5000,
        .assigned_persona = "researcher",
    });

    // Dependents should be skipped
    try std.testing.expectEqual(StepStatus.skipped, tracker.getStepStatus("analyze").?);
    try std.testing.expectEqual(StepStatus.skipped, tracker.getStepStatus("report").?);
    try std.testing.expectEqual(WorkflowStatus.failed, tracker.status);
}

test "execution tracker ready steps" {
    const wf = preset_workflows.code_review;
    var tracker = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tracker.deinit();

    const ready = try tracker.readySteps(std.testing.allocator);
    defer std.testing.allocator.free(ready);

    // analyze, security, quality should all be ready
    try std.testing.expectEqual(@as(usize, 3), ready.len);
}

test "execution tracker progress" {
    const wf = preset_workflows.research;
    var tracker = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tracker.deinit();

    const p = tracker.progress();
    try std.testing.expectEqual(@as(usize, 3), p.total);
    try std.testing.expectEqual(@as(usize, 0), p.completed);
}

test {
    std.testing.refAllDecls(@This());
}
