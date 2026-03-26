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

const types = @import("workflow/types.zig");
const definition = @import("workflow/definition.zig");
const tracker = @import("workflow/tracker.zig");
const presets = @import("workflow/presets.zig");

pub const StepStatus = types.StepStatus;
pub const Step = types.Step;
pub const StepResult = types.StepResult;
pub const WorkflowStatus = types.WorkflowStatus;
pub const ValidationResult = types.ValidationResult;

pub const WorkflowDef = definition.WorkflowDef;

pub const ExecutionTracker = tracker.ExecutionTracker;

pub const preset_workflows = presets.preset_workflows;

test "workflow validation - valid DAG" {
    const wf = preset_workflows.code_review;
    const result = try wf.validate(std.testing.allocator);
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
    const result = try wf.validate(std.testing.allocator);
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
    const result = try wf.validate(std.testing.allocator);
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
    const result = try wf.validate(std.testing.allocator);
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

    try std.testing.expectEqual(@as(usize, 2), layers.len);
    try std.testing.expectEqual(@as(usize, 3), layers[0].len);
    try std.testing.expectEqual(@as(usize, 1), layers[1].len);
}

test "execution tracker initial state" {
    const wf = preset_workflows.code_review;
    var tr = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tr.deinit();

    try std.testing.expectEqual(WorkflowStatus.created, tr.status);
    try std.testing.expectEqual(StepStatus.ready, tr.getStepStatus("analyze").?);
    try std.testing.expectEqual(StepStatus.ready, tr.getStepStatus("security").?);
    try std.testing.expectEqual(StepStatus.pending, tr.getStepStatus("synthesize").?);
}

test "execution tracker step progression" {
    const wf = preset_workflows.research;
    var tr = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tr.deinit();

    try std.testing.expectEqual(StepStatus.ready, tr.getStepStatus("gather").?);
    try std.testing.expectEqual(StepStatus.pending, tr.getStepStatus("analyze").?);

    tr.markRunning("gather");
    try std.testing.expectEqual(StepStatus.running, tr.getStepStatus("gather").?);

    try tr.markCompleted("gather", .{
        .step_id = "gather",
        .status = .completed,
        .output = "findings",
        .error_message = "",
        .duration_ns = 1000,
        .assigned_profile = "researcher",
    });

    try std.testing.expectEqual(StepStatus.ready, tr.getStepStatus("analyze").?);
}

test "execution tracker failure propagation" {
    const wf = preset_workflows.research;
    var tr = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tr.deinit();

    try tr.markFailed("gather", .{
        .step_id = "gather",
        .status = .failed,
        .output = "",
        .error_message = "timeout",
        .duration_ns = 5000,
        .assigned_profile = "researcher",
    });

    try std.testing.expectEqual(StepStatus.skipped, tr.getStepStatus("analyze").?);
    try std.testing.expectEqual(StepStatus.skipped, tr.getStepStatus("report").?);
    try std.testing.expectEqual(WorkflowStatus.failed, tr.status);
}

test "execution tracker ready steps" {
    const wf = preset_workflows.code_review;
    var tr = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tr.deinit();

    const ready = try tr.readySteps(std.testing.allocator);
    defer std.testing.allocator.free(ready);

    try std.testing.expectEqual(@as(usize, 3), ready.len);
}

test "execution tracker progress" {
    const wf = preset_workflows.research;
    var tr = try ExecutionTracker.init(std.testing.allocator, wf);
    defer tr.deinit();

    const p = tr.progress();
    try std.testing.expectEqual(@as(usize, 3), p.total);
    try std.testing.expectEqual(@as(usize, 0), p.completed);
}

test {
    std.testing.refAllDecls(@This());
}
