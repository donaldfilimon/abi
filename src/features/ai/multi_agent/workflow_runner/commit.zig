const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const session_mod = @import("session.zig");
const prepare_mod = @import("prepare.zig");
const execute_mod = @import("execute.zig");

pub fn commitStepOutcome(
    runner: anytype,
    session: *session_mod.RunSession,
    step_id: []const u8,
    step: *const workflow_mod.Step,
    prepared: *const prepare_mod.PreparedStep,
    outcome: execute_mod.StepExecutionOutcome,
) void {
    session.stats.total_retries += outcome.retry_count;

    if (outcome.status == .completed) {
        if (outcome.output) |output| {
            runner.blackboard.put(step.output_key, output, prepared.profile_name) catch |err| {
                std.log.warn("Failed to update blackboard: {t}", .{err});
            };
        }

        const tracker_result = workflow_mod.StepResult{
            .step_id = step_id,
            .status = .completed,
            .output = outcome.output orelse "",
            .error_message = "",
            .duration_ns = outcome.duration_ms * std.time.ns_per_ms,
            .assigned_profile = prepared.profile_name,
        };
        session.tracker.markCompleted(step_id, tracker_result) catch |err| {
            std.log.warn("Failed to mark step completed: {t}", .{err});
        };
        session.stats.completed_steps += 1;

        const output_copy = if (outcome.output) |output|
            runner.allocator.dupe(u8, output) catch null
        else
            null;
        session.step_results.put(runner.allocator, step_id, .{
            .step_id = step_id,
            .output = output_copy,
            .status = .completed,
            .assigned_profile = prepared.profile_name,
            .attempts = outcome.attempts,
            .duration_ms = outcome.duration_ms,
        }) catch |err| {
            std.log.warn("Failed to generate step log: {t}", .{err});
        };

        if (outcome.output) |output| runner.allocator.free(output);

        runner.event_bus.publish(.{
            .event_type = .agent_finished,
            .task_id = session.task_id,
            .success = true,
            .duration_ns = outcome.duration_ms * std.time.ns_per_ms,
        });
        runner.supervisor.resetAgent(prepared.profile_name);
        return;
    }

    const tracker_result = workflow_mod.StepResult{
        .step_id = step_id,
        .status = .failed,
        .output = "",
        .error_message = "step execution failed",
        .duration_ns = outcome.duration_ms * std.time.ns_per_ms,
        .assigned_profile = prepared.profile_name,
    };
    session.tracker.markFailed(step_id, tracker_result) catch |err| {
        std.log.warn("Failed to mark step failed: {t}", .{err});
    };
    session.stats.failed_steps += 1;

    session.step_results.put(runner.allocator, step_id, .{
        .step_id = step_id,
        .output = null,
        .status = .failed,
        .assigned_profile = prepared.profile_name,
        .attempts = outcome.attempts,
        .duration_ms = outcome.duration_ms,
    }) catch |err| {
        std.log.warn("Failed to generate step log: {t}", .{err});
    };

    if (outcome.output) |output| runner.allocator.free(output);

    runner.event_bus.publish(.{
        .event_type = .agent_finished,
        .task_id = session.task_id,
        .success = false,
        .duration_ns = outcome.duration_ms * std.time.ns_per_ms,
    });
}


test {
    std.testing.refAllDecls(@This());
}
