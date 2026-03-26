const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const session_mod = @import("session.zig");
const support = @import("support.zig");

pub fn finalizeRun(
    runner: anytype,
    def: *const workflow_mod.WorkflowDef,
    session: *session_mod.RunSession,
) @import("../types.zig").WorkflowResult {
    const total_ms: u64 = if (session.overall_timer) |*timer|
        timer.read() / std.time.ns_per_ms
    else
        0;
    session.stats.total_duration_ms = total_ms;

    const progress = session.tracker.progress();
    session.stats.skipped_steps = @intCast(progress.total - progress.completed - progress.failed - progress.running);

    var final_output: ?[]const u8 = null;
    if (def.steps.len > 0) {
        const last_step = def.steps[def.steps.len - 1];
        if (runner.blackboard.get(last_step.output_key)) |entry| {
            final_output = runner.allocator.dupe(u8, entry.value) catch null;
        }
    }

    const success = session.stats.failed_steps == 0 and session.stats.completed_steps > 0;
    if (success) {
        runner.event_bus.taskCompleted(session.task_id, total_ms * std.time.ns_per_ms);
    } else {
        runner.event_bus.taskFailed(session.task_id, "workflow had failures");
    }

    if (runner.learning_system != null) {
        support.recordWorkflowOutcome(runner, success);
    }

    return session.intoResult(success, final_output);
}

test {
    std.testing.refAllDecls(@This());
}
