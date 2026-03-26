const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const support = @import("support.zig");
const time = @import("../../../../foundation/mod.zig").time;

pub const StepExecutionOutcome = struct {
    output: ?[]const u8 = null,
    status: workflow_mod.StepStatus = .failed,
    attempts: u32 = 0,
    retry_count: u32 = 0,
    duration_ms: u64 = 0,
    escalated: bool = false,
};

pub fn executeStepAttempts(
    runner: anytype,
    tracker: *workflow_mod.ExecutionTracker,
    step_id: []const u8,
    prompt: []const u8,
    profile_name: []const u8,
) StepExecutionOutcome {
    var outcome = StepExecutionOutcome{};
    const agent = support.selectAgent(runner, profile_name);
    var step_timer = time.Timer.start() catch null;

    while (outcome.attempts <= runner.config.max_retries) : (outcome.attempts += 1) {
        if (agent) |resolved| {
            const result = resolved.process(prompt, runner.allocator) catch |err| {
                const action = support.handleFailure(runner, step_id, err, tracker);
                switch (action) {
                    .retry, .reassign, .restart => {
                        outcome.retry_count += 1;
                        continue;
                    },
                    .skip => break,
                    .escalate => {
                        outcome.escalated = true;
                        break;
                    },
                }
            };

            outcome.output = result;
            outcome.status = .completed;
            break;
        } else {
            break;
        }
    }

    outcome.duration_ms = if (step_timer) |*timer|
        timer.read() / std.time.ns_per_ms
    else
        0;

    return outcome;
}


test {
    std.testing.refAllDecls(@This());
}
