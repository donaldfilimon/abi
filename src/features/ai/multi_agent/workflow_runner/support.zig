const std = @import("std");
const workflow_mod = @import("../workflow.zig");
const roles = @import("../roles.zig");
const supervisor_mod = @import("../supervisor.zig");
const agents_mod = @import("../../agents/mod.zig");
const time = @import("../../../../foundation/mod.zig").time;
const build_options = @import("build_options");
const training = if (build_options.feat_training)
    @import("../../training/mod.zig")
else
    @import("../../training/stub.zig");
const FeedbackType = training.FeedbackType;

pub fn assignProfile(runner: anytype, step: *const workflow_mod.Step) ?roles.Profile {
    if (step.assigned_profile.len > 0) {
        return runner.profile_registry.get(step.assigned_profile);
    }
    if (step.required_capabilities.len > 0) {
        return runner.profile_registry.findBestMatch(step.required_capabilities);
    }
    return null;
}

pub fn gatherInputs(runner: anytype, step: *const workflow_mod.Step) ![]const u8 {
    if (step.input_keys.len == 0) return "";

    var parts = std.ArrayListUnmanaged(u8).empty;
    errdefer parts.deinit(runner.allocator);

    for (step.input_keys, 0..) |key, i| {
        if (runner.blackboard.get(key)) |entry| {
            if (i > 0) {
                try parts.appendSlice(runner.allocator, "\n");
            }
            try parts.appendSlice(runner.allocator, key);
            try parts.appendSlice(runner.allocator, ": ");
            try parts.appendSlice(runner.allocator, entry.value);
        }
    }

    if (parts.items.len == 0) return "";
    return parts.toOwnedSlice(runner.allocator);
}

pub fn buildPrompt(runner: anytype, template: []const u8, inputs: []const u8) ![]const u8 {
    if (inputs.len == 0) return template;

    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(runner.allocator);

    var i: usize = 0;
    while (i < template.len) {
        if (i + 7 <= template.len and std.mem.eql(u8, template[i .. i + 7], "{input}")) {
            try result.appendSlice(runner.allocator, inputs);
            i += 7;
        } else if (i + 9 <= template.len and std.mem.eql(u8, template[i .. i + 9], "{context}")) {
            try result.appendSlice(runner.allocator, inputs);
            i += 9;
        } else {
            try result.append(runner.allocator, template[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(runner.allocator);
}

pub fn selectAgent(runner: anytype, profile_name: []const u8) ?*agents_mod.Agent {
    if (runner.agent_map.get(profile_name)) |agent| return agent;

    var iter = runner.agent_map.iterator();
    if (iter.next()) |entry| {
        return entry.value_ptr.*;
    }
    return null;
}

pub fn handleFailure(
    runner: anytype,
    step_id: []const u8,
    _: anyerror,
    tracker: *workflow_mod.ExecutionTracker,
) supervisor_mod.SupervisorAction {
    const step = tracker.workflow.getStep(step_id);
    const severity: supervisor_mod.FailureSeverity = if (step) |s|
        (if (s.is_critical) .persistent else .transient)
    else
        .transient;

    const decision = runner.supervisor.reportFailure(.{
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

pub fn recordWorkflowOutcome(runner: anytype, success: bool) void {
    const sys = runner.learning_system orelse return;

    const bb_keys = runner.blackboard.keys(runner.allocator) catch return;
    defer {
        for (bb_keys) |key| runner.allocator.free(key);
        runner.allocator.free(bb_keys);
    }

    const feedback: FeedbackType = if (success) .positive else .negative;
    const confidence: f32 = if (success) 0.8 else 0.3;

    for (bb_keys) |key| {
        const entry = runner.blackboard.get(key) orelse continue;

        const summary = std.fmt.allocPrint(
            runner.allocator,
            "{s}={s}",
            .{ key, entry.value },
        ) catch continue;
        defer runner.allocator.free(summary);

        const tokens = runner.allocator.alloc(u32, summary.len) catch continue;
        defer runner.allocator.free(tokens);
        for (summary, 0..) |byte, i| {
            tokens[i] = @intCast(byte);
        }

        sys.recordExperience(
            tokens,
            tokens,
            feedback,
            confidence,
            .text_conversation,
        ) catch continue;
    }
}

test {
    std.testing.refAllDecls(@This());
}
