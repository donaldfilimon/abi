//! `abi agent plan <input>` — dry-run agent planning via the scheduler.

const std = @import("std");
const abi = @import("abi");
const format = @import("agent_format.zig");

pub fn handleAgentPlanInput(io: std.Io, allocator: std.mem.Allocator, input: []const u8) !u8 {
    const augmented = try abi.features.ai.file_context.buildAgentContext(
        io,
        allocator,
        input,
        ".",
        abi.features.ai.file_context.DEFAULT_BUDGET_BYTES,
        .{ .include_tree = true, .include_git_diff = true, .git_stat_only = true },
    );
    defer allocator.free(augmented);

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    var mem_tracker = abi.memory.MemoryTracker.init(allocator);
    defer mem_tracker.deinit();
    var tracking_alloc = abi.memory.TrackingAllocator.init(allocator, &mem_tracker);
    sched.setMemoryTracker(&mem_tracker);

    const plan_allocator = tracking_alloc.allocator();
    const result = try abi.features.ai.runAgentWithScheduler(plan_allocator, &sched, "agent:plan", .{ .name = "cli-agent", .instructions = "Plan only; do not execute.", .dry_run = true }, augmented);
    defer result.deinit(plan_allocator);
    std.debug.print("{s}\n", .{result.output});
    format.printSchedulerStats(sched.stats());
    format.printMemoryTrackerStats(mem_tracker.getPeakUsage(), mem_tracker.getRecordCount());
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
