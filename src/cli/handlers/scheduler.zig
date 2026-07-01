const std = @import("std");
const build_options = @import("build_options");
const memory_mod = @import("../../core/memory.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const usage_mod = @import("../usage.zig");
const telemetry = if (build_options.feat_telemetry)
    @import("../../features/telemetry/mod.zig")
else
    @import("../../features/telemetry/stub.zig");

/// `abi scheduler status`: print scheduler run statistics alongside memory
/// tracker usage. Rejects any other invocation with a usage error. Returns the
/// process exit code.
pub fn handleScheduler(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3 or !std.mem.eql(u8, args[2], "status")) {
        return usage_mod.usageError("usage: abi scheduler status");
    }

    var tracker = memory_mod.MemoryTracker.init(allocator);
    defer tracker.deinit();

    var sched = scheduler_mod.Scheduler.init(allocator);
    defer sched.deinit();
    sched.setMemoryTracker(&tracker);

    _ = try sched.submit("scheduler-status-probe", .low, statusProbeTask, null);
    try sched.runAll();

    const stats = sched.stats();
    std.debug.print(
        \\scheduler status
        \\source=cli-scheduler-status
        \\mode=one-shot
        \\running={d} pending={d} completed={d} failed={d} cancelled={d} total_tasks={d}
        \\memory_tracker=attached peak={d}B current={d}B records={d}
        \\
    , .{
        stats.running,
        stats.pending,
        stats.completed,
        stats.failed,
        stats.cancelled,
        stats.total_tasks,
        tracker.getPeakUsage(),
        tracker.getCurrentUsage(),
        tracker.getRecordCount(),
    });

    // Read-back of the always-on telemetry table (Prometheus text exposition,
    // incl. total/distinct/dropped self-observability). No-op notice when
    // -Dfeat-telemetry=false.
    const telemetry_text = try telemetry.writeText(allocator);
    defer allocator.free(telemetry_text);
    std.debug.print("{s}", .{telemetry_text});

    return 0;
}

fn statusProbeTask(ctx: ?*anyopaque) !void {
    _ = ctx;
}

test "scheduler handler reports one-shot status" {
    try std.testing.expectEqual(@as(u8, 0), try handleScheduler(std.testing.allocator, &.{ "abi", "scheduler", "status" }));
    try std.testing.expectEqual(@as(u8, 2), try handleScheduler(std.testing.allocator, &.{ "abi", "scheduler" }));
}

test {
    std.testing.refAllDecls(@This());
}
