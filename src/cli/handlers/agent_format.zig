//! Shared stdout formatting for `abi agent` subcommand handlers.
//!
//! Pure print helpers extracted so plan/train/multi/spawn/browser paths share
//! the same scheduler and memory-tracker summary lines without duplicating
//! format strings.

const std = @import("std");

pub fn printSchedulerStats(s: anytype) void {
    std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
}

pub fn printMemoryTrackerStats(peak_bytes: usize, record_count: usize) void {
    std.debug.print("memory (tracker): peak={d}B records={d}\n", .{ peak_bytes, record_count });
}

pub fn printTrainingHeader() void {
    std.debug.print("training executed via scheduler (real tasks, not demos)\n", .{});
}

pub fn printProfileTrainingResult(profile: []const u8, message: []const u8, store_count: usize, backend: []const u8) void {
    std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ profile, message, store_count, backend });
}

pub fn printAllProfilesTrainingResult(message: []const u8, store_count: usize, backend: []const u8) void {
    std.debug.print("abbey,aviva,abi: {s} ({d} wdbx record(s), backend={s})\n", .{ message, store_count, backend });
}

pub fn printBackgroundTaskIds(task_ids: []const u64, specs: anytype) void {
    std.debug.print("submitted background agent tasks:\n", .{});
    for (task_ids, 0..) |id, n| {
        std.debug.print("  task_id={d} worker={s}\n", .{ id, specs[n].name });
    }
}

pub fn printBrowserPlannerBanner(aggregated: []const u8) void {
    std.debug.print("\n--- local planner worker (dry-run) ---\n{s}\n", .{aggregated});
}

test {
    std.testing.refAllDecls(@This());
}
