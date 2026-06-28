const std = @import("std");
const abi = @import("../../root.zig");
const usage_mod = @import("../usage.zig");

pub fn handleAgent(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi agent <plan|train|tui|os> ...");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent plan <input>");
        var sched = abi.scheduler.Scheduler.init(allocator);
        defer sched.deinit();

        var mem_tracker = abi.memory.MemoryTracker.init(allocator);
        defer mem_tracker.deinit();
        var tracking_alloc = abi.memory.TrackingAllocator.init(allocator, &mem_tracker);
        sched.setMemoryTracker(&mem_tracker);

        const plan_allocator = tracking_alloc.allocator();
        const result = try abi.features.ai.runAgentWithScheduler(plan_allocator, &sched, "agent:plan", .{ .name = "cli-agent", .instructions = "Plan only; do not execute.", .dry_run = true }, args[3]);
        defer result.deinit(plan_allocator);
        std.debug.print("{s}\n", .{result.output});
        const s = sched.stats();
        std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
        std.debug.print("memory (tracker): peak={d}B records={d}\n", .{ mem_tracker.getPeakUsage(), mem_tracker.getRecordCount() });
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent train <abbey|aviva|abi|all>");
        var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
        defer session.deinit();
        const store = session.storePtr();

        const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
        const artifact_dir = "zig-cache/agent-artifacts";
        const profile_arg = args[3];
        const is_all = std.mem.eql(u8, profile_arg, "all");

        var sched = abi.scheduler.Scheduler.init(allocator);
        defer sched.deinit();

        var mem_tracker = abi.memory.MemoryTracker.init(allocator);
        defer mem_tracker.deinit();
        var tracking_alloc = abi.memory.TrackingAllocator.init(allocator, &mem_tracker);
        sched.setMemoryTracker(&mem_tracker);

        var arena = std.heap.ArenaAllocator.init(tracking_alloc.allocator());
        defer arena.deinit();
        const task_alloc = arena.allocator();

        var training_contexts: std.ArrayListUnmanaged(*abi.features.ai.TrainingTaskContext) = .empty;
        defer {
            for (training_contexts.items) |ctx| ctx.deinitResult();
            training_contexts.deinit(allocator);
        }

        if (is_all) {
            for (abi.features.ai.known_profiles) |p| {
                const label = p.label();
                const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{label});
                const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
                ctx.* = .{
                    .allocator = allocator,
                    .store = store,
                    .config = .{
                        .profile = label,
                        .dataset = dataset,
                        .artifact_dir = artifact_dir,
                    },
                };
                if (abi.features.ai.submitTrainingTask(&sched, name, ctx)) |_| {
                    try training_contexts.append(allocator, ctx);
                } else |err| {
                    if (!abi.features.ai.isFeatureDisabled(err)) return err;
                }
            }
        } else {
            const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{profile_arg});
            const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
            ctx.* = .{
                .allocator = allocator,
                .store = store,
                .config = .{
                    .profile = profile_arg,
                    .dataset = dataset,
                    .artifact_dir = artifact_dir,
                },
            };
            if (abi.features.ai.submitTrainingTask(&sched, name, ctx)) |_| {
                try training_contexts.append(allocator, ctx);
            } else |err| {
                if (!abi.features.ai.isFeatureDisabled(err)) return err;
            }
        }

        try sched.runAll();

        const s = sched.stats();
        std.debug.print("training executed via scheduler (real tasks, not demos)\n", .{});
        std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });

        std.debug.print("memory (tracker): peak={d}B records={d}\n", .{ mem_tracker.getPeakUsage(), mem_tracker.getRecordCount() });

        if (training_contexts.items.len == 0) {
            const result = if (is_all)
                try abi.features.ai.trainKnownProfiles(allocator, store, dataset, artifact_dir)
            else
                try abi.features.ai.trainWithStore(allocator, store, .{
                    .profile = profile_arg,
                    .dataset = dataset,
                    .artifact_dir = artifact_dir,
                });
            defer result.deinit(allocator);

            std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ result.profile, result.message, store.count(), result.acceleration_backend });
            return 0;
        }

        if (is_all) {
            var records_stored: usize = 0;
            var backend: []const u8 = "unknown";
            for (training_contexts.items) |ctx| {
                const result = ctx.result orelse return error.MissingTrainingResult;
                records_stored += result.records_stored;
                backend = result.acceleration_backend;
            }
            const message: []const u8 = if (records_stored == 0)
                "known agent profiles accepted; wdbx feature is disabled for this build"
            else
                "known agent profiles recorded in wdbx";
            std.debug.print("abbey,aviva,abi: {s} ({d} wdbx record(s), backend={s})\n", .{ message, store.count(), backend });
        } else {
            const result = training_contexts.items[0].result orelse return error.MissingTrainingResult;
            std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ result.profile, result.message, store.count(), result.acceleration_backend });
        }
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len != 3) return usage_mod.usageError("usage: abi agent tui");

        var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
        defer session.deinit();
        const store = session.storePtr();

        var sched = abi.scheduler.Scheduler.init(allocator);
        defer sched.deinit();

        var repl = abi.features.tui.ReplLoop.init(allocator, store, &sched, .{});
        defer repl.deinit();
        repl.run(io) catch |err| {
            std.debug.print("error: interactive REPL failed: {s}\n", .{@errorName(err)});
            return 1;
        };
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "os") and args.len >= 5) {
        return handleAgentOs(io, allocator, args);
    } else {
        return usage_mod.usageError("usage: abi agent <plan|train|tui|os dry-run|os execute> ...");
    }
}

pub fn handleAgentOs(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    _ = allocator;
    const os_cmd = args[3];
    const start: usize = if (std.mem.eql(u8, os_cmd, "execute") and args.len >= 6 and std.mem.eql(u8, args[4], "--confirm")) 5 else 4;
    if (start >= args.len) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");

    // Intent is honest audit metadata: arbitrary user argv is not provably
    // read-only, so it is classified `.unknown` (the policy gate is the
    // allow-list + workspace containment, not this label).
    const request = abi.features.os_control.CommandRequest{
        .intent = .unknown,
        .argv = args[start..],
    };

    if (std.mem.eql(u8, os_cmd, "dry-run")) {
        const rendered = try abi.features.os_control.renderDryRun(std.heap.page_allocator, request);
        defer std.heap.page_allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    } else if (std.mem.eql(u8, os_cmd, "execute")) {
        if (start != 5) return usage_mod.usageError("usage: abi agent os execute --confirm <cmd> [args...]");
        // Workspace containment (os_control.pathsContained) only rejects
        // absolute path args that escape `workspace_root`. Falling back to "/"
        // when PWD is unset would make every absolute path "contained" and
        // silently disable the guard, so refuse instead of widening the sandbox.
        // Portable env lookup (no libc), borrowed from the captured process
        // environment. On Windows PWD is typically unset, so this resolves to
        // the safe refuse path rather than widening the sandbox.
        const workspace_root = abi.foundation.env.get("PWD") orelse "";
        if (workspace_root.len == 0 or !std.fs.path.isAbsolute(workspace_root)) {
            return usage_mod.usageError("cannot determine an absolute workspace root (PWD unset); refusing to execute");
        }
        const policy = abi.features.os_control.Policy{
            .workspace_root = workspace_root,
            .dry_run_only = false,
            .allow_execution = true,
            .allowed_commands = &.{ "true", "pwd", "ls", "whoami", "date" },
            .require_confirmation = true,
        };
        const execute_request = abi.features.os_control.CommandRequest{
            .intent = .unknown,
            .argv = args[start..],
            .confirm_execution = true,
        };
        const result = try abi.features.os_control.executeConfirmed(std.heap.page_allocator, io, execute_request, policy);
        std.debug.print("{s}\n", .{result.message});
        return result.exit_code orelse 0;
    } else {
        return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    }
}

test "agent dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    // Only the cheap usage branches are exercised — none of these spin up the
    // scheduler, store, or TUI, and none execute an OS command.
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "plan" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "train" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os" }));
    // `execute` without the mandatory `--confirm` token must refuse.
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "execute", "ls" }));
}

test {
    std.testing.refAllDecls(@This());
}
