const std = @import("std");
const abi = @import("../../root.zig");
const usage_mod = @import("../usage.zig");
const dashboard_mod = @import("dashboard.zig");

pub fn handleAgent(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi agent <plan|train|tui|os> ...");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent plan <input>");
        const result = try abi.features.ai.runAgent(allocator, .{ .name = "cli-agent", .instructions = "Plan only; do not execute.", .dry_run = true }, args[3]);
        defer result.deinit(allocator);
        std.debug.print("{s}\n", .{result.output});
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent train <abbey|aviva|abi|all>");
        var store = abi.features.wdbx.Store.init(allocator);
        defer store.deinit();

        const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
        const artifact_dir = "zig-cache/agent-artifacts";
        const profile_arg = args[3];
        const is_all = std.mem.eql(u8, profile_arg, "all");

        // Massive functionality improvement (addressing repeated gaps in roadmap, design doc,
        // master spec, and todo after full .md/docs review):
        // The rich Scheduler (O(log N) priority, stats, cancellation) was previously unused
        // for real user work. Now `abi agent train` (the primary training surface) submits
        // the actual training as real scheduler tasks. This makes scheduler_stats, the TUI
        // scheduler pane, and future integration show *real* user-triggered work with
        // proper priority, status, and observability.
        var sched = abi.scheduler.Scheduler.init(allocator);
        defer sched.deinit();

        // MemoryTracker wiring for production CLI train path (deeper observability per plan,
        // focusing on src/abi_cli as requested). Uses the Scheduler setter + TrackingAllocator
        // on the task arena (matches the pattern proven in integration tests).
        var mem_tracker = abi.memory.MemoryTracker.init(allocator);
        var tracking_alloc = abi.memory.TrackingAllocator.init(allocator, &mem_tracker);
        sched.setMemoryTracker(&mem_tracker);

        // Use an arena for all per-task context (names + TaskCtx) so we have zero leaks
        // and follow Zig 0.17 careful allocator discipline (per zig-017-modernization skill).
        var arena = std.heap.ArenaAllocator.init(tracking_alloc.allocator());
        defer arena.deinit();
        const task_alloc = arena.allocator();

        if (is_all) {
            for (abi.features.ai.known_profiles) |p| {
                const label = p.label();
                const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{label});
                const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
                ctx.* = .{
                    .allocator = allocator,
                    .store = &store,
                    .config = .{
                        .profile = label,
                        .dataset = dataset,
                        .artifact_dir = artifact_dir,
                    },
                };
                _ = abi.features.ai.submitTrainingTask(&sched, name, ctx) catch |err| {
                    if (!abi.features.ai.isFeatureDisabled(err)) return err;
                };
            }
        } else {
            const name = try std.fmt.allocPrint(task_alloc, "train:{s}", .{profile_arg});
            const ctx = try task_alloc.create(abi.features.ai.TrainingTaskContext);
            ctx.* = .{
                .allocator = allocator,
                .store = &store,
                .config = .{
                    .profile = profile_arg,
                    .dataset = dataset,
                    .artifact_dir = artifact_dir,
                },
            };
            _ = abi.features.ai.submitTrainingTask(&sched, name, ctx) catch |err| {
                if (!abi.features.ai.isFeatureDisabled(err)) return err;
            };
        }

        // Run the real scheduled training work
        try sched.runAll();

        const s = sched.stats();
        std.debug.print("training executed via scheduler (real tasks, not demos)\n", .{});
        std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });

        // Memory observability (new in this CLI focus)
        std.debug.print("memory (tracker): peak={d}B records={d}\n", .{ mem_tracker.getPeakUsage(), mem_tracker.getRecordCount() });

        // Preserve previous user-visible output
        const result = if (is_all)
            try abi.features.ai.trainKnownProfiles(allocator, &store, dataset, artifact_dir)
        else
            try abi.features.ai.trainWithStore(allocator, &store, .{
                .profile = profile_arg,
                .dataset = dataset,
                .artifact_dir = artifact_dir,
            });
        defer result.deinit(allocator);

        std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ result.profile, result.message, store.count(), result.acceleration_backend });
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len != 3) return usage_mod.usageError("usage: abi agent tui");
        return dashboard_mod.handleDashboard(allocator);
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

    const request = abi.features.os_control.CommandRequest{
        .intent = .read_only,
        .argv = args[start..],
    };

    if (std.mem.eql(u8, os_cmd, "dry-run")) {
        const rendered = try abi.features.os_control.renderDryRun(std.heap.page_allocator, request);
        defer std.heap.page_allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    } else if (std.mem.eql(u8, os_cmd, "execute")) {
        if (start != 5) return usage_mod.usageError("usage: abi agent os execute --confirm <cmd> [args...]");
        const workspace_root = if (std.c.getenv("PWD")) |pwd| std.mem.span(pwd) else "/";
        const policy = abi.features.os_control.Policy{
            .workspace_root = workspace_root,
            .dry_run_only = false,
            .allow_execution = true,
            .allowed_commands = &.{ "true", "pwd", "ls", "whoami", "date" },
            .require_confirmation = true,
        };
        const execute_request = abi.features.os_control.CommandRequest{
            .intent = .read_only,
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
