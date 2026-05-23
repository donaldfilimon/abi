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
        const result = if (std.mem.eql(u8, args[3], "all"))
            try abi.features.ai.trainKnownProfiles(allocator, &store, dataset, artifact_dir)
        else
            try abi.features.ai.trainWithStore(allocator, &store, .{
                .profile = args[3],
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
