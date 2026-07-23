//! Multi-agent family: `abi agent multi|spawn|browser ...`
//!
//! Concurrent profile runs, custom worker spawn (optional background), and
//! local browser-orchestration planning (dry-run default).

const std = @import("std");
const abi = @import("abi");
const usage_mod = @import("../usage.zig");
const help = @import("agent_help.zig");
const format = @import("agent_format.zig");

pub fn handleAgentMultiInput(io: std.Io, allocator: std.mem.Allocator, input: []const u8) !u8 {
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

    var result = try abi.features.ai.runMultiAgentWithScheduler(allocator, &sched, "agent:multi", augmented);
    defer result.deinit(allocator);
    std.debug.print("{s}\n", .{result.aggregated});
    format.printSchedulerStats(sched.stats());
    return 0;
}

pub fn handleAgentSpawnArgv(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    _ = io;
    if (args.len < 4) return usage_mod.usageError("usage: abi agent spawn [--background] [--workers <spec>] <input>");
    const cli_args = args[3..];

    const parsed = help.parseSpawnArgs(allocator, cli_args) catch {
        return usage_mod.usageError("usage: abi agent spawn [--background] [--workers <spec>] <input>");
    } orelse return usage_mod.usageError("usage: abi agent spawn [--background] [--workers <spec>] <input>");
    defer allocator.free(parsed.input);

    var owned_specs: ?[]abi.features.ai.AgentWorkerSpec = null;
    const specs_slice: []const abi.features.ai.AgentWorkerSpec = blk: {
        if (parsed.workers_spec) |spec_text| {
            owned_specs = abi.features.ai.parseWorkerSpecs(allocator, spec_text) catch |err| switch (err) {
                error.InvalidWorkerSpec, error.InvalidAgentToolHint => return usage_mod.usageError(
                    "usage: abi agent spawn [--background] [--workers \"name|instructions|hints;...\"] <input>",
                ),
                else => return err,
            };
            break :blk owned_specs.?;
        }
        break :blk &[_]abi.features.ai.AgentWorkerSpec{
            .{ .name = "smart-agent", .instructions = "General-purpose planning and exploration.", .tool_hints = &.{ .plan, .explore } },
        };
    };
    defer if (owned_specs) |specs| abi.features.ai.freeWorkerSpecs(allocator, specs);

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    if (parsed.background) {
        var batch = try abi.features.ai.submitAgentsBackground(allocator, &sched, "agent:spawn", specs_slice, parsed.input);
        defer batch.deinit();
        format.printBackgroundTaskIds(batch.task_ids, specs_slice);
        try sched.runAll();
        var collected = try abi.features.ai.collectBackgroundBatch(allocator, &batch, specs_slice);
        defer collected.deinit(allocator);
        std.debug.print("{s}\n", .{collected.aggregated});
    } else {
        var result = try abi.features.ai.runCustomMultiAgentWithScheduler(allocator, &sched, "agent:spawn", specs_slice, parsed.input);
        defer result.deinit(allocator);
        std.debug.print("{s}\n", .{result.aggregated});
    }

    format.printSchedulerStats(sched.stats());
    return 0;
}

pub fn handleAgentBrowserArgv(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    _ = io;
    if (args.len < 4) return usage_mod.usageError("usage: abi agent browser [--url <url>] [--execute --confirm] <task>");
    const cli_args = args[3..];

    const parsed = help.parseBrowserArgs(allocator, cli_args) catch |err| switch (err) {
        error.Usage => return usage_mod.usageError("usage: abi agent browser --execute --confirm <task>"),
        else => return err,
    } orelse return usage_mod.usageError("usage: abi agent browser [--url <url>] [--execute --confirm] <task>");
    defer allocator.free(parsed.task);

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    var plan = try abi.features.ai.planBrowserOrchestration(allocator, parsed.task, parsed.url, parsed.execute_confirmed);
    defer plan.deinit(allocator);
    std.debug.print("{s}\n", .{plan.output});

    const browser_specs = [_]abi.features.ai.AgentWorkerSpec{
        .{ .name = "browser-planner", .instructions = "Plan safe browser steps; never access credentials.", .tool_hints = &.{ .browser, .plan } },
    };
    var agent_result = try abi.features.ai.runCustomMultiAgentWithScheduler(allocator, &sched, "agent:browser", &browser_specs, parsed.task);
    defer agent_result.deinit(allocator);
    format.printBrowserPlannerBanner(agent_result.aggregated);

    format.printSchedulerStats(sched.stats());
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
