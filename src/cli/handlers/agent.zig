const std = @import("std");
const abi = @import("../../root.zig");
const usage_mod = @import("../usage.zig");
const help = @import("agent_help.zig");

const OS_ALLOWED_COMMANDS = &.{ "true", "pwd", "ls", "whoami", "date" };

/// Wrapper that creates a PluginManager, loads bundled plugins, and dispatches
/// a plugin slash-command through its `run` method. Designed to be passed as
/// `ReplConfig.plugin_dispatch` from the TUI layer.
fn dispatchPluginCommand(allocator: std.mem.Allocator, plugin: []const u8, cmd_name: []const u8, arg: []const u8) ![]u8 {
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    abi.plugins.loadBundled(&pm);
    // Mirror `__context__:<name>`: slash-commands reach plugins as `__cmd__:<name>`
    // with an optional newline-delimited argument payload.
    if (cmd_name.len == 0) return pm.run(allocator, plugin, arg);
    const input = if (arg.len == 0)
        try std.fmt.allocPrint(allocator, "__cmd__:{s}", .{cmd_name})
    else
        try std.fmt.allocPrint(allocator, "__cmd__:{s}\n{s}", .{ cmd_name, arg });
    defer allocator.free(input);
    return pm.run(allocator, plugin, input);
}

/// `abi agent <plan|train|tui|os|multi|spawn|browser> ...`: dispatch agent subcommands.
/// `plan` runs a dry-run agent over a single input (reporting scheduler and
/// memory-tracker stats), `train` trains one or all AI profiles against the
/// durable store, `tui` launches the interactive REPL, and `os` delegates to
/// `handleAgentOs`. Returns the process exit code.
pub fn handleAgent(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi agent <plan|train|tui|os|multi|spawn|browser> ...");

    const sub_cmd = args[2];
    if (usage_mod.isHelpToken(sub_cmd)) return usage_mod.printCommandHelp("agent");
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentPlanHelp();
        return handleAgentPlan(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentTrainHelp();
        return handleAgentTrain(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentTuiHelp();
        return handleAgentTui(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "os")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentOsHelp();
        return handleAgentOs(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "multi")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentMultiHelp();
        return handleAgentMulti(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "spawn")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentSpawnHelp();
        return handleAgentSpawn(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "browser")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentBrowserHelp();
        return handleAgentBrowser(io, allocator, args);
    } else {
        return usage_mod.usageError("usage: abi agent <plan|train|tui|os|multi|spawn|browser> ...");
    }
}

fn handleAgentPlan(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent plan <input>");
    return handleAgentPlanInput(io, allocator, args[3]);
}

pub fn handleAgentPlanInput(io: std.Io, allocator: std.mem.Allocator, input: []const u8) !u8 {
    var budget = abi.features.ai.file_context.ContextBudget.init(abi.features.ai.file_context.DEFAULT_BUDGET_BYTES);
    const augmented = try abi.features.ai.file_context.resolveAndInject(io, allocator, input, ".", &budget);
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
    const s = sched.stats();
    std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
    std.debug.print("memory (tracker): peak={d}B records={d}\n", .{ mem_tracker.getPeakUsage(), mem_tracker.getRecordCount() });
    return 0;
}

fn handleAgentTrain(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent train <abbey|aviva|abi|all>");
    return handleAgentTrainProfile(io, allocator, args[3]);
}

pub fn handleAgentTrainProfile(io: std.Io, allocator: std.mem.Allocator, profile_arg: []const u8) !u8 {
    var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
    const artifact_dir = "zig-cache/agent-artifacts";
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
}

fn handleAgentTui(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3) return usage_mod.usageError("usage: abi agent tui");
    return handleAgentTuiNoArgs(io, allocator);
}

pub fn handleAgentTuiNoArgs(io: std.Io, allocator: std.mem.Allocator) !u8 {
    var session = try abi.features.wdbx.durable_store.Session.open(io, allocator);
    defer session.deinit();
    const store = session.storePtr();

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    // Build plugin command list from the Registry
    var reg = abi.registry.Registry.init(allocator);
    defer reg.deinit();
    try reg.loadPlugins();

    var plugin_cmds = std.ArrayListUnmanaged(abi.features.tui.PluginSlashCommand).empty;
    defer {
        for (plugin_cmds.items) |pc| {
            allocator.free(pc.name);
            allocator.free(pc.summary);
            allocator.free(pc.plugin);
            for (pc.aliases) |a| allocator.free(a);
            allocator.free(pc.aliases);
        }
        plugin_cmds.deinit(allocator);
    }

    {
        const plugins = try reg.snapshotPlugins(allocator);
        defer abi.registry.Registry.freePluginSnapshot(allocator, plugins);
        for (plugins) |plugin| {
            for (plugin.commands) |cmd| {
                const name = try allocator.dupe(u8, cmd.name);
                errdefer allocator.free(name);
                const summary = try allocator.dupe(u8, cmd.summary);
                errdefer allocator.free(summary);
                const plugin_name = try allocator.dupe(u8, plugin.name);
                errdefer allocator.free(plugin_name);
                const aliases = try allocator.alloc([]const u8, cmd.aliases.len);
                errdefer allocator.free(aliases);
                for (cmd.aliases, 0..) |a, j| {
                    aliases[j] = try allocator.dupe(u8, a);
                }
                try plugin_cmds.append(allocator, .{
                    .name = name,
                    .summary = summary,
                    .plugin = plugin_name,
                    .aliases = aliases,
                });
            }
        }
    }

    // Collect context snippets from plugin context providers
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    abi.plugins.loadBundled(&pm);
    const context_snippets = try pm.collectContextSnippets(allocator);
    errdefer allocator.free(context_snippets);

    var repl = abi.features.tui.ReplLoop.init(allocator, store, &sched, .{
        .plugin_commands = plugin_cmds.items,
        .plugin_dispatch = dispatchPluginCommand,
        .context_snippets = context_snippets,
    });
    defer {
        allocator.free(context_snippets);
        repl.deinit();
    }
    repl.run(io) catch |err| {
        if (err == error.FeatureDisabled) {
            std.debug.print("error: TUI feature is disabled in this build; rebuild without -Dfeat-tui=false to use `abi agent tui`\n", .{});
            return 1;
        }
        std.debug.print("error: interactive REPL failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

pub fn handleAgentMultiInput(io: std.Io, allocator: std.mem.Allocator, input: []const u8) !u8 {
    var budget = abi.features.ai.file_context.ContextBudget.init(abi.features.ai.file_context.DEFAULT_BUDGET_BYTES);
    const augmented = try abi.features.ai.file_context.resolveAndInject(io, allocator, input, ".", &budget);
    defer allocator.free(augmented);

    var sched = abi.scheduler.Scheduler.init(allocator);
    defer sched.deinit();

    var result = try abi.features.ai.runMultiAgentWithScheduler(allocator, &sched, "agent:multi", augmented);
    defer result.deinit(allocator);
    std.debug.print("{s}\n", .{result.aggregated});
    const s = sched.stats();
    std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
    return 0;
}

fn handleAgentMulti(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent multi <input>");
    return handleAgentMultiInput(io, allocator, args[3]);
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
        std.debug.print("submitted background agent tasks:\n", .{});
        for (batch.task_ids, 0..) |id, n| {
            std.debug.print("  task_id={d} worker={s}\n", .{ id, specs_slice[n].name });
        }
        try sched.runAll();
        var collected = try abi.features.ai.collectBackgroundBatch(allocator, &batch, specs_slice);
        defer collected.deinit(allocator);
        std.debug.print("{s}\n", .{collected.aggregated});
    } else {
        var result = try abi.features.ai.runCustomMultiAgentWithScheduler(allocator, &sched, "agent:spawn", specs_slice, parsed.input);
        defer result.deinit(allocator);
        std.debug.print("{s}\n", .{result.aggregated});
    }

    const s = sched.stats();
    std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
    return 0;
}

fn handleAgentSpawn(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    return handleAgentSpawnArgv(io, allocator, args);
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
    std.debug.print("\n--- local planner worker (dry-run) ---\n{s}\n", .{agent_result.aggregated});

    const s = sched.stats();
    std.debug.print("scheduler: running={d} pending={d} completed={d} failed={d}\n", .{ s.running, s.pending, s.completed, s.failed });
    return 0;
}

fn handleAgentBrowser(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    return handleAgentBrowserArgv(io, allocator, args);
}

/// `abi agent os <dry-run|execute --confirm> <cmd> [args...]`: run an OS-control
/// command request through the os_control policy gate. `execute` requires an
/// explicit `--confirm`; without it (or for `dry-run`) the command is only
/// audited, never run. Arbitrary user argv is classified `.unknown` intent — the
/// real gate is the allow-list plus workspace containment. Returns the exit code.
pub fn handleAgentOs(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 4) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    const os_cmd = args[3];
    const execute = std.mem.eql(u8, os_cmd, "execute");
    const dry_run = std.mem.eql(u8, os_cmd, "dry-run");
    if (!execute and !dry_run) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    if (execute and (args.len < 6 or !std.mem.eql(u8, args[4], "--confirm"))) {
        return usage_mod.usageError("usage: abi agent os execute --confirm <cmd> [args...]");
    }
    const start: usize = if (execute) 5 else 4;
    if (start >= args.len) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");

    const cwd_z = try std.process.currentPathAlloc(io, allocator);
    defer allocator.free(cwd_z);
    const policy = abi.features.os_control.Policy{
        .workspace_root = cwd_z,
        .dry_run_only = dry_run,
        .allow_execution = execute,
        .allowed_commands = OS_ALLOWED_COMMANDS,
        .require_confirmation = true,
    };
    const request = abi.features.os_control.CommandRequest{
        .intent = .unknown,
        .argv = args[start..],
        .cwd = cwd_z,
        .confirm_execution = execute,
    };

    if (dry_run) {
        const rendered = abi.features.os_control.renderDryRun(allocator, io, request, policy) catch |err| switch (err) {
            error.CommandDenied => {
                std.debug.print("error: command denied by os-control policy\n", .{});
                return 1;
            },
            else => return err,
        };
        defer allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    }

    const result = abi.features.os_control.executeConfirmed(allocator, io, request, policy) catch |err| switch (err) {
        error.CommandDenied => {
            std.debug.print("error: command denied by os-control policy\n", .{});
            return 1;
        },
        else => return err,
    };
    std.debug.print("{s}\n", .{result.message});
    return result.exit_code orelse 0;
}

test "agent dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "plan" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "train" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "execute", "ls" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "execute", "ls", "--confirm" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "multi" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn", "--workers" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgentBrowserArgv(t, allocator, &.{ "abi", "agent", "browser", "--execute", "open docs" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgentBrowserArgv(t, allocator, &.{ "abi", "agent", "browser" }));
}

test "agent handler help returns success before side effects" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "plan", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "train", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "tui", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "multi", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "browser", "help" }));
}

test "agent os dry-run denies commands outside the policy allow-list" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 1), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "rm" }));
}

test "agent os dry-run allow-listed command succeeds without executing" {
    // Skip when os_control is stubbed out (-Dfeat-os-control=false) or the
    // target has no trusted executable table (non-macOS/Linux): both make
    // every request deny by design.
    if (abi.features.os_control.trustedCommandSpec("pwd") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const t = std.testing.io;
    // The handler resolves the ambient cwd itself and uses it as both the
    // policy workspace root and the request cwd, so containment holds in any
    // real directory. `renderDryRun` applies the full policy gate and renders
    // the escaped plan — it never spawns a child process.
    try std.testing.expectEqual(@as(u8, 0), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "pwd" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgentOs(t, allocator, &.{ "abi", "agent", "os", "dry-run", "true" }));
}

test {
    std.testing.refAllDecls(@This());
}
