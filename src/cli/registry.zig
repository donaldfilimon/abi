//! Declarative CLI command registry — the framework backbone for the `abi`
//! command surface.
//!
//! The frozen command *metadata* (name / usage / summary) is owned by the
//! `std`-only `usage.zig` module so the `cli_usage` build module (which is wired
//! without the `abi`/`build_options` imports) can keep projecting the help
//! surface from it. This module imports that metadata and augments each command
//! with its argument spec and handler, producing the single table that
//! `dispatch.zig` walks. Handler *invocation* lives here because this module is
//! only ever compiled inside the CLI executable graph (which has the handler
//! imports), never inside the standalone `cli_usage` module.

const std = @import("std");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers/mod.zig");
const arg = @import("arg.zig");

pub const completion = @import("completion.zig");
pub const help_json = @import("help_json.zig");
pub const CompletionShell = completion.CompletionShell;
pub const writeShellCompletion = completion.writeShellCompletion;
pub const printShellCompletion = completion.printShellCompletion;
pub const writeHelpJson = help_json.writeHelpJson;
pub const printHelpJson = help_json.printHelpJson;

pub const Arg = arg.Arg;
pub const Parsed = arg.Parsed;

/// Execution context handed to typed handlers.
pub const Ctx = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
};

/// Legacy handler shape: receives the raw argv slice. Matches the historical
/// `CommandHandler` signature so existing handler functions wire in unchanged.
pub const RawHandler = *const fn (std.Io, std.mem.Allocator, []const []const u8) anyerror!u8;

/// Typed handler shape: receives the parsed argument set. Used by commands
/// migrated onto the generic argument parser (`arg.zig`).
pub const Handler = *const fn (Ctx, Parsed) anyerror!u8;

pub const Command = struct {
    name: []const u8,
    summary: []const u8,
    usage: []const u8,
    args: []const Arg = &.{},
    subcommands: []const Command = &.{},
    handler: ?Handler = null,
    raw_handler: ?RawHandler = null,
};

pub const Shortcut = struct {
    token: []const u8,
    command: []const u8,
    summary: []const u8,
};

pub const shortcuts = [_]Shortcut{
    .{
        .token = "--tui",
        .command = "tui",
        .summary = "Legacy top-level diagnostics dashboard shortcut; accepts the same flags as `abi tui`.",
    },
};

pub fn commandNameForShortcut(token: []const u8) ?[]const u8 {
    for (shortcuts) |shortcut| {
        if (std.mem.eql(u8, shortcut.token, token)) return shortcut.command;
    }
    return null;
}

pub fn parseCompletionShell(token: []const u8) ?CompletionShell {
    if (std.mem.eql(u8, token, "bash")) return .bash;
    if (std.mem.eql(u8, token, "zsh")) return .zsh;
    if (std.mem.eql(u8, token, "fish")) return .fish;
    return null;
}

/// Look up the frozen metadata for `name` in the `usage` table at comptime so
/// the registry never re-declares the contract-tested name/usage/summary text.
fn meta(comptime name: []const u8) usage_mod.Command {
    inline for (usage_mod.commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    @compileError("registry: no usage metadata for command '" ++ name ++ "'");
}

/// Metadata-only command (e.g. `help`, which `dispatch` intercepts before the
/// table walk).
fn metaCmd(comptime name: []const u8) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage };
}

/// Command wired to a raw `(io, allocator, argv)` handler.
fn rawCmd(comptime name: []const u8, handler: RawHandler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .raw_handler = handler };
}

fn rawCmdWithSubcommands(comptime name: []const u8, subcommands: []const Command, handler: RawHandler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .subcommands = subcommands, .raw_handler = handler };
}

/// Command wired to a typed handler driven by the generic argument parser.
fn typedCmd(comptime name: []const u8, args: []const Arg, handler: Handler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .args = args, .handler = handler };
}

fn typedCmdWithSubcommands(comptime name: []const u8, args: []const Arg, subcommands: []const Command, handler: Handler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .args = args, .subcommands = subcommands, .handler = handler };
}

fn subcommandCmd(comptime name: []const u8, subcommands: []const Command) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .subcommands = subcommands };
}

// --- Typed argument specs ----------------------------------------------------

const complete_args = [_]Arg{
    .{ .name = "live", .kind = .flag, .help = "serve anthropic models over the live transport" },
    .{ .name = "confirm", .kind = .flag, .help = "confirm on-device FoundationModels execution" },
    .{ .name = "learn", .kind = .flag, .help = "run the SEA self-learning loop" },
    .{ .name = "model", .kind = .value, .help = "select a catalog model id (e.g. claude-fable-5)" },
    .{ .name = "input", .kind = .positional, .required = true, .help = "completion prompt" },
};

const train_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "training input" },
};

const agent_plan_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "agent planning input" },
};

const agent_train_args = [_]Arg{
    .{ .name = "profile", .kind = .positional, .required = true, .choices = &.{ "abbey", "aviva", "abi", "all" }, .help = "agent profile" },
};

const agent_subcommands = [_]Command{
    .{ .name = "plan", .summary = "Run a dry-run agent planning task through the scheduler and print memory tracker statistics.", .usage = "abi agent plan <input>", .args = &agent_plan_args, .handler = agentPlanHandler },
    .{ .name = "train", .summary = "Train one known local profile or all known profiles against the durable WDBX store.", .usage = "abi agent train <abbey|aviva|abi|all>", .args = &agent_train_args, .handler = agentTrainHandler },
    .{ .name = "tui", .summary = "Launch the interactive ABI agent REPL. Non-tty input falls back to line mode.", .usage = "abi agent tui", .handler = agentTuiHandler },
    .{ .name = "os", .summary = "Audit an OS-control command request. execute requires --confirm and the policy allow-list.", .usage = "abi agent os <dry-run|execute --confirm> <cmd> [args...]", .raw_handler = handlers.agent_mod.handleAgentOs },
};

const plugin_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "list", "run" }, .help = "plugin subcommand" },
    .{ .name = "name", .kind = .positional, .help = "plugin name for run" },
    .{ .name = "input", .kind = .positional, .greedy = true, .help = "optional plugin input" },
};

const plugin_subcommands = [_]Command{
    .{ .name = "list", .summary = "Print the generated plugin registry with each installed plugin module.", .usage = "abi plugin list" },
    .{ .name = "run", .summary = "Run a bundled plugin by registry name with optional text input.", .usage = "abi plugin run <name> [input]" },
};

const auth_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "signin", "logout", "status" }, .help = "auth subcommand" },
    .{ .name = "service", .kind = .positional, .help = "service for signin" },
};

const auth_subcommands = [_]Command{
    .{ .name = "status", .summary = "Show which local connector credentials are configured.", .usage = "abi auth status" },
    .{ .name = "logout", .summary = "Remove the local ABI credential file when present.", .usage = "abi auth logout" },
    .{ .name = "signin", .summary = "Prompt for a credential and persist it in the local ABI credential file.", .usage = "abi auth signin <openai|anthropic|discord|grok|twilio>" },
};

const scheduler_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"status"}, .help = "scheduler subcommand" },
};

const twilio_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"simulate"}, .help = "twilio subcommand" },
    .{ .name = "input", .kind = .positional, .required = true, .help = "simulation input" },
};

const dashboard_args = [_]Arg{
    .{ .name = "pane", .kind = .value, .choices = &.{ "1", "2", "3", "4", "5", "system", "plugins", "storage", "wdbx", "scheduler", "memory" }, .help = "initial diagnostics pane" },
    .{ .name = "plain", .kind = .flag, .help = "render without ANSI color/style escapes" },
    .{ .name = "no-color", .kind = .flag, .help = "alias for --plain" },
    .{ .name = "compact", .kind = .flag, .help = "render only the selected diagnostics pane" },
    .{ .name = "once", .kind = .flag, .help = "force one-shot output even when stdin is a terminal" },
    .{ .name = "interval", .kind = .value, .value_kind = .uint, .help = "interactive refresh interval in milliseconds (100-60000)" },
    .{ .name = "json", .kind = .flag, .help = "emit one machine-readable JSON dashboard snapshot" },
    .{ .name = "list-panes", .kind = .flag, .help = "print dashboard pane names, titles, and hotkeys" },
};

const dashboard_usage = "usage: abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]";

const nn_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "train", "sample" }, .help = "nn subcommand" },
    .{ .name = "jsonl", .kind = .value, .help = "JSONL training dataset path" },
    .{ .name = "field", .kind = .value, .help = "JSONL text field (default: text)" },
    .{ .name = "text", .kind = .value, .help = "sample corpus text" },
    .{ .name = "seed", .kind = .value, .help = "sample seed character" },
    .{ .name = "n", .kind = .value, .value_kind = .uint, .help = "sample character count" },
    .{ .name = "input", .kind = .positional, .help = "inline training text" },
};

const nn_subcommands = [_]Command{
    .{ .name = "train", .summary = "Train the miniature local character model from inline text or a JSONL text field.", .usage = "abi nn train \"<text>\" | train --jsonl <path> [--field <name>]" },
    .{ .name = "sample", .summary = "Train on <corpus>, then greedily emit k characters from the seed byte.", .usage = "abi nn sample --text \"<corpus>\" --seed <char> --n <k>" },
};

const wdbx_subcommands = [_]Command{
    .{ .name = "db", .summary = "Manage segment checkpoints, WAL recovery, and snapshot integrity.", .usage = "abi wdbx db <init|verify|compact> <path> [keep]" },
    .{ .name = "block", .summary = "Append or inspect SHA-linked conversation blocks in a WDBX checkpoint.", .usage = "abi wdbx block <insert|get> <path> ..." },
    .{ .name = "query", .summary = "Print store stats or run semantic/persona-scoped retrieval over a recovered store.", .usage = "abi wdbx query <path> [text] [persona]" },
    .{ .name = "benchmark", .summary = "Measure local insert/search timing for the in-process vector store.", .usage = "abi wdbx benchmark [count]" },
    .{ .name = "cluster", .summary = "Run single-node status, in-process consensus demo, or authenticated cluster RPC serving.", .usage = "abi wdbx cluster status | cluster demo [nodes] | cluster serve <port> [node] [host]" },
    .{ .name = "compute", .summary = "Report CPU/GPU/NPU/TPU backend selection and fallback state.", .usage = "abi wdbx compute info" },
    .{ .name = "secure", .summary = "Demonstrate local compression plus reference homomorphic aggregation; not security-audited FHE.", .usage = "abi wdbx secure demo" },
    .{ .name = "gpu", .summary = "Report GPU backend capability and native-kernel status.", .usage = "abi wdbx gpu info" },
    .{ .name = "api", .summary = "Serve the loopback WDBX REST API; optional bearer token via ABI_WDBX_REST_TOKEN.", .usage = "abi wdbx api serve [port]" },
};

// --- Typed handlers ----------------------------------------------------------
// Thin adapters that forward parsed arguments to the existing handler functions.
// Handler output text is preserved verbatim (never re-authored).

fn completeHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.handleComplete(
        ctx.io,
        ctx.allocator,
        .{
            .input = parsed.value("input").?,
            .model = parsed.value("model"),
            .live = parsed.flag("live"),
            .confirmed = parsed.flag("confirm"),
            .learn = parsed.flag("learn"),
        },
    );
}

fn trainHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.handleTrain(ctx.allocator, parsed.value("input").?);
}

fn agentPlanHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.agent_mod.handleAgentPlanInput(ctx.allocator, parsed.value("input").?);
}

fn agentTrainHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.agent_mod.handleAgentTrainProfile(ctx.io, ctx.allocator, parsed.value("profile").?);
}

fn agentTuiHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed;
    return handlers.agent_mod.handleAgentTuiNoArgs(ctx.io, ctx.allocator);
}

fn backendsHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = ctx;
    _ = parsed;
    return handlers.handleBackends();
}

fn pluginHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    const command = parsed.value("command").?;
    const name = parsed.value("name");
    const input = parsed.value("input") orelse "";

    if (std.mem.eql(u8, command, "list")) {
        if (name != null or input.len != 0) return usage_mod.usageError("usage: abi plugin list");
        return handlers.plugin_mod.handlePluginList(ctx.allocator);
    }

    if (std.mem.eql(u8, command, "run")) {
        const plugin_name = name orelse return usage_mod.usageError("usage: abi plugin run <name> [input]");
        return handlers.plugin_mod.handlePluginRun(ctx.allocator, plugin_name, input);
    }

    return usage_mod.usageError("usage: abi plugin list | run <name> [input]");
}

fn authHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    const command = parsed.value("command").?;
    const service = parsed.value("service");

    if (std.mem.eql(u8, command, "status")) {
        if (service != null) return usage_mod.usageError("usage: abi auth status");
        return handlers.auth_mod.handleAuthStatus(ctx.allocator);
    }

    if (std.mem.eql(u8, command, "logout")) {
        if (service != null) return usage_mod.usageError("usage: abi auth logout");
        return handlers.auth_mod.handleAuthLogout(ctx.allocator);
    }

    if (std.mem.eql(u8, command, "signin")) {
        const provider = service orelse return usage_mod.usageError("usage: abi auth signin <openai|anthropic|discord|grok|twilio>");
        return handlers.auth_mod.handleAuthSignin(ctx.io, ctx.allocator, provider);
    }

    return usage_mod.usageError("usage: abi auth <signin|logout|status>");
}

fn dashboardHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    const initial_pane = if (parsed.value("pane")) |pane|
        handlers.dashboard_mod.dashboardPaneIndexForToken(pane) orelse return usage_mod.usageError(dashboard_usage)
    else
        0;
    const color = !(parsed.flag("plain") or parsed.flag("no-color"));
    const refresh_interval_ms = if (parsed.uint("interval")) |raw|
        handlers.dashboard_mod.validRefreshInterval(raw) orelse return usage_mod.usageError(dashboard_usage)
    else
        handlers.dashboard_mod.DEFAULT_REFRESH_INTERVAL_MS;
    return handlers.dashboard_mod.handleDashboardWithOptions(ctx.allocator, .{
        .initial_pane = initial_pane,
        .color = color,
        .compact = parsed.flag("compact"),
        .force_one_shot = parsed.flag("once"),
        .refresh_interval_ms = refresh_interval_ms,
        .format = if (parsed.flag("json")) .json else .text,
        .list_panes = parsed.flag("list-panes"),
    });
}

fn schedulerHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed.value("command").?;
    return handlers.handleSchedulerStatus(ctx.allocator);
}

fn twilioHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed.value("command").?;
    return handlers.twilio_mod.handleTwilioSimulate(ctx.allocator, parsed.value("input").?);
}

fn nnHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    const command = parsed.value("command").?;
    const input = parsed.value("input");
    const jsonl_path = parsed.value("jsonl");
    const field = parsed.value("field") orelse "text";
    const text = parsed.value("text");
    const seed = parsed.value("seed");
    const n = if (parsed.uint("n")) |raw| std.math.cast(usize, raw) orelse return usage_mod.usageError("usage: abi nn sample --text \"<corpus>\" --seed <char> --n <k>") else 16;

    if (std.mem.eql(u8, command, "train")) {
        if (text != null or seed != null or parsed.value("n") != null) {
            return usage_mod.usageError("usage: abi nn train \"<text>\" | train --jsonl <path> [--field <name>]");
        }
        if (input == null and jsonl_path == null) {
            return usage_mod.usageError("usage: abi nn train \"<text>\" | train --jsonl <path> [--field <name>]");
        }
        return handlers.nn_mod.handleNnTrain(ctx.allocator, input, jsonl_path, field);
    }

    if (std.mem.eql(u8, command, "sample")) {
        if (input != null or jsonl_path != null or parsed.value("field") != null) {
            return usage_mod.usageError("usage: abi nn sample --text \"<corpus>\" --seed <char> --n <k>");
        }
        if (text == null or seed == null) {
            return usage_mod.usageError("usage: abi nn sample --text \"<corpus>\" --seed <char> --n <k>");
        }
        return handlers.nn_mod.handleNnSample(ctx.allocator, text, seed, n);
    }

    return usage_mod.usageError("usage: abi nn <command> ...");
}

fn printChoices(choices: []const []const u8) void {
    if (choices.len == 0) return;
    std.debug.print(" choices=", .{});
    for (choices, 0..) |choice, idx| {
        if (idx != 0) std.debug.print("|", .{});
        std.debug.print("{s}", .{choice});
    }
}

fn printArgSpec(args: []const Arg) void {
    if (args.len == 0) return;
    std.debug.print("\nArguments:\n", .{});
    for (args) |a| {
        switch (a.kind) {
            .flag => {
                std.debug.print("  --{s:<18} {s}", .{ a.name, a.help });
                printChoices(a.choices);
                std.debug.print("\n", .{});
            },
            .value => {
                std.debug.print("  --{s} <value>       {s}", .{ a.name, a.help });
                printChoices(a.choices);
                std.debug.print("\n", .{});
            },
            .positional => {
                std.debug.print("  <{s}>             {s}{s}", .{ a.name, a.help, if (a.required) " (required)" else "" });
                printChoices(a.choices);
                std.debug.print("\n", .{});
            },
        }
    }
}

fn printExamples(examples: []const []const u8) void {
    if (examples.len == 0) return;
    std.debug.print("\nExamples:\n", .{});
    for (examples) |example| {
        std.debug.print("  {s}\n", .{example});
    }
}

fn printSubcommands(subcommands: []const Command) void {
    if (subcommands.len == 0) return;
    std.debug.print("\nSubcommands:\n", .{});
    for (subcommands) |subcommand| {
        std.debug.print("  {s:<12} {s}\n", .{ subcommand.name, subcommand.usage });
        std.debug.print("               {s}\n", .{subcommand.summary});
    }
}

pub fn printCommandHelp(name: []const u8) ?u8 {
    for (commands) |command| {
        if (!std.mem.eql(u8, command.name, name)) continue;
        const usage_meta = usage_mod.findCommand(name) orelse return null;
        std.debug.print("{s}\n\n{s}\n", .{ usage_meta.usage, usage_meta.summary });
        if (usage_meta.details.len > 0) {
            std.debug.print("\n{s}\n", .{usage_meta.details});
        }
        printSubcommands(command.subcommands);
        printArgSpec(command.args);
        printExamples(usage_meta.examples);
        return 0;
    }
    return null;
}

pub fn printCommandUsageHelp(name: []const u8) ?u8 {
    const usage_meta = usage_mod.findCommand(name) orelse return null;
    std.debug.print("usage: {s}\n", .{usage_meta.usage});
    if (usage_meta.details.len > 0) {
        std.debug.print("\n{s}\n", .{usage_meta.details});
    } else {
        std.debug.print("\n{s}\n", .{usage_meta.summary});
    }
    return 0;
}

pub fn printSubcommandUsageHelp(command: Command, subcommand_name: []const u8) ?u8 {
    for (command.subcommands) |subcommand| {
        if (!std.mem.eql(u8, subcommand.name, subcommand_name)) continue;
        std.debug.print("usage: {s}\n\n{s}\n", .{ subcommand.usage, subcommand.summary });
        return 0;
    }
    return null;
}

pub fn findRegistryCommand(name: []const u8) ?Command {
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    return null;
}

// --- Raw handler shims -------------------------------------------------------
// Commands still on the legacy `(io, allocator, argv)` contract. Thin adapters
// drop unused parameters and forward to the existing handler functions,
// reproducing the historical dispatch wrappers exactly.

/// The frozen 13-command surface, in the same order as `usage.commands`.
/// `help` is metadata-only; `dispatch` intercepts it before the table walk.
pub const commands = [_]Command{
    metaCmd("help"),
    typedCmd("complete", &complete_args, completeHandler),
    typedCmd("train", &train_args, trainHandler),
    subcommandCmd("agent", &agent_subcommands),
    typedCmd("backends", &.{}, backendsHandler),
    typedCmdWithSubcommands("plugin", &plugin_args, &plugin_subcommands, pluginHandler),
    typedCmdWithSubcommands("auth", &auth_args, &auth_subcommands, authHandler),
    typedCmd("twilio", &twilio_args, twilioHandler),
    typedCmd("tui", &dashboard_args, dashboardHandler),
    typedCmd("dashboard", &dashboard_args, dashboardHandler),
    rawCmdWithSubcommands("wdbx", &wdbx_subcommands, handlers.handleWdbx),
    typedCmd("scheduler", &scheduler_args, schedulerHandler),
    typedCmdWithSubcommands("nn", &nn_args, &nn_subcommands, nnHandler),
};

test {
    std.testing.refAllDecls(@This());
}

test "registry help renders typed args" {
    try std.testing.expectEqual(@as(?u8, 0), printCommandHelp("complete"));
    try std.testing.expect(printCommandHelp("nope") == null);
}

test "registry help json emits parseable command metadata" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };
    try std.testing.expect(try writeHelpJson(&writer, std.testing.allocator, null, null));

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("abi.cli.help", root.get("type").?.string);
    const shortcut_items = root.get("shortcuts").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), shortcut_items.len);
    const shortcut = shortcut_items[0].object;
    try std.testing.expectEqualStrings("--tui", shortcut.get("token").?.string);
    try std.testing.expectEqualStrings("tui", shortcut.get("command").?.string);
    const completion_field = root.get("completion").?.object;
    try std.testing.expectEqualStrings("abi help --completion <bash|zsh|fish>", completion_field.get("usage").?.string);
    const shells = completion_field.get("shells").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), shells.len);
    try std.testing.expectEqualStrings("bash", shells[0].string);
    try std.testing.expectEqualStrings("zsh", shells[1].string);
    try std.testing.expectEqualStrings("fish", shells[2].string);
    try std.testing.expectEqual(commands.len, root.get("commands").?.array.items.len);
    const first = root.get("commands").?.array.items[0].object;
    try std.testing.expectEqualStrings("help", first.get("name").?.string);
    try std.testing.expectEqualStrings("core", first.get("category").?.string);
}

test "registry help json emits focused subcommand metadata" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };
    try std.testing.expect(try writeHelpJson(&writer, std.testing.allocator, "wdbx", "cluster"));

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("wdbx", root.get("command").?.string);
    const sub = root.get("subcommand").?.object;
    try std.testing.expectEqualStrings("cluster", sub.get("name").?.string);
    try std.testing.expect(std.mem.indexOf(u8, sub.get("usage").?.string, "cluster serve") != null);
}

test "registry help json resolves and reports command shortcuts" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };
    try std.testing.expect(try writeHelpJson(&writer, std.testing.allocator, "--tui", null));

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, buf.items, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("tui", root.get("command").?.string);
    const shortcut_items = root.get("shortcuts").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), shortcut_items.len);
    const shortcut = shortcut_items[0].object;
    try std.testing.expectEqualStrings("--tui", shortcut.get("token").?.string);
}

test "registry shell completions expose commands shortcuts and typed flags" {
    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }
    };

    var bash_buf = std.ArrayListUnmanaged(u8).empty;
    defer bash_buf.deinit(std.testing.allocator);
    var bash_writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &bash_buf };
    try writeShellCompletion(&bash_writer, std.testing.allocator, .bash);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "_abi_complete") != null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "help complete train agent backends") != null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "tui|--tui)") != null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "--list-panes") != null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "words=\"list run \"") != null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "words=\"list run list run ") == null);
    try std.testing.expect(std.mem.indexOf(u8, bash_buf.items, "words=\"status logout signin signin ") == null);

    var zsh_buf = std.ArrayListUnmanaged(u8).empty;
    defer zsh_buf.deinit(std.testing.allocator);
    var zsh_writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &zsh_buf };
    try writeShellCompletion(&zsh_writer, std.testing.allocator, .zsh);
    try std.testing.expect(std.mem.indexOf(u8, zsh_buf.items, "#compdef abi") != null);
    try std.testing.expect(std.mem.indexOf(u8, zsh_buf.items, "compadd -- --pane") != null);
    try std.testing.expect(std.mem.indexOf(u8, zsh_buf.items, "bash zsh fish") != null);
    try std.testing.expect(std.mem.indexOf(u8, zsh_buf.items, "compadd -- list run \n") != null);
    try std.testing.expect(std.mem.indexOf(u8, zsh_buf.items, "compadd -- list run list run ") == null);

    var fish_buf = std.ArrayListUnmanaged(u8).empty;
    defer fish_buf.deinit(std.testing.allocator);
    var fish_writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &fish_buf };
    try writeShellCompletion(&fish_writer, std.testing.allocator, .fish);
    try std.testing.expect(std.mem.indexOf(u8, fish_buf.items, "complete -c abi -f") != null);
    try std.testing.expect(std.mem.indexOf(u8, fish_buf.items, "__fish_seen_subcommand_from tui --tui") != null);
    try std.testing.expect(std.mem.indexOf(u8, fish_buf.items, "-l list-panes") != null);
    try std.testing.expect(std.mem.indexOf(u8, fish_buf.items, "' -a 'list run '\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, fish_buf.items, "' -a 'list run list run ") == null);
}

test "registry command metadata stays in usage order" {
    try std.testing.expectEqual(usage_mod.commands.len, commands.len);
    for (usage_mod.commands, commands) |usage_command, command| {
        try std.testing.expectEqualStrings(usage_command.name, command.name);
        try std.testing.expectEqualStrings(usage_command.summary, command.summary);
        try std.testing.expectEqualStrings(usage_command.usage, command.usage);
    }
}

test "registry commands have exactly one dispatch path except metadata-only help" {
    for (commands) |command| {
        const has_typed = command.handler != null;
        const has_raw = command.raw_handler != null;
        if (std.mem.eql(u8, command.name, "help")) {
            try std.testing.expect(!has_typed and !has_raw);
        } else if (command.subcommands.len != 0 and !has_typed and !has_raw) {
            try std.testing.expect(command.args.len == 0);
        } else {
            try std.testing.expect(has_typed != has_raw);
        }
    }
}
