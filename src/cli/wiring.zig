//! CLI handler closures and typed argument specs — the wiring layer between
//! the declarative command table (registry.zig) and the per-command handler
//! implementations (handlers/). Extracted from registry.zig to keep the
//! registry focused on types, the frozen command table, and help rendering.

const std = @import("std");
const reg = @import("registry.zig");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers/mod.zig");
const arg = @import("arg.zig");
const env = @import("../foundation/env.zig");

const Ctx = reg.Ctx;
const Command = reg.Command;
const Arg = arg.Arg;
const Parsed = arg.Parsed;

// --- Typed argument specs ----------------------------------------------------

pub const complete_args = [_]Arg{
    .{ .name = "live", .kind = .flag, .help = "serve anthropic models over the live transport" },
    .{ .name = "confirm", .kind = .flag, .help = "confirm on-device FoundationModels execution" },
    .{ .name = "learn", .kind = .flag, .help = "run the SEA self-learning loop" },
    .{ .name = "soul", .kind = .value, .help = "soul layout JSON file for neural routing" },
    .{ .name = "soul-alpha", .kind = .value, .help = "blend weight for soul neural routing (0.0-1.0)" },
    .{ .name = "stream", .kind = .flag, .help = "stream completion tokens to stdout" },
    .{ .name = "model", .kind = .value, .help = "select a catalog model id (e.g. claude-fable-5)" },
    .{ .name = "input", .kind = .positional, .required = true, .help = "completion prompt" },
};

pub const train_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "training input" },
};

const agent_plan_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "agent planning input" },
};

const agent_train_args = [_]Arg{
    .{ .name = "profile", .kind = .positional, .required = true, .choices = &.{ "abbey", "aviva", "abi", "all" }, .help = "agent profile" },
};

const agent_multi_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "multi-agent task input" },
};

pub const agent_subcommands = [_]Command{
    .{ .name = "plan", .summary = "Run a dry-run agent planning task through the scheduler and print memory tracker statistics.", .usage = "abi agent plan <input>", .args = &agent_plan_args, .handler = agentPlanHandler },
    .{ .name = "train", .summary = "Train one known local profile or all known profiles against the durable WDBX store.", .usage = "abi agent train <abbey|aviva|abi|all>", .args = &agent_train_args, .handler = agentTrainHandler },
    .{ .name = "tui", .summary = "Launch the interactive ABI agent REPL. Non-tty input falls back to line mode.", .usage = "abi agent tui", .handler = agentTuiHandler },
    .{ .name = "multi", .summary = "Run Abbey, Aviva, and Abi concurrently via the scheduler.", .usage = "abi agent multi <input>", .args = &agent_multi_args, .handler = agentMultiHandler },
    .{ .name = "spawn", .summary = "Create custom smart-agent workers with optional background scheduler submission.", .usage = "abi agent spawn [--background] [--workers <spec>] <input>", .raw_handler = handlers.agent_mod.handleAgentSpawnArgv },
    .{ .name = "browser", .summary = "Plan browser automation locally (dry-run default; external MCP for real navigation).", .usage = "abi agent browser [--url <url>] [--execute --confirm] <task>", .raw_handler = handlers.agent_mod.handleAgentBrowserArgv },
    .{ .name = "os", .summary = "Audit an OS-control command request. execute requires --confirm and the policy allow-list.", .usage = "abi agent os <dry-run|execute --confirm> <cmd> [args...]", .raw_handler = handlers.agent_mod.handleAgentOs },
};

pub const plugin_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "list", "run" }, .help = "plugin subcommand" },
    .{ .name = "name", .kind = .positional, .help = "plugin name for run" },
    .{ .name = "input", .kind = .positional, .greedy = true, .help = "optional plugin input" },
};

pub const plugin_subcommands = [_]Command{
    .{ .name = "list", .summary = "Print the generated plugin registry with each installed plugin module.", .usage = "abi plugin list" },
    .{ .name = "run", .summary = "Run a bundled plugin by registry name with optional text input.", .usage = "abi plugin run <name> [input]" },
};

pub const auth_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "signin", "logout", "status" }, .help = "auth subcommand" },
    .{ .name = "service", .kind = .positional, .help = "service for signin" },
};

pub const auth_subcommands = [_]Command{
    .{ .name = "status", .summary = "Show which local connector credentials are configured.", .usage = "abi auth status" },
    .{ .name = "logout", .summary = "Remove the local ABI credential file when present.", .usage = "abi auth logout" },
    .{ .name = "signin", .summary = "Prompt for a credential and persist it in the local ABI credential file.", .usage = "abi auth signin <openai|anthropic|discord|grok|twilio>" },
};

pub const scheduler_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"status"}, .help = "scheduler subcommand" },
};

pub const twilio_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"simulate"}, .help = "twilio subcommand" },
    .{ .name = "input", .kind = .positional, .required = true, .help = "simulation input" },
};

pub const dashboard_args = [_]Arg{
    .{ .name = "pane", .kind = .value, .choices = &.{ "1", "2", "3", "4", "5", "system", "plugins", "storage", "wdbx", "scheduler", "memory" }, .help = "initial diagnostics pane" },
    .{ .name = "plain", .kind = .flag, .help = "render without ANSI color/style escapes" },
    .{ .name = "no-color", .kind = .flag, .help = "alias for --plain" },
    .{ .name = "compact", .kind = .flag, .help = "render only the selected diagnostics pane" },
    .{ .name = "once", .kind = .flag, .help = "force one-shot output even when stdin is a terminal" },
    .{ .name = "interval", .kind = .value, .value_kind = .uint, .help = "interactive refresh interval in milliseconds (100-60000)" },
    .{ .name = "json", .kind = .flag, .help = "emit one machine-readable JSON dashboard snapshot" },
    .{ .name = "list-panes", .kind = .flag, .help = "print dashboard pane names, titles, and hotkeys" },
};

pub const dashboard_usage = "usage: abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]";

pub const nn_args = [_]Arg{
    .{ .name = "command", .kind = .positional, .required = true, .choices = &.{ "train", "sample" }, .help = "nn subcommand" },
    .{ .name = "jsonl", .kind = .value, .help = "JSONL training dataset path" },
    .{ .name = "field", .kind = .value, .help = "JSONL text field (default: text)" },
    .{ .name = "text", .kind = .value, .help = "sample corpus text" },
    .{ .name = "seed", .kind = .value, .help = "sample seed character" },
    .{ .name = "n", .kind = .value, .value_kind = .uint, .help = "sample character count" },
    .{ .name = "input", .kind = .positional, .help = "inline training text" },
};

pub const nn_subcommands = [_]Command{
    .{ .name = "train", .summary = "Train the miniature local character model from inline text or a JSONL text field.", .usage = "abi nn train \"<text>\" | train --jsonl <path> [--field <name>]" },
    .{ .name = "sample", .summary = "Train on <corpus>, then greedily emit k characters from the seed byte.", .usage = "abi nn sample --text \"<corpus>\" --seed <char> --n <k>" },
};

pub const wdbx_subcommands = [_]Command{
    .{ .name = "db", .summary = "Manage segment checkpoints, WAL recovery, and snapshot integrity.", .usage = "abi wdbx db <init|verify|compact> <path> [keep]" },
    .{ .name = "block", .summary = "Append or inspect SHA-linked conversation blocks in a WDBX checkpoint.", .usage = "abi wdbx block <insert|get> <path> ..." },
    .{ .name = "query", .summary = "Print store stats or run semantic/persona-scoped retrieval over a recovered store.", .usage = "abi wdbx query <path> [text] [persona]" },
    .{ .name = "benchmark", .summary = "Measure local insert/search timing for the in-process vector store.", .usage = "abi wdbx benchmark [count]" },
    .{ .name = "cluster", .summary = "Run single-node status, in-process consensus demo, or authenticated cluster RPC serving.", .usage = "abi wdbx cluster status | cluster demo [nodes] | cluster serve <port> [node] [host]" },
    .{ .name = "compute", .summary = "Report CPU/GPU/NPU/TPU backend selection and fallback state.", .usage = "abi wdbx compute info" },
    .{ .name = "secure", .summary = "Demonstrate local compression plus reference homomorphic aggregation; not security-audited FHE.", .usage = "abi wdbx secure demo" },
    .{ .name = "gpu", .summary = "Report GPU backend capability and native-kernel status.", .usage = "abi wdbx gpu info" },
    .{ .name = "api", .summary = "Serve the loopback WDBX REST API; bearer token via " ++ env.WDBX_REST_TOKEN_ENV ++ "; TLS via ABI_WDBX_TLS_CERT/KEY (proxy-terminated).", .usage = "abi wdbx api serve [port]" },
};

// --- Typed handlers ----------------------------------------------------------
// Thin adapters that forward parsed arguments to the existing handler functions.
// Handler output text is preserved verbatim (never re-authored).

/// Parse and validate a `--soul-alpha` token at the CLI edge. Returns null for
/// a malformed number or a value outside the documented 0.0-1.0 blend range;
/// callers surface null as a usage error (exit 2) per the malformed-numeric-arg
/// contract instead of silently substituting the default.
pub fn parseSoulAlpha(token: []const u8) ?f32 {
    const alpha = std.fmt.parseFloat(f32, token) catch return null;
    if (std.math.isNan(alpha) or alpha < 0.0 or alpha > 1.0) return null;
    return alpha;
}

pub fn completeHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    const soul_alpha: f32 = if (parsed.value("soul-alpha")) |s|
        parseSoulAlpha(s) orelse return usage_mod.usageError("--soul-alpha must be a number between 0.0 and 1.0")
    else
        0.5;
    return handlers.handleComplete(
        ctx.io,
        ctx.allocator,
        .{
            .input = parsed.value("input").?,
            .model = parsed.value("model"),
            .live = parsed.flag("live"),
            .confirmed = parsed.flag("confirm"),
            .learn = parsed.flag("learn"),
            .stream = parsed.flag("stream"),
            .soul = parsed.value("soul"),
            .soul_alpha = soul_alpha,
        },
    );
}

pub fn trainHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.handleTrain(ctx.allocator, parsed.value("input").?);
}

fn agentPlanHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.agent_mod.handleAgentPlanInput(ctx.io, ctx.allocator, parsed.value("input").?);
}

fn agentTrainHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.agent_mod.handleAgentTrainProfile(ctx.io, ctx.allocator, parsed.value("profile").?);
}

fn agentMultiHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.agent_mod.handleAgentMultiInput(ctx.io, ctx.allocator, parsed.value("input").?);
}

fn agentTuiHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed;
    return handlers.agent_mod.handleAgentTuiNoArgs(ctx.io, ctx.allocator);
}

pub fn backendsHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = ctx;
    _ = parsed;
    return handlers.handleBackends();
}

pub fn pluginHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
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

pub fn authHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
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

pub fn dashboardHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
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

pub fn schedulerHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed.value("command").?;
    return handlers.handleSchedulerStatus(ctx.allocator);
}

pub fn twilioHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed.value("command").?;
    return handlers.twilio_mod.handleTwilioSimulate(ctx.allocator, parsed.value("input").?);
}

pub fn nnHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
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

test "parseSoulAlpha accepts in-range blend weights" {
    try std.testing.expectEqual(@as(f32, 0.0), parseSoulAlpha("0.0").?);
    try std.testing.expectEqual(@as(f32, 0.5), parseSoulAlpha("0.5").?);
    try std.testing.expectEqual(@as(f32, 1.0), parseSoulAlpha("1").?);
}

test "parseSoulAlpha rejects malformed and out-of-range tokens" {
    try std.testing.expect(parseSoulAlpha("abc") == null);
    try std.testing.expect(parseSoulAlpha("") == null);
    try std.testing.expect(parseSoulAlpha("nan") == null);
    try std.testing.expect(parseSoulAlpha("inf") == null);
    try std.testing.expect(parseSoulAlpha("-0.1") == null);
    try std.testing.expect(parseSoulAlpha("1.5") == null);
}

test {
    std.testing.refAllDecls(@This());
}
