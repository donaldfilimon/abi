//! CLI command metadata table — the frozen, contract-tested set of 13 top-level
//! commands. This module is intentionally std-only so it can be tested in
//! isolation and stays independent of the feature graph. The command list here
//! is asserted by `tests/contracts/surface.zig`.
//!
//! Do NOT add commands here without also updating the handler wiring in
//! `registry.zig` and the contract assertion in `surface.zig`.
const std = @import("std");

pub const Command = struct {
    name: []const u8,
    usage: []const u8,
    summary: []const u8,
    category: Category,
    details: []const u8 = "",
    examples: []const []const u8 = &.{},
};

pub const Category = enum {
    core,
    ai,
    data,
    system,
    network,
};

const category_order = [_]Category{ .core, .ai, .data, .system, .network };

fn categoryTitle(category: Category) []const u8 {
    return switch (category) {
        .core => "Core",
        .ai => "AI",
        .data => "Data",
        .system => "System",
        .network => "Network",
    };
}

pub fn categoryName(category: Category) []const u8 {
    return switch (category) {
        .core => "core",
        .ai => "ai",
        .data => "data",
        .system => "system",
        .network => "network",
    };
}

pub const commands = [_]Command{
    .{
        .name = "help",
        .usage = "abi help [--json|--completion <bash|zsh|fish>] [command] [subcommand]",
        .summary = "Show top-level, command, subcommand, shortcut, JSON, or shell-completion help",
        .category = .core,
        .details = "--json prints typed command/subcommand metadata plus top-level shortcut and completion-shell metadata for automation and docs tooling. --completion emits a shell completion script for bash, zsh, or fish.",
        .examples = &.{
            "abi help --json",
            "abi help --completion bash",
            "abi help complete",
            "abi help --tui",
            "abi help wdbx cluster",
            "abi help --json dashboard",
        },
    },
    .{
        .name = "complete",
        .usage = "abi complete [--live] [--confirm] [--learn] [--stream] [--model <id>] <input>",
        .summary = "Run completion through local ABI agent pipeline; --model selects catalog id, --live enables explicit live transport",
        .category = .ai,
        .details = "--model selects a catalog id, --live enables explicit live transport, --learn runs the SEA self-learning loop, and apple-fm requires --confirm. Use `--` before a prompt that starts with `-`.",
        .examples = &.{
            "abi complete \"summarize repository status\"",
            "abi complete -- --literal-leading-dash",
            "abi complete --learn --model claude-fable-5 \"plan next repair\"",
        },
    },
    .{
        .name = "train",
        .usage = "abi train <input>",
        .summary = "Run the AI pipeline compatibility trainer",
        .category = .ai,
        .examples = &.{"abi train \"local routing note\""},
    },
    .{
        .name = "agent",
        .usage = "abi agent <plan|train|tui|multi|spawn|browser|os> ...",
        .summary = "Agent planning, training, TUI, multi-worker orchestration, browser planning, OS-control",
        .category = .ai,
        .details = "Subcommands: plan, train, tui, multi, spawn (custom workers), browser (local orchestration), os dry-run/execute.",
        .examples = &.{
            "abi agent plan \"inspect WDBX persistence\"",
            "abi agent tui",
            "abi agent multi \"coordinate release\"",
            "abi agent spawn --workers \"scout|Explore pages|explore,browser\" \"task\"",
            "abi agent browser --url https://example.com \"open docs\"",
            "abi agent os dry-run ls",
        },
    },
    .{
        .name = "backends",
        .usage = "abi backends",
        .summary = "Show GPU, accelerator, shader, MLIR status, feature flags, and version info",
        .category = .system,
        .details = "Reports truthful linked/native/fallback status for local compute backends, lists all build-time feature flags with their enabled/disabled status, and shows framework version and build info.",
        .examples = &.{"abi backends"},
    },
    .{
        .name = "plugin",
        .usage = "abi plugin list | run <name> [input]",
        .summary = "Inspect or execute installed plugins",
        .category = .data,
        .details = "The list path reads the generated plugin registry; run loads bundled plugins and dispatches by plugin name.",
        .examples = &.{
            "abi plugin list",
            "abi plugin run hello \"input\"",
        },
    },
    .{
        .name = "auth",
        .usage = "abi auth <signin|logout|status>",
        .summary = "Manage external-service authentication state",
        .category = .system,
        .details = "Credential helpers are local; live connectors still require explicit credentials and live transport selection.",
        .examples = &.{"abi auth status"},
    },
    .{
        .name = "twilio",
        .usage = "abi twilio simulate <input>",
        .summary = "Run a local Twilio voice-agent simulation",
        .category = .network,
        .details = "Offline simulation only; live Twilio transport requires explicit connector credentials and live mode elsewhere.",
        .examples = &.{"abi twilio simulate \"billing question\""},
    },
    .{
        .name = "tui",
        .usage = "abi tui [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]",
        .summary = "Render the diagnostics dashboard",
        .category = .system,
        .details = "Launches the interactive diagnostics dashboard when stdin is a terminal; otherwise renders a one-shot dashboard frame. --pane accepts system, plugins, storage/wdbx, scheduler, memory, or 1-5. --plain/--no-color omit ANSI styling for logs. --compact renders only the selected pane. --once forces one-shot output; --interval sets the interactive refresh cadence in milliseconds. --json emits one machine-readable snapshot. --list-panes prints pane names, titles, and hotkeys without launching the dashboard.",
        .examples = &.{
            "abi tui",
            "abi tui --pane memory",
            "abi tui --compact --pane scheduler",
            "abi tui --plain --once",
            "abi tui --interval 250",
            "abi tui --json",
            "abi tui --list-panes",
        },
    },
    .{
        .name = "dashboard",
        .usage = "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]",
        .summary = "Render the diagnostics dashboard",
        .category = .system,
        .details = "Alias of `abi tui` for the diagnostics dashboard. --pane selects the initially highlighted diagnostics pane; --plain/--no-color omit ANSI styling. --compact renders only the selected pane. --once forces one-shot output; --interval sets the interactive refresh cadence in milliseconds. --json emits one machine-readable snapshot. --list-panes prints pane names, titles, and hotkeys without launching the dashboard.",
        .examples = &.{
            "abi dashboard",
            "abi dashboard --pane scheduler",
            "abi dashboard --compact --pane scheduler",
            "abi dashboard --plain --once",
            "abi dashboard --interval 250",
            "abi dashboard --json",
            "abi dashboard --list-panes",
        },
    },
    .{
        .name = "wdbx",
        .usage = "abi wdbx <db|block|query|benchmark|cluster|compute|secure|gpu|api> ...",
        .summary = "Operate WDBX storage, WAL, blocks, stats, and demos",
        .category = .data,
        .details = "Subcommands: db init|verify|compact, block insert|get, query, benchmark, cluster status|demo|serve, compute info, secure demo, gpu info, api serve.",
        .examples = &.{
            "abi wdbx db verify",
            "abi wdbx query \"local memory\"",
        },
    },
    .{
        .name = "scheduler",
        .usage = "abi scheduler status",
        .summary = "Report scheduler and memory tracker status",
        .category = .system,
        .details = "Runs a one-shot scheduler status task and reports attached memory tracker counters.",
        .examples = &.{"abi scheduler status"},
    },
    .{
        .name = "nn",
        .usage = "abi nn train \"<text>\" | train --jsonl <path> [--field <name>] | sample --text \"<corpus>\" --seed <char> --n <k>",
        .summary = "Run the miniature character-level demo trainer",
        .category = .ai,
        .details = "This is a small local demo trainer, not a production/LLM/distributed trainer.",
        .examples = &.{"abi nn sample --text \"hello\" --seed h --n 16"},
    },
};

pub fn printUsage() void {
    std.debug.print("Usage: abi <command> [args...]\n       abi --tui [dashboard flags]\n\n", .{});
    for (category_order) |category| {
        std.debug.print("\x1b[1m{s}\x1b[0m\n", .{categoryTitle(category)});
        for (commands) |command| {
            if (command.category == category) {
                std.debug.print("  \x1b[36m{s:<12}\x1b[0m {s}\n", .{ command.name, command.summary });
            }
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("Run `abi help [--json|--completion <shell>] <command> [subcommand]` for details.\n", .{});
}

pub fn findCommand(name: []const u8) ?Command {
    for (commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    return null;
}

pub fn isHelpToken(token: []const u8) bool {
    return std.mem.eql(u8, token, "help") or std.mem.eql(u8, token, "--help") or std.mem.eql(u8, token, "-h");
}

pub fn printCommandHelp(name: []const u8) u8 {
    if (findCommand(name)) |command| {
        std.debug.print("{s}\n\n{s}\n", .{ command.usage, command.summary });
        if (command.details.len > 0) {
            std.debug.print("\n{s}\n", .{command.details});
        }
        if (command.examples.len > 0) {
            std.debug.print("\nExamples:\n", .{});
            for (command.examples) |example| {
                std.debug.print("  {s}\n", .{example});
            }
        }
        return 0;
    }
    std.debug.print("error: unknown command '{s}'\n\n", .{name});
    printUsage();
    return 2;
}

pub fn usageError(message: []const u8) u8 {
    std.debug.print("error: {s}\n", .{message});
    return 2;
}

test "usage: known command returns exit 0" {
    try std.testing.expectEqual(@as(u8, 0), printCommandHelp("help"));
    try std.testing.expectEqual(@as(u8, 0), printCommandHelp("wdbx"));
}

test "usage: findCommand locates metadata" {
    const complete = findCommand("complete") orelse return error.MissingCommand;
    try std.testing.expect(std.mem.indexOf(u8, complete.summary, "--model") != null);
    try std.testing.expect(findCommand("nope") == null);
    try std.testing.expectEqualStrings("ai", categoryName(complete.category));
}

test "usage: help token classifier" {
    try std.testing.expect(isHelpToken("help"));
    try std.testing.expect(isHelpToken("--help"));
    try std.testing.expect(isHelpToken("-h"));
    try std.testing.expect(!isHelpToken("--helper"));
}

test "usage: unknown command returns exit 2" {
    try std.testing.expectEqual(@as(u8, 2), printCommandHelp("nope"));
}

test "usage: usageError returns exit 2" {
    try std.testing.expectEqual(@as(u8, 2), usageError("bad arg"));
}

test {
    std.testing.refAllDecls(@This());
}
