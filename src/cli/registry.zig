//! Declarative CLI command registry — the framework backbone for the `abi`
//! command surface.
//!
//! The frozen command *metadata* (name / usage / summary) is owned by the
//! `std`-only `usage.zig` module so the `cli_usage` build module (which is wired
//! without the `abi`/`build_options` imports) can keep projecting the help
//! surface from it. This module imports that metadata and augments each command
//! with its argument spec and handler, producing the single table that
//! `dispatch.zig` walks. Handler *invocation* lives in `wiring.zig` (extracted
//! to keep this module focused on types, the frozen table, and help rendering).

const std = @import("std");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers/mod.zig");
const arg = @import("arg.zig");
const wiring = @import("wiring.zig");

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

/// The frozen 13-command surface, in the same order as `usage.commands`.
/// `help` is metadata-only; `dispatch` intercepts it before the table walk.
/// Arg specs and handler closures live in `wiring.zig`.
pub const commands = [_]Command{
    metaCmd("help"),
    typedCmd("complete", &wiring.complete_args, wiring.completeHandler),
    typedCmd("train", &wiring.train_args, wiring.trainHandler),
    subcommandCmd("agent", &wiring.agent_subcommands),
    typedCmd("backends", &.{}, wiring.backendsHandler),
    typedCmdWithSubcommands("plugin", &wiring.plugin_args, &wiring.plugin_subcommands, wiring.pluginHandler),
    typedCmdWithSubcommands("auth", &wiring.auth_args, &wiring.auth_subcommands, wiring.authHandler),
    typedCmd("twilio", &wiring.twilio_args, wiring.twilioHandler),
    typedCmd("tui", &wiring.dashboard_args, wiring.dashboardHandler),
    typedCmd("dashboard", &wiring.dashboard_args, wiring.dashboardHandler),
    rawCmdWithSubcommands("wdbx", &wiring.wdbx_subcommands, handlers.handleWdbx),
    typedCmd("scheduler", &wiring.scheduler_args, wiring.schedulerHandler),
    typedCmdWithSubcommands("nn", &wiring.nn_args, &wiring.nn_subcommands, wiring.nnHandler),
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
