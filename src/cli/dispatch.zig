//! CLI dispatch — walks the declarative `registry.commands` table.
//!
//! `help`/`--help`/`-h` are intercepted before the table walk (they have
//! command-specific behavior); every other command is matched by name and
//! invoked through either its typed `handler` (the argument spec is parsed via
//! `arg.parse`, and a parse failure emits the command's frozen `.usage` string
//! with exit code 2) or its legacy `raw_handler`.

const std = @import("std");
const usage_mod = @import("usage.zig");
const registry = @import("registry.zig");
const arg = @import("arg.zig");
const handlers_mod = @import("handlers/mod.zig");

const MAX_SUGGESTION_LEN = 64;

const Suggestion = struct {
    value: []const u8,
    leading_dashes: bool = false,
};

fn choiceContains(choices: []const []const u8, token: []const u8) bool {
    for (choices) |choice| {
        if (std.mem.eql(u8, choice, token)) return true;
    }
    return false;
}

fn isChoiceSubcommandHelp(command: registry.Command, args: []const []const u8) bool {
    if (command.handler == null or args.len != 4 or !usage_mod.isHelpToken(args[3])) return false;
    if (command.args.len == 0) return false;
    const first = command.args[0];
    return first.kind == .positional and choiceContains(first.choices, args[2]);
}

fn findCommand(name: []const u8) ?registry.Command {
    for (registry.commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    return null;
}

fn findSubcommand(command: registry.Command, name: []const u8) ?registry.Command {
    for (command.subcommands) |subcommand| {
        if (std.mem.eql(u8, subcommand.name, name)) return subcommand;
    }
    return null;
}

fn optionFor(spec: []const arg.Arg, name: []const u8) ?arg.Arg {
    for (spec) |a| {
        if (a.kind == .positional) continue;
        if (std.mem.eql(u8, a.name, name)) return a;
    }
    return null;
}

fn nextPositionalSpec(spec: []const arg.Arg, used: []const bool) ?usize {
    for (spec, 0..) |a, idx| {
        if (a.kind == .positional and !used[idx]) return idx;
    }
    return null;
}

fn editDistance(a_raw: []const u8, b_raw: []const u8) usize {
    if (a_raw.len > MAX_SUGGESTION_LEN or b_raw.len > MAX_SUGGESTION_LEN) return MAX_SUGGESTION_LEN + 1;

    var previous: [MAX_SUGGESTION_LEN + 1]usize = undefined;
    var current: [MAX_SUGGESTION_LEN + 1]usize = undefined;

    for (0..b_raw.len + 1) |idx| previous[idx] = idx;

    for (a_raw, 0..) |a_ch, a_idx| {
        current[0] = a_idx + 1;
        for (b_raw, 0..) |b_ch, b_idx| {
            const cost: usize = if (std.ascii.toLower(a_ch) == std.ascii.toLower(b_ch)) 0 else 1;
            const deletion = previous[b_idx + 1] + 1;
            const insertion = current[b_idx] + 1;
            const substitution = previous[b_idx] + cost;
            current[b_idx + 1] = @min(@min(deletion, insertion), substitution);
        }
        @memcpy(previous[0 .. b_raw.len + 1], current[0 .. b_raw.len + 1]);
    }

    return previous[b_raw.len];
}

fn suggestionThreshold(token_len: usize) usize {
    if (token_len <= 3) return 1;
    if (token_len <= 8) return 2;
    return 3;
}

fn bestSuggestion(token: []const u8, candidates: []const []const u8) ?[]const u8 {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;

    for (candidates) |candidate| {
        if (candidate.len == 0) continue;
        const score = editDistance(token, candidate);
        if (score < best_score) {
            best_score = score;
            best = candidate;
        }
    }

    if (best) |candidate| {
        if (best_score <= suggestionThreshold(@max(token.len, candidate.len))) return candidate;
    }
    return null;
}

fn suggestCommand(name: []const u8) ?Suggestion {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;
    for (registry.commands) |command| {
        const score = editDistance(name, command.name);
        if (score < best_score) {
            best_score = score;
            best = command.name;
        }
    }
    if (best) |value| {
        if (best_score <= suggestionThreshold(@max(name.len, value.len))) return .{ .value = value };
    }
    return null;
}

fn suggestSubcommand(command: registry.Command, name: []const u8) ?Suggestion {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;
    for (command.subcommands) |subcommand| {
        const score = editDistance(name, subcommand.name);
        if (score < best_score) {
            best_score = score;
            best = subcommand.name;
        }
    }
    if (best) |value| {
        if (best_score <= suggestionThreshold(@max(name.len, value.len))) return .{ .value = value };
    }
    return null;
}

fn suggestForArgs(spec: []const arg.Arg, argv: []const []const u8, start_index: usize) ?Suggestion {
    var used_positionals = std.mem.zeroes([32]bool);
    if (spec.len > used_positionals.len) return null;

    var i: usize = start_index;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (std.mem.eql(u8, token, "--")) break;
        if (std.mem.startsWith(u8, token, "--")) {
            const name = token[2..];
            const option = optionFor(spec, name) orelse {
                var best: ?[]const u8 = null;
                var best_score: usize = MAX_SUGGESTION_LEN + 1;
                for (spec) |a| {
                    if (a.kind == .positional) continue;
                    const score = editDistance(name, a.name);
                    if (score < best_score) {
                        best_score = score;
                        best = a.name;
                    }
                }
                if (best) |value| {
                    if (best_score <= suggestionThreshold(@max(name.len, value.len))) {
                        return .{ .value = value, .leading_dashes = true };
                    }
                }
                return null;
            };

            if (option.kind == .value) {
                i += 1;
                if (i >= argv.len) return null;
                if (option.choices.len != 0 and !choiceContains(option.choices, argv[i])) {
                    if (bestSuggestion(argv[i], option.choices)) |value| return .{ .value = value };
                }
            }
            continue;
        }

        const pos_idx = nextPositionalSpec(spec, used_positionals[0..spec.len]) orelse return null;
        used_positionals[pos_idx] = true;
        const positional = spec[pos_idx];
        if (positional.choices.len != 0 and !choiceContains(positional.choices, token)) {
            if (bestSuggestion(token, positional.choices)) |value| return .{ .value = value };
        }
        if (positional.greedy) break;
    }
    return null;
}

fn printHint(suggestion: Suggestion) void {
    if (suggestion.leading_dashes) {
        std.debug.print("hint: did you mean `--{s}`?\n", .{suggestion.value});
    } else {
        std.debug.print("hint: did you mean `{s}`?\n", .{suggestion.value});
    }
}

fn usageErrorWithSuggestion(message: []const u8, suggestion: ?Suggestion) u8 {
    std.debug.print("error: {s}\n", .{message});
    if (suggestion) |candidate| printHint(candidate);
    return 2;
}

fn unknownCommandError(cmd: []const u8) u8 {
    std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
    if (suggestCommand(cmd)) |suggestion| printHint(suggestion);
    usage_mod.printUsage();
    return 2;
}

const HelpRequest = struct {
    json: bool = false,
    completion: ?registry.CompletionShell = null,
    names: [2][]const u8 = undefined,
    count: usize = 0,

    fn command(self: HelpRequest) ?[]const u8 {
        return if (self.count >= 1) self.names[0] else null;
    }

    fn subcommand(self: HelpRequest) ?[]const u8 {
        return if (self.count >= 2) self.names[1] else null;
    }
};

const help_usage = "usage: abi help [--json|--completion <bash|zsh|fish>] [command] [subcommand]";

fn parseHelpRequest(args: []const []const u8, start_index: usize) error{Usage}!HelpRequest {
    var request = HelpRequest{};
    var i: usize = start_index;
    while (i < args.len) : (i += 1) {
        const token = args[i];
        if (std.mem.eql(u8, token, "--json")) {
            if (request.json or request.completion != null) return error.Usage;
            request.json = true;
            continue;
        }
        if (std.mem.eql(u8, token, "--completion")) {
            if (request.json or request.completion != null) return error.Usage;
            i += 1;
            if (i >= args.len) return error.Usage;
            request.completion = registry.parseCompletionShell(args[i]) orelse return error.Usage;
            continue;
        }
        if (request.count >= request.names.len) return error.Usage;
        request.names[request.count] = token;
        request.count += 1;
    }
    return request;
}

fn runHelpRequest(allocator: std.mem.Allocator, request: HelpRequest) !u8 {
    if (request.completion) |shell| {
        if (request.count != 0) return usage_mod.usageError(help_usage);
        return registry.printShellCompletion(allocator, shell);
    }

    if (request.json) {
        if (try registry.printHelpJson(allocator, request.command(), request.subcommand())) |exit_code| return exit_code;
        if (request.command()) |command_name| {
            const resolved_name = registry.commandNameForShortcut(command_name) orelse command_name;
            if (findCommand(resolved_name)) |command| {
                if (request.subcommand()) |subcommand_name| {
                    return usageErrorWithSuggestion(command.usage, suggestSubcommand(command, subcommand_name));
                }
            }
            return unknownCommandError(command_name);
        }
        return usage_mod.usageError(help_usage);
    }

    switch (request.count) {
        0 => {
            usage_mod.printUsage();
            return 0;
        },
        1 => {
            const command_name = registry.commandNameForShortcut(request.names[0]) orelse request.names[0];
            if (registry.printCommandHelp(command_name)) |exit_code| return exit_code;
            if (usage_mod.findCommand(command_name) != null) return usage_mod.printCommandHelp(command_name);
            return unknownCommandError(command_name);
        },
        2 => {
            const command_name = registry.commandNameForShortcut(request.names[0]) orelse request.names[0];
            const subcommand_name = request.names[1];
            if (findCommand(command_name)) |command| {
                return registry.printSubcommandUsageHelp(command, subcommand_name) orelse usageErrorWithSuggestion(command.usage, suggestSubcommand(command, subcommand_name));
            }
            return unknownCommandError(command_name);
        },
        else => return usage_mod.usageError(help_usage),
    }
}

pub fn runCli(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 2) {
        // Redesigned default: launch interactive TUI dashboard (no subcommand = interactive mode)
        return handlers_mod.renderTui(allocator);
    }

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        const help_request = parseHelpRequest(args, 2) catch return usage_mod.usageError(help_usage);
        return try runHelpRequest(allocator, help_request);
    }

    for (registry.commands) |command| {
        if (!std.mem.eql(u8, cmd, command.name)) continue;

        if (args.len == 3 and usage_mod.isHelpToken(args[2]) and (command.handler != null or command.subcommands.len != 0)) {
            return registry.printCommandHelp(command.name) orelse usage_mod.printCommandHelp(command.name);
        }

        if (command.handler == null and command.raw_handler == null and command.subcommands.len != 0 and args.len >= 3) {
            if (findSubcommand(command, args[2])) |subcommand| {
                if (args.len == 4 and usage_mod.isHelpToken(args[3])) {
                    if (registry.printSubcommandUsageHelp(command, args[2])) |exit_code| return exit_code;
                }
                if (subcommand.handler) |handler| {
                    var parsed = arg.parseFrom(allocator, subcommand.args, args, 3) catch |err| switch (err) {
                        error.Usage => return usage_mod.usageError(subcommand.usage),
                        else => |e| return e,
                    };
                    defer parsed.deinit();
                    return handler(.{ .io = io, .allocator = allocator }, parsed);
                }
                if (subcommand.raw_handler) |raw_handler| {
                    return raw_handler(io, allocator, args);
                }
                return registry.printSubcommandUsageHelp(command, args[2]) orelse usage_mod.usageError(command.usage);
            }
            return usageErrorWithSuggestion(command.usage, suggestSubcommand(command, args[2]));
        }

        if (isChoiceSubcommandHelp(command, args)) {
            if (registry.printSubcommandUsageHelp(command, args[2])) |exit_code| return exit_code;
            return registry.printCommandUsageHelp(command.name) orelse usage_mod.printCommandHelp(command.name);
        }

        if (command.handler) |handler| {
            var parsed = arg.parse(allocator, command.args, args) catch |err| switch (err) {
                error.Usage => return usageErrorWithSuggestion(command.usage, suggestForArgs(command.args, args, 2)),
                else => |e| return e,
            };
            defer parsed.deinit();
            return handler(.{ .io = io, .allocator = allocator }, parsed);
        }
        if (command.raw_handler) |raw_handler| {
            return raw_handler(io, allocator, args);
        }
        // Metadata-only command (e.g. `help`) reaching the table walk: fall
        // through to the unknown-command path below.
        break;
    }

    return unknownCommandError(cmd);
}

test "runCli intercepts help, accepts no-args, rejects unknown + malformed grammar" {
    const t = std.testing.io;
    const a = std.testing.allocator;
    // No args launches the dashboard fallback in non-tty tests; help prints usage.
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{"abi"}));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "complete" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "agent", "plan" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "plugin", "run" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "wdbx", "cluster" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--json" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--completion", "bash" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--completion", "zsh" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--completion", "fish" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--json", "dashboard" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--json", "--tui" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "--tui" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "help", "wdbx", "cluster", "--json" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "--help", "--json" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "complete", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "scheduler", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "scheduler", "status", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "twilio", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "twilio", "simulate", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "plugin", "list", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "plugin", "run", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "auth", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "auth", "status", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "auth", "logout", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "auth", "signin", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "agent", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "agent", "plan", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "agent", "train", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "agent", "tui", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "agent", "os", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "nn", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "nn", "train", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "nn", "sample", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--pane", "memory" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "tui", "--pane", "wdbx" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--plain" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "tui", "--no-color", "--pane", "scheduler" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--compact", "--pane", "scheduler" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--once", "--interval", "250" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--json" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "dashboard", "--list-panes" }));
    try std.testing.expectEqual(@as(u8, 0), try runCli(t, a, &.{ "abi", "tui", "--json", "--list-panes" }));
    // Unknown command exits 2.
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "boguscommand" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "complte" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "complete", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "agent", "plan", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--json", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--json", "--json" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--completion" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--completion", "powershell" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--completion", "bash", "dashboard" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "help", "--json", "--completion", "bash" }));
    // A typed command missing its required positional fails parsing (before the
    // handler runs) and maps to the frozen usage string with exit 2.
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "complete" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "scheduler", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "plugin", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "plugin", "list", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "plugin", "run" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "auth", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "auth", "status", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "auth", "signin" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "auth", "signin", "notaservice" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "dashboard", "--pane", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "tui", "--pane" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "dashboard", "--plain", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "dashboard", "--interval", "99" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "dashboard", "--interval", "60001" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "plan" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "plan", "a", "b" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "train" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "train", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "tui", "extra" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "os" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "agent", "os", "execute", "ls" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "nn", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "nn", "train" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "nn", "sample" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "nn", "sample", "--text", "hello" }));
    try std.testing.expectEqual(@as(u8, 2), try runCli(t, a, &.{ "abi", "nn", "sample", "--text", "hello", "--seed", "h", "--n", "nope" }));
}

test {
    std.testing.refAllDecls(@This());
}

test "suggestions choose close commands, subcommands, flags, and choices" {
    const command = findCommand("agent") orelse return error.MissingCommand;
    const dashboard = findCommand("dashboard") orelse return error.MissingCommand;

    try std.testing.expectEqualStrings("complete", (suggestCommand("complte") orelse return error.MissingSuggestion).value);
    try std.testing.expectEqualStrings("train", (suggestSubcommand(command, "trian") orelse return error.MissingSuggestion).value);

    const pane_suggestion = suggestForArgs(dashboard.args, &.{ "abi", "dashboard", "--pane", "memry" }, 2) orelse return error.MissingSuggestion;
    try std.testing.expectEqualStrings("memory", pane_suggestion.value);
    try std.testing.expect(!pane_suggestion.leading_dashes);

    const flag_suggestion = suggestForArgs(dashboard.args, &.{ "abi", "dashboard", "--plian" }, 2) orelse return error.MissingSuggestion;
    try std.testing.expectEqualStrings("plain", flag_suggestion.value);
    try std.testing.expect(flag_suggestion.leading_dashes);
}
