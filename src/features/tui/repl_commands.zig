//! Slash-command parsing, completion, and status formatting for the REPL.
//!
//! Extracted from `repl.zig` so the pure, unit-testable command classification
//! and formatting helpers live separately from the interactive `ReplLoop`. The
//! loop owns IO, store, scheduler, and dispatch; this module owns the
//! declarative slash-command table and the pure string helpers.
//!
//! Git/diff/commit pure helpers (colorizeDiff, diffArgv, commitArgvFor, etc.)
//! live in the sibling `repl_git_helpers.zig` and are re-exported here.

const std = @import("std");
const builtin = @import("builtin");
const models = @import("../ai/models.zig");
const repl_types = @import("repl_types.zig");
const utils = @import("../../foundation/utils.zig");
const git_helpers = @import("repl_git_helpers.zig");

// Re-export pure git/diff/commit helpers from the extracted leaf
pub const colorizeDiff = git_helpers.colorizeDiff;
pub const diffArgv = git_helpers.diffArgv;
pub const diffWantsStat = git_helpers.diffWantsStat;
pub const commitAddArgv = git_helpers.commitAddArgv;
pub const commitArgvFor = git_helpers.commitArgvFor;
pub const CommitMessageOutcome = git_helpers.CommitMessageOutcome;
pub const accumulateCommitMessage = git_helpers.accumulateCommitMessage;
pub const homeEnvVarName = git_helpers.homeEnvVarName;
pub const syncClisLauncherPath = git_helpers.syncClisLauncherPath;

/// Maximum length of a model id settable via `/model`.
pub const MODEL_STORAGE_BYTES = 128;

pub const SpecialCommand = enum { quit, reset, help, model, profile, status, history, context, syncclis, open, diff, commit, features, learn, live, save, load, sessions, clear, pane, unknown };

pub const SlashCommand = struct {
    kind: SpecialCommand,
    name: []const u8,
    aliases: []const []const u8 = &.{},
    summary: []const u8,
};

pub const slash_commands = [_]SlashCommand{
    .{ .kind = .help, .name = "help", .aliases = &.{"h"}, .summary = "Show this help" },
    .{ .kind = .model, .name = "model", .summary = "Switch the completion model (alias-resolved)" },
    .{ .kind = .profile, .name = "profile", .summary = "Show profile routing status" },
    .{ .kind = .status, .name = "status", .aliases = &.{"stat"}, .summary = "Show session, model, and persistence status" },
    .{ .kind = .history, .name = "history", .aliases = &.{"hist"}, .summary = "Show recent session turns and persisted blocks" },
    .{ .kind = .reset, .name = "reset", .summary = "Reset the turn counter and start a fresh session" },
    .{ .kind = .context, .name = "context", .summary = "Show current context state (open file, turn history)" },
    .{ .kind = .syncclis, .name = "sync-clis", .aliases = &.{"syncclis"}, .summary = "Sync skills/plugins/commands/experiences across CLIs" },
    .{ .kind = .open, .name = "open", .summary = "Read a file into the prompt context" },
    .{ .kind = .diff, .name = "diff", .summary = "Show git diff (use --stat for summary)" },
    .{ .kind = .commit, .name = "commit", .summary = "Stage all changes and create a commit" },
    .{ .kind = .quit, .name = "quit", .aliases = &.{ "q", "exit" }, .summary = "Exit the REPL" },
    .{ .kind = .features, .name = "features", .aliases = &.{"feat"}, .summary = "Show active build-time features" },
    .{ .kind = .learn, .name = "learn", .summary = "Toggle SEA self-learning mode on/off" },
    .{ .kind = .live, .name = "live", .summary = "Toggle live Anthropic transport (requires credentials)" },
    .{ .kind = .save, .name = "save", .summary = "Save session context to ~/.abi/sessions/<name>.json" },
    .{ .kind = .load, .name = "load", .summary = "Restore session context from ~/.abi/sessions/<name>.json" },
    .{ .kind = .sessions, .name = "sessions", .aliases = &.{"ls-sessions"}, .summary = "List saved sessions in ~/.abi/sessions/" },
    .{ .kind = .clear, .name = "clear", .aliases = &.{"cls"}, .summary = "Clear the terminal screen" },
    .{ .kind = .pane, .name = "pane", .summary = "Toggle split view (chat left, git diff --stat right)" },
};

pub const SlashCompletion = union(enum) {
    none,
    unique: []const u8,
    ambiguous: []const *const SlashCommand,
};

fn firstWhitespace(input: []const u8) ?usize {
    for (input, 0..) |byte, idx| {
        if (std.ascii.isWhitespace(byte)) return idx;
    }
    return null;
}

fn matchesSlashCommand(def: SlashCommand, token: []const u8) bool {
    if (std.mem.eql(u8, token, def.name)) return true;
    for (def.aliases) |alias| {
        if (std.mem.eql(u8, token, alias)) return true;
    }
    return false;
}

/// Classify a line as a slash-command. Non-slash lines (ordinary prompts) and
/// unrecognized slash-commands both map to `.unknown`; callers distinguish the
/// two by checking for a leading `/`.
pub fn parseSpecialCommand(line: []const u8) SpecialCommand {
    if (line.len == 0 or line[0] != '/') return .unknown;
    const body = line[1..];
    const end = firstWhitespace(body) orelse body.len;
    const cmd = body[0..end];
    for (slash_commands) |def| {
        if (matchesSlashCommand(def, cmd)) return def.kind;
    }
    return .unknown;
}

/// Return the trimmed argument following a slash-command token, or "" if none.
pub fn specialArg(line: []const u8) []const u8 {
    const sp = firstWhitespace(line) orelse return "";
    return std.mem.trim(u8, line[sp + 1 ..], " \t\r");
}

/// Find canonical slash-command names for a partially typed command token.
/// Completion is intentionally unavailable after whitespace, so prompts and
/// `/model` arguments always retain their literal input behavior.
pub fn completeSlashCommand(input: []const u8, matches: *[slash_commands.len]*const SlashCommand) SlashCompletion {
    if (input.len < 2 or input[0] != '/' or firstWhitespace(input) != null) return .none;
    const prefix = input[1..];
    var count: usize = 0;
    for (&slash_commands) |*def| {
        var matched = std.mem.startsWith(u8, def.name, prefix);
        for (def.aliases) |alias| matched = matched or std.mem.startsWith(u8, alias, prefix);
        if (matched) {
            matches[count] = def;
            count += 1;
        }
    }
    return switch (count) {
        0 => .none,
        1 => .{ .unique = matches[0].name },
        else => .{ .ambiguous = matches[0..count] },
    };
}

/// Format a one-line `/history` summary header into `buf`. Pure (no IO/store) so
/// it is unit-testable; falls back to a fixed string if formatting overflows.
pub fn formatHistoryHeader(buf: []u8, turn_count: usize, block_count: usize) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "history: {d} turn(s) this session, {d} persisted block(s)",
        .{ turn_count, block_count },
    ) catch "history: summary unavailable";
}

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

pub fn validModelId(id: []const u8) bool {
    if (id.len == 0 or id.len > MODEL_STORAGE_BYTES) return false;
    for (id) |byte| {
        if (byte < 0x21 or byte >= 0x7f or std.ascii.isWhitespace(byte)) return false;
    }
    return true;
}

/// Format a one-line `/status` summary into `buf`. Pure so contracts can verify
/// the operator-facing status fields without constructing a live store.
pub fn formatStatusLine(
    buf: []u8,
    session_id: i64,
    turn_count: usize,
    model: []const u8,
    store_turns: bool,
    block_count: usize,
) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "status: session_id={d} turns={d} model={s} provider={s} store_turns={s} persisted_blocks={d}",
        .{
            session_id,
            turn_count,
            model,
            models.providerOf(model).label(),
            boolText(store_turns),
            block_count,
        },
    ) catch "status: summary unavailable";
}

/// A slash-command registered by a plugin at runtime. Mirrors the registry's
/// PluginCommand but lives outside the registry module so repl_commands stays
/// free of IO/registry dependencies.
pub const PluginSlashCommand = struct {
    name: []const u8,
    summary: []const u8,
    plugin: []const u8,
    aliases: []const []const u8 = &.{},
};

/// Check whether a command token (the text after `/`) matches any plugin
/// command by name or alias. Returns the matched PluginSlashCommand or null.
pub fn matchPluginCommandToken(token: []const u8, plugin_cmds: []const PluginSlashCommand) ?PluginSlashCommand {
    for (plugin_cmds) |cmd| {
        if (std.mem.eql(u8, token, cmd.name)) return cmd;
        for (cmd.aliases) |alias| {
            if (std.mem.eql(u8, token, alias)) return cmd;
        }
    }
    return null;
}

/// Print plugin-registered commands as a help section.
pub fn printPluginHelp(plugin_cmds: []const PluginSlashCommand) void {
    if (plugin_cmds.len == 0) return;
    std.debug.print("\nPlugin commands:\n", .{});
    for (plugin_cmds) |cmd| {
        if (cmd.aliases.len > 0) {
            var alias_buf: [256]u8 = undefined;
            var alias_idx: usize = 0;
            for (cmd.aliases, 0..) |alias, ai| {
                if (ai > 0) {
                    alias_buf[alias_idx] = ',';
                    alias_idx += 1;
                    alias_buf[alias_idx] = ' ';
                    alias_idx += 1;
                }
                @memcpy(alias_buf[alias_idx..][0..alias.len], alias);
                alias_idx += alias.len;
            }
            const alias_str = alias_buf[0..alias_idx];
            std.debug.print("  /{s} ({s})  {s} [{s}]\n", .{ cmd.name, alias_str, cmd.summary, cmd.plugin });
        } else {
            std.debug.print("  /{s:<14} {s} [{s}]\n", .{ cmd.name, cmd.summary, cmd.plugin });
        }
    }
}

/// Format a one-line `/open` status into `buf`, showing the current file context.
pub fn formatOpenStatus(buf: []u8, path: []const u8, bytes: usize) []const u8 {
    if (path.len == 0) return "open: no file loaded";
    return std.fmt.bufPrint(buf, "open: {s} ({d} bytes loaded into context)", .{ path, bytes }) catch "open: status unavailable";
}

/// Format a multi-line `/context` status string. Allocates the result; caller
/// owns the returned slice. Shows open file (if any), turn/history counts, and
/// a preview of the accumulated turn history.
pub fn formatContextStatus(allocator: std.mem.Allocator, open_path: []const u8, open_content: []const u8, turn_count: usize, history_count: usize, turn_history_preview: []const u8) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    if (open_path.len > 0) {
        try buf.appendSlice(allocator, "context: open file: ");
        try buf.appendSlice(allocator, open_path);
        try buf.appendSlice(allocator, " (");
        {
            var num_buf: [32]u8 = undefined;
            const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{open_content.len});
            try buf.appendSlice(allocator, num_str);
        }
        try buf.appendSlice(allocator, " bytes)\n");
    } else {
        try buf.appendSlice(allocator, "context: no file loaded\n");
    }

    {
        try buf.appendSlice(allocator, "context: ");
        {
            var num_buf: [32]u8 = undefined;
            const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{turn_count});
            try buf.appendSlice(allocator, num_str);
        }
        try buf.appendSlice(allocator, " turn(s) this session, ");
        {
            var num_buf: [32]u8 = undefined;
            const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{history_count});
            try buf.appendSlice(allocator, num_str);
        }
        try buf.appendSlice(allocator, " history entr(ies)\n");
    }

    if (turn_history_preview.len > 0) {
        const preview_len = @min(turn_history_preview.len, @as(usize, 200));
        try buf.appendSlice(allocator, "context: history preview (");
        {
            var num_buf: [32]u8 = undefined;
            const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{preview_len});
            try buf.appendSlice(allocator, num_str);
        }
        try buf.appendSlice(allocator, " chars):\n");
        try buf.appendSlice(allocator, turn_history_preview[0..preview_len]);
        try buf.appendSlice(allocator, "\n");
    } else {
        try buf.appendSlice(allocator, "context: (no turn history)\n");
    }

    return buf.toOwnedSlice(allocator);
}

pub fn printHelp() void {
    printHelpWithPlugins(&.{});
}

pub fn printHelpWithPlugins(plugin_cmds: []const PluginSlashCommand) void {
    std.debug.print("Commands:\n", .{});
    for (slash_commands) |def| {
        std.debug.print("  /{s:<14} {s}\n", .{ def.name, def.summary });
    }
    printPluginHelp(plugin_cmds);
    std.debug.print("  <text>           Run a completion and persist the turn\n\n", .{});
}

/// Format the stub acknowledgment printed when no plugin dispatch callback is
/// configured. Returns an owned slice; caller frees. Pure: no IO.
pub fn formatPluginCommandAck(allocator: std.mem.Allocator, cmd: PluginSlashCommand, arg: []const u8) ![]u8 {
    if (arg.len > 0) {
        return std.fmt.allocPrint(allocator, "[plugin:{s}] /{s} '{s}'", .{ cmd.plugin, cmd.name, arg });
    } else {
        return std.fmt.allocPrint(allocator, "[plugin:{s}] /{s}", .{ cmd.plugin, cmd.name });
    }
}

/// Format the turn-history ring buffer as a `user:` / `assistant:` preview.
/// Pure: no IO, no store. Caller frees the returned slice.
pub fn formatTurnHistoryPreview(
    allocator: std.mem.Allocator,
    turn_history: []const repl_types.TurnEntry,
    count: usize,
    head: usize,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const idx = (head + repl_types.MAX_TURN_HISTORY - count + i) % repl_types.MAX_TURN_HISTORY;
        if (idx >= turn_history.len) break;
        const entry = &turn_history[idx];
        if (entry.input.len > 0) {
            try out.appendSlice(allocator, "user: ");
            try out.appendSlice(allocator, entry.input);
            try out.appendSlice(allocator, "\n");
        }
        if (entry.response.len > 0) {
            try out.appendSlice(allocator, "assistant: ");
            try out.appendSlice(allocator, entry.response);
            try out.appendSlice(allocator, "\n");
        }
    }
    return out.toOwnedSlice(allocator);
}

/// Build the augmented prompt prefix from plugin context snippets, the
/// turn-history ring buffer, and the open-file context. Returns an owned
/// slice the caller must free, or null when no context is active (so the
/// caller can pass the raw line through unchanged). Pure: no IO, no store.
pub fn buildCompletionContext(
    allocator: std.mem.Allocator,
    context_snippets: []const u8,
    turn_history: []const repl_types.TurnEntry,
    turn_history_count: usize,
    turn_history_head: usize,
    open_path: []const u8,
    open_content: []const u8,
    resolved_line: []const u8,
) !?[]u8 {
    var ctx_parts = std.ArrayListUnmanaged(u8).empty;
    defer ctx_parts.deinit(allocator);

    if (context_snippets.len > 0) {
        try ctx_parts.appendSlice(allocator, context_snippets);
    }

    if (turn_history_count > 0) {
        try ctx_parts.appendSlice(allocator, "[history]\n");
        const preview = try formatTurnHistoryPreview(allocator, turn_history, turn_history_count, turn_history_head);
        defer allocator.free(preview);
        try ctx_parts.appendSlice(allocator, preview);
        try ctx_parts.appendSlice(allocator, "[/history]\n");
    }

    if (open_content.len > 0) {
        try ctx_parts.appendSlice(allocator, "[file: ");
        try ctx_parts.appendSlice(allocator, open_path);
        try ctx_parts.appendSlice(allocator, "]\n");
        try ctx_parts.appendSlice(allocator, open_content);
        try ctx_parts.appendSlice(allocator, "\n");
    }

    if (ctx_parts.items.len == 0) return null;

    try ctx_parts.appendSlice(allocator, "---\n");
    try ctx_parts.appendSlice(allocator, resolved_line);
    return try ctx_parts.toOwnedSlice(allocator);
}

// ── Tests ─────────────────────────────────────────────────────────────────

test "buildCompletionContext returns null with no history file or snippets" {
    const turn_history: [repl_types.MAX_TURN_HISTORY]repl_types.TurnEntry = @splat(repl_types.TurnEntry{ .input = "", .response = "" });
    const result = try buildCompletionContext(
        std.testing.allocator,
        "",
        &turn_history,
        0,
        0,
        "",
        "",
        "hello",
    );
    try std.testing.expect(result == null);
}

test "buildCompletionContext prepends turn history in insertion order" {
    var state = repl_types.ReplState.init(.{});
    state.pushTurn(std.testing.allocator, "hello", "world");
    state.pushTurn(std.testing.allocator, "foo", "bar");
    defer state.clearTurnHistory(std.testing.allocator);

    const result = try buildCompletionContext(
        std.testing.allocator,
        "",
        &state.turn_history,
        state.turn_history_count,
        state.turn_history_head,
        "",
        "",
        "next prompt",
    );
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "[history]\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "user: hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "assistant: world") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "user: foo") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "[/history]\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "---\nnext prompt") != null);
}

test "buildCompletionContext wraps ring buffer at MAX_TURN_HISTORY" {
    var state = repl_types.ReplState.init(.{});
    var i: usize = 0;
    while (i < repl_types.MAX_TURN_HISTORY + 2) : (i += 1) {
        var buf: [32]u8 = undefined;
        const label = std.fmt.bufPrint(&buf, "t{d:0>2}", .{i}) catch unreachable;
        state.pushTurn(std.testing.allocator, label, label);
    }
    defer state.clearTurnHistory(std.testing.allocator);

    const result = try buildCompletionContext(
        std.testing.allocator,
        "",
        &state.turn_history,
        state.turn_history_count,
        state.turn_history_head,
        "",
        "",
        "prompt",
    );
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "user: t00") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "user: t01") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "user: t02") != null);
}

test "buildCompletionContext prepends file context block" {
    const turn_history: [repl_types.MAX_TURN_HISTORY]repl_types.TurnEntry = @splat(repl_types.TurnEntry{ .input = "", .response = "" });
    const result = try buildCompletionContext(
        std.testing.allocator,
        "",
        &turn_history,
        0,
        0,
        "src/main.zig",
        "pub fn main() void {}",
        "prompt",
    );
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "[file: src/main.zig]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "pub fn main() void {}") != null);
}

test "buildCompletionContext prepends plugin context snippets" {
    const turn_history: [repl_types.MAX_TURN_HISTORY]repl_types.TurnEntry = @splat(repl_types.TurnEntry{ .input = "", .response = "" });
    const result = try buildCompletionContext(
        std.testing.allocator,
        "PLUGIN_CONTEXT_HERE",
        &turn_history,
        0,
        0,
        "",
        "",
        "prompt",
    );
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    try std.testing.expect(std.mem.indexOf(u8, result.?, "PLUGIN_CONTEXT_HERE") != null);
}

test "buildCompletionContext combines all three sources in order snippets history file" {
    var state = repl_types.ReplState.init(.{});
    state.pushTurn(std.testing.allocator, "q", "a");
    defer state.clearTurnHistory(std.testing.allocator);

    const result = try buildCompletionContext(
        std.testing.allocator,
        "SNIPPET",
        &state.turn_history,
        state.turn_history_count,
        state.turn_history_head,
        "file.zig",
        "code",
        "prompt",
    );
    try std.testing.expect(result != null);
    defer std.testing.allocator.free(result.?);
    const snippet_pos = std.mem.indexOf(u8, result.?, "SNIPPET").?;
    const history_pos = std.mem.indexOf(u8, result.?, "[history]").?;
    const file_pos = std.mem.indexOf(u8, result.?, "[file: file.zig]").?;
    const prompt_pos = std.mem.indexOf(u8, result.?, "---\nprompt").?;
    try std.testing.expect(snippet_pos < history_pos);
    try std.testing.expect(history_pos < file_pos);
    try std.testing.expect(file_pos < prompt_pos);
}

test {
    std.testing.refAllDecls(@This());
}
