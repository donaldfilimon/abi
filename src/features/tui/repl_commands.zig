//! Slash-command parsing, completion, and status formatting for the REPL.
//!
//! Extracted from `repl.zig` so the pure, unit-testable command classification
//! and formatting helpers live separately from the interactive `ReplLoop`. The
//! loop owns IO, store, scheduler, and dispatch; this module owns the
//! declarative slash-command table and the pure string helpers.

const std = @import("std");
const models = @import("../ai/models.zig");

/// Maximum length of a model id settable via `/model`.
pub const MODEL_STORAGE_BYTES = 128;

pub const SpecialCommand = enum { quit, reset, help, model, profile, status, history, context, syncclis, open, diff, commit, features, learn, save, load, unknown };

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
    .{ .kind = .diff, .name = "diff", .summary = "Show git diff for the working tree" },
    .{ .kind = .commit, .name = "commit", .summary = "Stage all changes and create a commit" },
    .{ .kind = .quit, .name = "quit", .aliases = &.{ "q", "exit" }, .summary = "Exit the REPL" },
    .{ .kind = .features, .name = "features", .aliases = &.{"feat"}, .summary = "Show active build-time features" },
    .{ .kind = .learn, .name = "learn", .summary = "Toggle SEA self-learning mode on/off" },
    .{ .kind = .save, .name = "save", .summary = "Save session context to ~/.abi/sessions/<name>.json" },
    .{ .kind = .load, .name = "load", .summary = "Restore session context from ~/.abi/sessions/<name>.json" },
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
            // Format aliases as a comma-separated list in parentheses
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

    // Open file section
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

    // Turn history section
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

    // Preview of turn history
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

test {
    std.testing.refAllDecls(@This());
}
