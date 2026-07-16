//! Terminal input loops for the interactive REPL.
//!
//! Extracted from `repl.zig` so the dispatch hub stays focused on
//! slash-command routing. This leaf owns the Phase-1 line-at-a-time stdin
//! loop, the Phase-2 raw-mode loop with bounded line editing, key decoding,
//! prompt redraw, tab completion, and the Ctrl-R reverse-search glue.
//!
//! The two loop entry points take the owning `ReplLoop` as `anytype` so they
//! can call back into its `dispatchLine` without a circular import; every
//! other helper is a free function taking exactly the pieces it needs
//! (allocator, prompt prefix, terminal, editor).

const std = @import("std");
const terminal = @import("terminal.zig");
const line_editor = @import("line_editor.zig");
const cmds = @import("repl_commands.zig");

/// Phase-1 path: read one line at a time from stdin via the standard reader.
pub fn runLineMode(loop: anytype, io: std.Io) !void {
    var buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &buf);

    while (true) {
        std.debug.print("{s}", .{loop.state.config.prompt_prefix});

        const maybe_line = stdin_reader.interface.takeDelimiter('\n') catch |err| {
            std.debug.print("\nrepl: input error: {s}\n", .{@errorName(err)});
            break;
        };
        const raw = maybe_line orelse break; // EOF
        const line = std.mem.trim(u8, raw, " \t\r\n");
        if (line.len == 0) continue;

        switch (try loop.dispatchLine(line, io)) {
            .quit => break,
            .keep_going => {},
        }
    }
}

/// Raw-mode path with bounded line editing. The terminal itself supplies
/// bytes; `line_editor` decodes them into safe editor actions so controls
/// never become part of a submitted prompt.
pub fn runRawMode(loop: anytype, term: *terminal.InteractiveTerminal, io: std.Io) !void {
    var editor = line_editor.LineEditor.init(loop.allocator);
    defer editor.deinit();
    var decoder = line_editor.KeyDecoder{};
    const prompt = loop.state.config.prompt_prefix;
    resetRawPrompt(prompt, &editor);

    while (true) {
        const key = try readRawKey(term, &decoder) orelse break;
        switch (key) {
            .printable => |byte| if (try editor.insertPrintable(byte)) redrawRawInput(prompt, &editor),
            .left => if (editor.moveLeft()) redrawRawInput(prompt, &editor),
            .right => if (editor.moveRight()) redrawRawInput(prompt, &editor),
            .home => if (editor.moveHome()) redrawRawInput(prompt, &editor),
            .end => if (editor.moveEnd()) redrawRawInput(prompt, &editor),
            .backspace => if (editor.deleteBackward()) redrawRawInput(prompt, &editor),
            .delete => if (editor.deleteForward()) redrawRawInput(prompt, &editor),
            .up => if (try editor.historyUp()) redrawRawInput(prompt, &editor),
            .down => if (try editor.historyDown()) redrawRawInput(prompt, &editor),
            .tab => try applyTabCompletion(prompt, &editor),
            .enter => {
                finishRawLine(&editor);
                std.debug.print("\n", .{});
                const line = std.mem.trim(u8, editor.text(), " \t\r\n");
                const outcome = if (line.len > 0) blk: {
                    try editor.recordSubmitted();
                    break :blk try loop.dispatchLine(line, io);
                } else .keep_going;
                editor.clear();
                if (outcome == .quit) return;
                resetRawPrompt(prompt, &editor);
            },
            .newline => if (try editor.insertNewline()) redrawRawInput(prompt, &editor),
            .ctrl_r => try startReverseSearch(loop.allocator, prompt, term, &editor),
            .ctrl_l => {
                std.debug.print("\x1b[2J\x1b[H", .{});
                resetRawPrompt(prompt, &editor);
            },
            .ctrl_k => if (editor.killToEnd()) redrawRawInput(prompt, &editor),
            .ctrl_u => if (editor.killToBeginning()) redrawRawInput(prompt, &editor),
            .ctrl_w => if (editor.deletePreviousWord()) redrawRawInput(prompt, &editor),
            .eof => break,
            .ignore => {},
        }
    }
}

/// Decode one terminal action. Escape sequences are given a short bounded
/// wait; a lone or malformed escape is cancelled and has no editor effect.
fn readRawKey(term: *terminal.InteractiveTerminal, decoder: *line_editor.KeyDecoder) !?line_editor.Key {
    const first = term.readKey() orelse return null;
    if (decoder.feed(first)) |key| return key;
    while (decoder.pending()) {
        if (!term.pollInput(30)) {
            decoder.cancelPending();
            return .ignore;
        }
        const next = term.readKey() orelse {
            decoder.cancelPending();
            return null;
        };
        if (decoder.feed(next)) |key| return key;
    }
    return .ignore;
}

fn redrawRawInput(prompt_prefix: []const u8, editor: *const line_editor.LineEditor) void {
    std.debug.print("\x1b[u{s}{s}\x1b[0J", .{ prompt_prefix, editor.text() });
    const trailing = editor.text().len - editor.cursor;
    if (trailing > 0) std.debug.print("\x1b[{d}D", .{trailing});
}

/// Anchor a prompt at its first cell. Each redraw restores this anchor and
/// clears to the end of the terminal, removing stale wrapped rows from a
/// long (up to 4 KiB) line before rendering the current editor state.
fn resetRawPrompt(prompt_prefix: []const u8, editor: *const line_editor.LineEditor) void {
    std.debug.print("\x1b[s", .{});
    redrawRawInput(prompt_prefix, editor);
}

fn finishRawLine(editor: *const line_editor.LineEditor) void {
    const trailing = editor.text().len - editor.cursor;
    if (trailing > 0) std.debug.print("\x1b[{d}C", .{trailing});
}

fn applyTabCompletion(prompt_prefix: []const u8, editor: *line_editor.LineEditor) !void {
    var matches: [cmds.slash_commands.len]*const cmds.SlashCommand = undefined;
    switch (cmds.completeSlashCommand(editor.text(), &matches)) {
        .none => {},
        .unique => |name| {
            var command: [cmds.MODEL_STORAGE_BYTES]u8 = undefined;
            const completed = try std.fmt.bufPrint(&command, "/{s}", .{name});
            try editor.replace(completed);
            redrawRawInput(prompt_prefix, editor);
        },
        .ambiguous => |defs| {
            std.debug.print("\nmatches:", .{});
            for (defs) |def| std.debug.print(" /{s}", .{def.name});
            std.debug.print("\n", .{});
            resetRawPrompt(prompt_prefix, editor);
        },
    }
}

/// Ctrl-R: start an incremental reverse history search. Reads bytes from
/// the terminal until Enter (accept), Ctrl-C/Esc (cancel), or backspace
/// (shorten query). Matching history entries replace the editor buffer.
fn startReverseSearch(
    allocator: std.mem.Allocator,
    prompt_prefix: []const u8,
    term: *terminal.InteractiveTerminal,
    editor: *line_editor.LineEditor,
) !void {
    var query = std.ArrayListUnmanaged(u8).empty;
    defer query.deinit(allocator);

    std.debug.print("\n(reverse-search) ", .{});
    redrawRawInput(prompt_prefix, editor);

    while (true) {
        const byte = term.readKey() orelse break;
        switch (byte) {
            '\r', '\n' => {
                if (editor.text().len > 0) {
                    std.debug.print("\n", .{});
                    return;
                }
            },
            0x03, 0x1b => {
                editor.clear();
                std.debug.print("\n(cancelled)\n", .{});
                resetRawPrompt(prompt_prefix, editor);
                return;
            },
            0x08, 0x7f => {
                if (query.items.len > 0) {
                    _ = query.pop();
                    editor.clear();
                    if (query.items.len > 0) {
                        _ = try editor.searchHistory(query.items);
                    }
                    std.debug.print("\r(reverse-search) {s} ", .{query.items});
                    redrawRawInput(prompt_prefix, editor);
                }
            },
            else => if (byte >= 0x20 and byte < 0x7f) {
                try query.append(allocator, byte);
                editor.clear();
                if (try editor.searchHistory(query.items)) |_| {
                    std.debug.print("\r(reverse-search) {s} ", .{query.items});
                    redrawRawInput(prompt_prefix, editor);
                } else {
                    std.debug.print("\r(reverse-search) {s} (no match)", .{query.items});
                }
            },
        }
    }
}

test "slash completion canonicalizes unique prefixes and reports ambiguity" {
    var matches: [cmds.slash_commands.len]*const cmds.SlashCommand = undefined;

    switch (cmds.completeSlashCommand("/mod", &matches)) {
        .unique => |name| try std.testing.expectEqualStrings("model", name),
        else => return error.ExpectedUniqueModel,
    }
    switch (cmds.completeSlashCommand("/stat", &matches)) {
        .unique => |name| try std.testing.expectEqualStrings("status", name),
        else => return error.ExpectedUniqueStatus,
    }
    switch (cmds.completeSlashCommand("/s", &matches)) {
        .ambiguous => |defs| {
            try std.testing.expectEqual(@as(usize, 4), defs.len);
            try std.testing.expectEqualStrings("status", defs[0].name);
            try std.testing.expectEqualStrings("sync-clis", defs[1].name);
            try std.testing.expectEqualStrings("save", defs[2].name);
            try std.testing.expectEqualStrings("sessions", defs[3].name);
        },
        else => return error.ExpectedAmbiguousSlashPrefix,
    }
    switch (cmds.completeSlashCommand("/model abi", &matches)) {
        .none => {},
        else => return error.ExpectedLiteralModelArgument,
    }
    switch (cmds.completeSlashCommand("ordinary prompt", &matches)) {
        .none => {},
        else => return error.ExpectedLiteralPrompt,
    }
}

test "raw editor rejects control bytes and accepts printable ASCII" {
    var editor = line_editor.LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    try std.testing.expect(!(try editor.insertPrintable(0x1b)));
    try std.testing.expect(!(try editor.insertPrintable(0x00)));
    try std.testing.expect(!(try editor.insertPrintable(0x7f)));
    try std.testing.expect(try editor.insertPrintable(0x20));
    try std.testing.expect(try editor.insertPrintable(0x7e));
    try std.testing.expectEqualStrings(" ~", editor.text());
}

test {
    std.testing.refAllDecls(@This());
}
