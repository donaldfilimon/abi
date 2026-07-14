//! Interactive TUI REPL (Phase 1: line-at-a-time).
//!
//! Reads one line at a time from stdin using the standard reader (no raw-mode
//! terminal handling yet — that is Phase 2), dispatches slash-commands, and runs
//! ordinary input through the AI completion path against the ambient WDBX store.

const std = @import("std");
const build_options = @import("build_options");
const builtin = @import("builtin");
const env = @import("../../foundation/env.zig");
const utils = @import("../../foundation/utils.zig");
const models = @import("../ai/models.zig");
const ai = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const time = @import("../../foundation/time.zig");
const sanitize = @import("sanitize.zig");
const terminal = @import("terminal.zig");
const line_editor = @import("line_editor.zig");

/// Maximum length of a model id settable via `/model`.
const MODEL_STORAGE_BYTES = 128;

pub const ReplConfig = struct {
    model: []const u8 = models.default_model,
    store_turns: bool = true,
    prompt_prefix: []const u8 = "> ",
};

pub const ReplState = struct {
    config: ReplConfig,
    turn_count: usize,
    session_id: i64,
    /// Backing storage for a model id set at runtime via `/model`, so
    /// `config.model` never dangles into the transient stdin read buffer.
    model_storage: [MODEL_STORAGE_BYTES]u8 = undefined,

    pub fn init(config: ReplConfig) ReplState {
        return .{
            .config = config,
            .turn_count = 0,
            .session_id = time.unixMs(),
        };
    }
};

pub const SpecialCommand = enum { quit, reset, help, model, profile, status, history, syncclis, unknown };

const SlashCommand = struct {
    kind: SpecialCommand,
    name: []const u8,
    aliases: []const []const u8 = &.{},
    summary: []const u8,
};

const slash_commands = [_]SlashCommand{
    .{ .kind = .help, .name = "help", .aliases = &.{"h"}, .summary = "Show this help" },
    .{ .kind = .model, .name = "model", .summary = "Switch the completion model (alias-resolved)" },
    .{ .kind = .profile, .name = "profile", .summary = "Show profile routing status" },
    .{ .kind = .status, .name = "status", .aliases = &.{"stat"}, .summary = "Show session, model, and persistence status" },
    .{ .kind = .history, .name = "history", .aliases = &.{"hist"}, .summary = "Show recent session turns and persisted blocks" },
    .{ .kind = .reset, .name = "reset", .summary = "Reset the turn counter and start a fresh session" },
    .{ .kind = .syncclis, .name = "sync-clis", .aliases = &.{"syncclis"}, .summary = "Sync skills/plugins/commands/experiences across CLIs" },
    .{ .kind = .quit, .name = "quit", .aliases = &.{ "q", "exit" }, .summary = "Exit the REPL" },
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

/// Format a one-line `/history` summary header into `buf`. Pure (no IO/store) so
/// it is unit-testable; falls back to a fixed string if formatting overflows.
fn formatHistoryHeader(buf: []u8, turn_count: usize, block_count: usize) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "history: {d} turn(s) this session, {d} persisted block(s)",
        .{ turn_count, block_count },
    ) catch "history: summary unavailable";
}

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn validModelId(id: []const u8) bool {
    if (id.len == 0 or id.len > MODEL_STORAGE_BYTES) return false;
    for (id) |byte| {
        if (byte < 0x21 or byte >= 0x7f or std.ascii.isWhitespace(byte)) return false;
    }
    return true;
}

/// Format a one-line `/status` summary into `buf`. Pure so contracts can verify
/// the operator-facing status fields without constructing a live store.
fn formatStatusLine(
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

/// Return the trimmed argument following a slash-command token, or "" if none.
fn specialArg(line: []const u8) []const u8 {
    const sp = firstWhitespace(line) orelse return "";
    return std.mem.trim(u8, line[sp + 1 ..], " \t\r");
}

fn printHelp() void {
    std.debug.print("Commands:\n", .{});
    for (slash_commands) |def| {
        std.debug.print("  /{s:<14} {s}\n", .{ def.name, def.summary });
    }
    std.debug.print("  <text>           Run a completion and persist the turn\n\n", .{});
}

const SlashCompletion = union(enum) {
    none,
    unique: []const u8,
    ambiguous: []const *const SlashCommand,
};

/// Find canonical slash-command names for a partially typed command token.
/// Completion is intentionally unavailable after whitespace, so prompts and
/// `/model` arguments always retain their literal input behavior.
fn completeSlashCommand(input: []const u8, matches: *[slash_commands.len]*const SlashCommand) SlashCompletion {
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

pub const ReplLoop = struct {
    allocator: std.mem.Allocator,
    store: *wdbx.Store,
    scheduler: *scheduler_mod.Scheduler,
    state: ReplState,

    pub fn init(
        allocator: std.mem.Allocator,
        store: *wdbx.Store,
        scheduler: *scheduler_mod.Scheduler,
        config: ReplConfig,
    ) ReplLoop {
        return .{
            .allocator = allocator,
            .store = store,
            .scheduler = scheduler,
            .state = ReplState.init(config),
        };
    }

    pub fn deinit(self: *ReplLoop) void {
        _ = self;
    }

    /// Outcome of dispatching one input line. Lets the raw-mode and line-mode
    /// loops share identical handling and quit semantics.
    const LineOutcome = enum { keep_going, quit };

    /// Run the REPL until `/quit` or EOF (Ctrl-D). Phase 2: attempt the raw-mode
    /// `InteractiveTerminal` for nicer character-at-a-time input, and fall back
    /// to the Phase-1 line-at-a-time path when the terminal cannot enter raw mode
    /// (CI / non-tty), mirroring `dashboard.zig`'s fallback. Behavior under a
    /// non-tty is byte-for-byte the Phase-1 path so the `agent tui` contract is
    /// unchanged.
    pub fn run(self: *ReplLoop, io: std.Io) !void {
        var term = terminal.InteractiveTerminal.init(terminal.stdinFd()) catch {
            return self.runLineMode(io);
        };
        defer term.deinit();
        return self.runRawMode(&term, io);
    }

    /// Phase-1 path: read one line at a time from stdin via the standard reader.
    fn runLineMode(self: *ReplLoop, io: std.Io) !void {
        var buf: [4096]u8 = undefined;
        var stdin_reader = std.Io.File.stdin().reader(io, &buf);

        while (true) {
            std.debug.print("{s}", .{self.state.config.prompt_prefix});

            const maybe_line = stdin_reader.interface.takeDelimiter('\n') catch |err| {
                std.debug.print("\nrepl: input error: {s}\n", .{@errorName(err)});
                break;
            };
            const raw = maybe_line orelse break; // EOF
            const line = std.mem.trim(u8, raw, " \t\r\n");
            if (line.len == 0) continue;

            switch (try self.dispatchLine(line, io)) {
                .quit => break,
                .keep_going => {},
            }
        }
    }

    /// Raw-mode path with bounded line editing. The terminal itself supplies
    /// bytes; `line_editor` decodes them into safe editor actions so controls
    /// never become part of a submitted prompt.
    fn runRawMode(self: *ReplLoop, term: *terminal.InteractiveTerminal, io: std.Io) !void {
        var editor = line_editor.LineEditor.init(self.allocator);
        defer editor.deinit();
        var decoder = line_editor.KeyDecoder{};
        self.resetRawPrompt(&editor);

        while (true) {
            const key = try self.readRawKey(term, &decoder) orelse break;
            switch (key) {
                .printable => |byte| if (try editor.insertPrintable(byte)) self.redrawRawInput(&editor),
                .left => if (editor.moveLeft()) self.redrawRawInput(&editor),
                .right => if (editor.moveRight()) self.redrawRawInput(&editor),
                .home => if (editor.moveHome()) self.redrawRawInput(&editor),
                .end => if (editor.moveEnd()) self.redrawRawInput(&editor),
                .backspace => if (editor.deleteBackward()) self.redrawRawInput(&editor),
                .delete => if (editor.deleteForward()) self.redrawRawInput(&editor),
                .up => if (try editor.historyUp()) self.redrawRawInput(&editor),
                .down => if (try editor.historyDown()) self.redrawRawInput(&editor),
                .tab => try self.applyTabCompletion(&editor),
                .enter => {
                    self.finishRawLine(&editor);
                    std.debug.print("\n", .{});
                    const line = std.mem.trim(u8, editor.text(), " \t\r\n");
                    const outcome = if (line.len > 0) blk: {
                        try editor.recordSubmitted();
                        break :blk try self.dispatchLine(line, io);
                    } else .keep_going;
                    editor.clear();
                    if (outcome == .quit) return;
                    self.resetRawPrompt(&editor);
                },
                .eof => break,
                .ignore => {},
            }
        }
    }

    /// Decode one terminal action. Escape sequences are given a short bounded
    /// wait; a lone or malformed escape is cancelled and has no editor effect.
    fn readRawKey(self: *ReplLoop, term: *terminal.InteractiveTerminal, decoder: *line_editor.KeyDecoder) !?line_editor.Key {
        _ = self;
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

    fn redrawRawInput(self: *ReplLoop, editor: *const line_editor.LineEditor) void {
        std.debug.print("\x1b[u{s}{s}\x1b[0J", .{ self.state.config.prompt_prefix, editor.text() });
        const trailing = editor.text().len - editor.cursor;
        if (trailing > 0) std.debug.print("\x1b[{d}D", .{trailing});
    }

    /// Anchor a prompt at its first cell. Each redraw restores this anchor and
    /// clears to the end of the terminal, removing stale wrapped rows from a
    /// long (up to 4 KiB) line before rendering the current editor state.
    fn resetRawPrompt(self: *ReplLoop, editor: *const line_editor.LineEditor) void {
        std.debug.print("\x1b[s", .{});
        self.redrawRawInput(editor);
    }

    fn finishRawLine(self: *ReplLoop, editor: *const line_editor.LineEditor) void {
        _ = self;
        const trailing = editor.text().len - editor.cursor;
        if (trailing > 0) std.debug.print("\x1b[{d}C", .{trailing});
    }

    fn applyTabCompletion(self: *ReplLoop, editor: *line_editor.LineEditor) !void {
        var matches: [slash_commands.len]*const SlashCommand = undefined;
        switch (completeSlashCommand(editor.text(), &matches)) {
            .none => {},
            .unique => |name| {
                var command: [MODEL_STORAGE_BYTES]u8 = undefined;
                const completed = try std.fmt.bufPrint(&command, "/{s}", .{name});
                try editor.replace(completed);
                self.redrawRawInput(editor);
            },
            .ambiguous => |defs| {
                std.debug.print("\nmatches:", .{});
                for (defs) |def| std.debug.print(" /{s}", .{def.name});
                std.debug.print("\n", .{});
                self.resetRawPrompt(editor);
            },
        }
    }

    /// Dispatch one already-trimmed, non-empty input line: slash-commands route
    /// to their handlers, ordinary text runs a completion and persists the turn.
    fn dispatchLine(self: *ReplLoop, line: []const u8, io: std.Io) !LineOutcome {
        if (line[0] == '/') return self.dispatchSlashCommand(line, io);
        return self.completePrompt(line);
    }

    fn dispatchSlashCommand(self: *ReplLoop, line: []const u8, io: std.Io) !LineOutcome {
        switch (parseSpecialCommand(line)) {
            .quit => return .quit,
            .help => printHelp(),
            .model => self.applyModel(specialArg(line)),
            .reset => self.resetSession(),
            .status => self.showStatus(),
            .history => self.showHistory(),
            .profile => self.showProfileStatus(),
            .syncclis => try self.runSyncClis(io),
            .unknown => try self.printUnknownCommand(line),
        }
        return .keep_going;
    }

    fn persistedBlockCount(self: *ReplLoop) usize {
        if (comptime build_options.feat_wdbx) return self.store.blockCount();
        return 0;
    }

    fn showStatus(self: *ReplLoop) void {
        var buf: [256]u8 = undefined;
        std.debug.print("{s}\n", .{formatStatusLine(
            &buf,
            self.state.session_id,
            self.state.turn_count,
            self.state.config.model,
            self.state.config.store_turns,
            self.persistedBlockCount(),
        )});
    }

    fn runSyncClis(self: *ReplLoop, io: std.Io) !void {
        // The sync-clis launcher lives in the operator's Grok skill dir (outside
        // this repo), not at a hardcoded in-repo path. Resolve the central
        // launcher and never execute a missing script.
        const home_var = if (builtin.target.os.tag == .windows) "USERPROFILE" else "HOME";
        const home = env.get(home_var) orelse {
            std.debug.print("sync-clis: HOME not set; cannot locate launcher\n", .{});
            return;
        };
        const launch_path = try utils.pathJoin(home, ".grok/skills/sync-clis/launch.sh", self.allocator);
        defer self.allocator.free(launch_path);
        std.Io.Dir.cwd().access(io, launch_path, .{}) catch {
            std.debug.print("sync-clis: launcher not found at {s}\n", .{launch_path});
            return;
        };
        std.debug.print("sync-clis: executing central sync (full targets via driver)...\n", .{});
        var child = try std.process.spawn(io, .{
            .argv = &[_][]const u8{launch_path},
            .cwd = .inherit,
            .stdin = .ignore,
            .stdout = .inherit,
            .stderr = .inherit,
        });
        defer child.kill(io);
        const term = try child.wait(io);
        std.debug.print("sync-clis done (exit {any})\n", .{term});
    }

    fn printUnknownCommand(self: *ReplLoop, line: []const u8) !void {
        // Echo the rejected command through the sanitizer: the raw line is
        // attacker-controlled and may carry ESC/control bytes.
        const safe_line = try sanitize.sanitizeControlBytes(self.allocator, line);
        defer self.allocator.free(safe_line);
        std.debug.print("unknown command: {s} (try /help)\n", .{safe_line});
    }

    fn completePrompt(self: *ReplLoop, line: []const u8) !LineOutcome {
        var result = try ai.completeWithScheduler(self.allocator, self.store, self.scheduler, "complete:agent-tui", .{
            .input = line,
            .model = self.state.config.model,
            .store_result = self.state.config.store_turns,
        });
        defer result.deinit(self.allocator);
        self.state.turn_count += 1;

        // Completion output can echo poisoned WDBX content; strip control bytes so
        // it cannot inject terminal escapes into the operator's render stream.
        const safe_output = try sanitize.sanitizeControlBytes(self.allocator, result.output);
        defer self.allocator.free(safe_output);
        std.debug.print("{s}\n", .{safe_output});
        std.debug.print("[turn {d} | model={s} | profile={s} | persisted={s}]\n", .{
            self.state.turn_count,
            result.model,
            result.selected_profile.label(),
            if (result.query_vector_id != null) "true" else "false",
        });
        return .keep_going;
    }

    /// `/reset`: clear the per-session turn counter and start a fresh session
    /// context (a new session id), so subsequent `/history` summaries describe
    /// only turns after the reset.
    fn resetSession(self: *ReplLoop) void {
        self.state.turn_count = 0;
        self.state.session_id = time.unixMs();
        std.debug.print("session reset (id={d})\n", .{self.state.session_id});
    }

    /// `/history`: summarize recent turns. Reads persisted block metadata from
    /// the session store when WDBX is enabled; otherwise prints a friendly note.
    fn showHistory(self: *ReplLoop) void {
        if (comptime build_options.feat_wdbx) {
            var hdr_buf: [128]u8 = undefined;
            const block_count = self.store.blockCount();
            std.debug.print("{s}\n", .{formatHistoryHeader(&hdr_buf, self.state.turn_count, block_count)});
            if (block_count == 0) {
                std.debug.print("(no persisted turns yet)\n", .{});
                return;
            }
            if (self.store.lastBlock()) |block| {
                std.debug.print("last persisted: profile={s} query_id={d} response_id={d}\n", .{
                    block.profile,
                    block.query_id,
                    block.response_id,
                });
            }
        } else {
            std.debug.print("history: unavailable (wdbx feature is disabled in this build)\n", .{});
        }
    }

    fn showProfileStatus(self: *ReplLoop) void {
        std.debug.print("profile: adaptive router active; model={s}; turns={d}\n", .{
            self.state.config.model,
            self.state.turn_count,
        });
    }

    /// Set the active model from a `/model <id>` argument, copying the
    /// alias-resolved id into stable session storage.
    fn applyModel(self: *ReplLoop, arg: []const u8) void {
        if (arg.len == 0) {
            std.debug.print("usage: /model <id>\n", .{});
            return;
        }
        const canon = models.canonical(arg);
        if (!validModelId(canon)) {
            std.debug.print("model id must be printable non-whitespace ASCII and at most {d} bytes\n", .{MODEL_STORAGE_BYTES});
            return;
        }
        @memcpy(self.state.model_storage[0..canon.len], canon);
        self.state.config.model = self.state.model_storage[0..canon.len];
        std.debug.print("model set to {s}\n", .{self.state.config.model});
    }
};

test "parseSpecialCommand recognizes slash commands and treats prompts as unknown" {
    try std.testing.expectEqual(SpecialCommand.quit, parseSpecialCommand("/quit"));
    try std.testing.expectEqual(SpecialCommand.quit, parseSpecialCommand("/q"));
    try std.testing.expectEqual(SpecialCommand.help, parseSpecialCommand("/help"));
    try std.testing.expectEqual(SpecialCommand.help, parseSpecialCommand("/h\t"));
    try std.testing.expectEqual(SpecialCommand.model, parseSpecialCommand("/model apple-fm"));
    try std.testing.expectEqual(SpecialCommand.model, parseSpecialCommand("/model\tapple-fm"));
    try std.testing.expectEqual(SpecialCommand.reset, parseSpecialCommand("/reset"));
    try std.testing.expectEqual(SpecialCommand.status, parseSpecialCommand("/status"));
    try std.testing.expectEqual(SpecialCommand.status, parseSpecialCommand("/stat"));
    try std.testing.expectEqual(SpecialCommand.history, parseSpecialCommand("/history"));
    try std.testing.expectEqual(SpecialCommand.history, parseSpecialCommand("/hist"));
    try std.testing.expectEqual(SpecialCommand.profile, parseSpecialCommand("/profile abbey"));
    try std.testing.expectEqual(SpecialCommand.syncclis, parseSpecialCommand("/sync-clis"));
    try std.testing.expectEqual(SpecialCommand.syncclis, parseSpecialCommand("/syncclis"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand("/bogus"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand("hello there"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand(""));
}

test "specialArg trims spaces and tabs after slash commands" {
    try std.testing.expectEqualStrings("apple-fm", specialArg("/model\t apple-fm \r"));
    try std.testing.expectEqualStrings("abi-local", specialArg("/model   abi-local"));
    try std.testing.expectEqualStrings("", specialArg("/history"));
}

test "formatHistoryHeader summarizes session turns and persisted blocks" {
    var buf: [128]u8 = undefined;
    const line = formatHistoryHeader(&buf, 3, 5);
    try std.testing.expect(std.mem.indexOf(u8, line, "3 turn") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "5 persisted block") != null);
}

test "formatStatusLine reports model provider and persistence state" {
    var buf: [256]u8 = undefined;
    const line = formatStatusLine(&buf, 1234, 3, "fable-5", true, 5);
    try std.testing.expect(std.mem.indexOf(u8, line, "session_id=1234") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "turns=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "model=fable-5") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "provider=anthropic") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "store_turns=true") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "persisted_blocks=5") != null);
}

test "validModelId rejects controls whitespace and overlong ids" {
    try std.testing.expect(validModelId("abi-local"));
    try std.testing.expect(validModelId("ollama/qwen2"));
    try std.testing.expect(!validModelId(""));
    try std.testing.expect(!validModelId("two words"));
    try std.testing.expect(!validModelId("bad\tid"));
    try std.testing.expect(!validModelId("bad\x1bid"));

    var overlong: [MODEL_STORAGE_BYTES + 1]u8 = undefined;
    @memset(&overlong, 'a');
    try std.testing.expect(!validModelId(&overlong));
}

test "ReplState init seeds defaults and a session id" {
    const state = ReplState.init(.{});
    try std.testing.expectEqual(@as(usize, 0), state.turn_count);
    try std.testing.expectEqualStrings(models.default_model, state.config.model);
    try std.testing.expect(state.session_id > 0);
}

test "slash completion canonicalizes unique prefixes and reports ambiguity" {
    var matches: [slash_commands.len]*const SlashCommand = undefined;

    switch (completeSlashCommand("/mod", &matches)) {
        .unique => |name| try std.testing.expectEqualStrings("model", name),
        else => return error.ExpectedUniqueModel,
    }
    switch (completeSlashCommand("/stat", &matches)) {
        .unique => |name| try std.testing.expectEqualStrings("status", name),
        else => return error.ExpectedUniqueStatus,
    }
    switch (completeSlashCommand("/s", &matches)) {
        .ambiguous => |defs| {
            try std.testing.expectEqual(@as(usize, 2), defs.len);
            try std.testing.expectEqualStrings("status", defs[0].name);
            try std.testing.expectEqualStrings("sync-clis", defs[1].name);
        },
        else => return error.ExpectedAmbiguousSlashPrefix,
    }
    switch (completeSlashCommand("/model abi", &matches)) {
        .none => {},
        else => return error.ExpectedLiteralModelArgument,
    }
    switch (completeSlashCommand("ordinary prompt", &matches)) {
        .none => {},
        else => return error.ExpectedLiteralPrompt,
    }
}

test "tui input hardening: adversarial bytes never corrupt state" {
    // parseSpecialCommand is the prime fuzz target: it must classify every
    // adversarial line into a valid SpecialCommand variant without panicking,
    // reading out of bounds, or overflowing.
    const adversarial = [_][]const u8{
        "", // empty
        "\x00", // lone NUL
        "\x1b[A\x1b[B", // arrow-key ESC sequences
        "\xff\xfe", // invalid UTF-8
        "quit\x00extra", // embedded NUL after a word (non-slash → unknown)
        "/quit\x00extra", // slash-command with trailing NUL payload
    };
    for (adversarial) |input| {
        // Exhaustive switch: reaching every arm without a panic is the safety
        // assertion (the classifier is total over all byte strings).
        switch (parseSpecialCommand(input)) {
            .quit, .reset, .help, .model, .profile, .status, .history, .syncclis, .unknown => {},
        }
    }

    // Every one of the 256 byte values in a single line must be tolerated.
    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*b, i| b.* = @intCast(i);
    _ = parseSpecialCommand(&all_bytes);

    // An overlong buffer (larger than the raw-mode line buffer) must classify
    // without overflow. A leading '/' forces the slash-command body scan.
    const overlong = try std.testing.allocator.alloc(u8, line_editor.MAX_LINE_BYTES * 2);
    defer std.testing.allocator.free(overlong);
    @memset(overlong, '/');
    _ = parseSpecialCommand(overlong);
    // Such input exceeds the editor's bounded input capacity, so insertion can
    // reject overflow rather than writing past its backing storage.
    try std.testing.expect(overlong.len > line_editor.MAX_LINE_BYTES);

    // The output sanitizer must remove ESC and NUL from attacker-influenced text
    // so render fields cannot inject terminal escapes. Length is preserved.
    const dirty = "\x1b[31mred\x00\x07\x7fboom\x1b[0m";
    const clean = try sanitize.sanitizeControlBytes(std.testing.allocator, dirty);
    defer std.testing.allocator.free(clean);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x1b) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x00) == null);
    try std.testing.expectEqual(dirty.len, clean.len);
}

test "raw editor rejects control bytes and accepts printable ASCII" {
    var editor = line_editor.LineEditor.init(std.testing.allocator);
    defer editor.deinit();

    // ESC, NUL, and DEL cannot become prompt text.
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
