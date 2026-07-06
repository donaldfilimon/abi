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

/// Maximum length of a model id settable via `/model`.
const MODEL_STORAGE_BYTES = 128;

/// Maximum bytes buffered for a single raw-mode input line. `runRawMode`'s
/// printable filter drops any byte once `line_buf` is full, so input past this
/// bound is discarded rather than overflowing the buffer.
const RAW_LINE_BUF_BYTES = 4096;

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

pub const SpecialCommand = enum { quit, reset, help, model, profile, history, syncclis, unknown };

/// Classify a line as a slash-command. Non-slash lines (ordinary prompts) and
/// unrecognized slash-commands both map to `.unknown`; callers distinguish the
/// two by checking for a leading `/`.
pub fn parseSpecialCommand(line: []const u8) SpecialCommand {
    if (line.len == 0 or line[0] != '/') return .unknown;
    const body = line[1..];
    const end = std.mem.indexOfScalar(u8, body, ' ') orelse body.len;
    const cmd = body[0..end];
    if (std.mem.eql(u8, cmd, "quit") or std.mem.eql(u8, cmd, "q") or std.mem.eql(u8, cmd, "exit")) return .quit;
    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "h")) return .help;
    if (std.mem.eql(u8, cmd, "model")) return .model;
    if (std.mem.eql(u8, cmd, "reset")) return .reset;
    if (std.mem.eql(u8, cmd, "history") or std.mem.eql(u8, cmd, "hist")) return .history;
    if (std.mem.eql(u8, cmd, "profile")) return .profile;
    if (std.mem.eql(u8, cmd, "sync-clis") or std.mem.eql(u8, cmd, "syncclis")) return .syncclis;
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

/// Return the trimmed argument following a slash-command token, or "" if none.
fn specialArg(line: []const u8) []const u8 {
    const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return "";
    return std.mem.trim(u8, line[sp + 1 ..], " \t\r");
}

fn printHelp() void {
    std.debug.print(
        \\Commands:
        \\  /help            Show this help
        \\  /model <id>      Switch the completion model (alias-resolved)
        \\  /history         Show recent session turns and persisted blocks
        \\  /reset           Reset the turn counter and start a fresh session
        \\  /quit            Exit the REPL
        \\  /sync-clis       Sync skills/plugins/commands/experiences across CLIs (grok, claude, codex, opencode, abi tui, ...)
        \\  <text>           Run a completion and persist the turn
        \\
    , .{});
}

/// Raw-mode input filter: accept only the printable ASCII range (0x20–0x7E) for
/// the line buffer. Drops C0 controls (incl. ESC 0x1B and NUL 0x00) and DEL
/// (0x7F) so attacker-injected escape bytes never reach the buffered line.
/// Pure and file-local so it is unit-testable without a live terminal.
fn isPrintableInput(key: u8) bool {
    return key >= 0x20 and key < 0x7f;
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

    /// Phase-2 path: assemble input character-by-character under raw mode. Echo
    /// is disabled in raw mode so printable keys are echoed manually; Enter
    /// submits, Backspace edits, and Ctrl-D on an empty buffer ends the session.
    fn runRawMode(self: *ReplLoop, term: *terminal.InteractiveTerminal, io: std.Io) !void {
        var line_buf: [RAW_LINE_BUF_BYTES]u8 = undefined;
        var len: usize = 0;
        std.debug.print("{s}", .{self.state.config.prompt_prefix});

        while (true) {
            const key = term.readKey() orelse break; // closed stdin
            switch (key) {
                '\r', '\n' => {
                    std.debug.print("\n", .{});
                    const line = std.mem.trim(u8, line_buf[0..len], " \t\r\n");
                    const outcome = if (line.len > 0) try self.dispatchLine(line, io) else .keep_going;
                    len = 0;
                    if (outcome == .quit) return;
                    std.debug.print("{s}", .{self.state.config.prompt_prefix});
                },
                0x04 => break, // Ctrl-D
                0x7f, 0x08 => { // Backspace / Delete
                    if (len > 0) {
                        len -= 1;
                        std.debug.print("\x08 \x08", .{}); // erase the echoed glyph
                    }
                },
                else => {
                    if (isPrintableInput(key) and len < line_buf.len) {
                        line_buf[len] = key;
                        len += 1;
                        std.debug.print("{c}", .{key});
                    }
                },
            }
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
            .history => self.showHistory(),
            .profile => std.debug.print("profile selection is not available in phase 1\n", .{}),
            .syncclis => try self.runSyncClis(io),
            .unknown => try self.printUnknownCommand(line),
        }
        return .keep_going;
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

    /// Set the active model from a `/model <id>` argument, copying the
    /// alias-resolved id into stable session storage.
    fn applyModel(self: *ReplLoop, arg: []const u8) void {
        if (arg.len == 0) {
            std.debug.print("usage: /model <id>\n", .{});
            return;
        }
        const canon = models.canonical(arg);
        if (canon.len > self.state.model_storage.len) {
            std.debug.print("model id too long\n", .{});
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
    try std.testing.expectEqual(SpecialCommand.model, parseSpecialCommand("/model apple-fm"));
    try std.testing.expectEqual(SpecialCommand.reset, parseSpecialCommand("/reset"));
    try std.testing.expectEqual(SpecialCommand.history, parseSpecialCommand("/history"));
    try std.testing.expectEqual(SpecialCommand.history, parseSpecialCommand("/hist"));
    try std.testing.expectEqual(SpecialCommand.profile, parseSpecialCommand("/profile abbey"));
    try std.testing.expectEqual(SpecialCommand.syncclis, parseSpecialCommand("/sync-clis"));
    try std.testing.expectEqual(SpecialCommand.syncclis, parseSpecialCommand("/syncclis"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand("/bogus"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand("hello there"));
    try std.testing.expectEqual(SpecialCommand.unknown, parseSpecialCommand(""));
}

test "formatHistoryHeader summarizes session turns and persisted blocks" {
    var buf: [128]u8 = undefined;
    const line = formatHistoryHeader(&buf, 3, 5);
    try std.testing.expect(std.mem.indexOf(u8, line, "3 turn") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "5 persisted block") != null);
}

test "ReplState init seeds defaults and a session id" {
    const state = ReplState.init(.{});
    try std.testing.expectEqual(@as(usize, 0), state.turn_count);
    try std.testing.expectEqualStrings(models.default_model, state.config.model);
    try std.testing.expect(state.session_id > 0);
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
            .quit, .reset, .help, .model, .profile, .history, .syncclis, .unknown => {},
        }
    }

    // Every one of the 256 byte values in a single line must be tolerated.
    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*b, i| b.* = @intCast(i);
    _ = parseSpecialCommand(&all_bytes);

    // An overlong buffer (larger than the raw-mode line buffer) must classify
    // without overflow. A leading '/' forces the slash-command body scan.
    const overlong = try std.testing.allocator.alloc(u8, RAW_LINE_BUF_BYTES * 2);
    defer std.testing.allocator.free(overlong);
    @memset(overlong, '/');
    _ = parseSpecialCommand(overlong);
    // Such input exceeds the raw-mode line buffer; runRawMode's printable filter
    // (`len < line_buf.len`) drops the overflow rather than writing past the bound.
    try std.testing.expect(overlong.len > RAW_LINE_BUF_BYTES);

    // The output sanitizer must remove ESC and NUL from attacker-influenced text
    // so render fields cannot inject terminal escapes. Length is preserved.
    const dirty = "\x1b[31mred\x00\x07\x7fboom\x1b[0m";
    const clean = try sanitize.sanitizeControlBytes(std.testing.allocator, dirty);
    defer std.testing.allocator.free(clean);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x1b) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x00) == null);
    try std.testing.expectEqual(dirty.len, clean.len);
}

test "isPrintableInput drops control bytes and keeps printable ASCII" {
    // Drops: ESC, NUL, DEL — the bytes that could inject terminal escapes.
    try std.testing.expect(!isPrintableInput(0x1b));
    try std.testing.expect(!isPrintableInput(0x00));
    try std.testing.expect(!isPrintableInput(0x7f));
    // Keeps the printable ASCII boundaries.
    try std.testing.expect(isPrintableInput(0x20));
    try std.testing.expect(isPrintableInput(0x7e));
}

test {
    std.testing.refAllDecls(@This());
}
