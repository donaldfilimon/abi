//! Interactive TUI REPL dispatch hub.
//!
//! Owns the `ReplLoop` and routes each input line: slash-commands go to their
//! handlers, ordinary input runs the AI completion path against the ambient
//! WDBX store.
//!
//! The surrounding leaves keep this module thin: slash-command parsing,
//! completion, and status/history formatting live in `repl_commands.zig`; the
//! raw-mode/line-mode terminal input loops in `repl_io.zig`; the prompt
//! completion path (SEA / local-bridge SSE / in-process streaming) in
//! `repl_complete.zig`; git and session subprocess work in
//! `repl_git_commands.zig`; session save/load in `repl_session.zig`.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("../ai/models.zig");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const time = @import("../../foundation/time.zig");
const sanitize = @import("sanitize.zig");
const terminal = @import("terminal.zig");
const line_editor = @import("line_editor.zig");
const cmds = @import("repl_commands.zig");
const repl_types = @import("repl_types.zig");
const repl_session = @import("repl_session.zig");
const git_cmds = @import("repl_git_commands.zig");
const repl_io = @import("repl_io.zig");
const repl_complete = @import("repl_complete.zig");

pub const MODEL_STORAGE_BYTES = repl_types.MODEL_STORAGE_BYTES;
pub const SpecialCommand = cmds.SpecialCommand;
pub const SlashCommand = cmds.SlashCommand;
pub const slash_commands = cmds.slash_commands;
pub const SlashCompletion = cmds.SlashCompletion;
pub const parseSpecialCommand = cmds.parseSpecialCommand;
pub const specialArg = cmds.specialArg;
pub const completeSlashCommand = cmds.completeSlashCommand;
pub const formatHistoryHeader = cmds.formatHistoryHeader;
pub const formatStatusLine = cmds.formatStatusLine;
pub const formatOpenStatus = cmds.formatOpenStatus;
pub const formatContextStatus = cmds.formatContextStatus;
pub const validModelId = cmds.validModelId;
pub const printHelp = cmds.printHelp;
pub const printHelpWithPlugins = cmds.printHelpWithPlugins;
pub const printPluginHelp = cmds.printPluginHelp;
pub const PluginSlashCommand = cmds.PluginSlashCommand;
pub const matchPluginCommandToken = cmds.matchPluginCommandToken;

pub const PluginDispatchFn = repl_types.PluginDispatchFn;
pub const ReplConfig = repl_types.ReplConfig;
pub const MAX_TURN_HISTORY = repl_types.MAX_TURN_HISTORY;
pub const TurnEntry = repl_types.TurnEntry;
pub const ReplState = repl_types.ReplState;
pub const SessionFile = repl_session.SessionFile;

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
        self.state.clearFileContext(self.allocator);
        self.state.clearTurnHistory(self.allocator);
    }

    /// Outcome of dispatching one input line. Lets the raw-mode and line-mode
    /// loops share identical handling and quit semantics.
    const LineOutcome = repl_types.LineOutcome;

    /// Run the REPL until `/quit` or EOF (Ctrl-D). Phase 2: attempt the raw-mode
    /// `InteractiveTerminal` for nicer character-at-a-time input, and fall back
    /// to the Phase-1 line-at-a-time path when the terminal cannot enter raw mode
    /// (CI / non-tty), mirroring `dashboard.zig`'s fallback. Behavior under a
    /// non-tty is byte-for-byte the Phase-1 path so the `agent tui` contract is
    /// unchanged. Both loops live in `repl_io.zig` and call back into
    /// `dispatchLine`.
    pub fn run(self: *ReplLoop, io: std.Io) !void {
        var term = terminal.InteractiveTerminal.init(terminal.stdinFd()) catch {
            return repl_io.runLineMode(self, io);
        };
        defer term.deinit();
        return repl_io.runRawMode(self, &term, io);
    }

    /// Dispatch one already-trimmed, non-empty input line: slash-commands route
    /// to their handlers, ordinary text runs a completion and persists the turn.
    /// Public so the input loops in `repl_io.zig` can call back into the hub.
    pub fn dispatchLine(self: *ReplLoop, line: []const u8, io: std.Io) !LineOutcome {
        if (line[0] == '/') return self.dispatchSlashCommand(line, io);
        return self.completePrompt(line, io);
    }

    fn dispatchSlashCommand(self: *ReplLoop, line: []const u8, io: std.Io) !LineOutcome {
        switch (parseSpecialCommand(line)) {
            .quit => return .quit,
            .help => printHelpWithPlugins(self.state.config.plugin_commands),
            .model => self.applyModel(specialArg(line)),
            .reset => self.resetSession(),
            .status => self.showStatus(),
            .history => self.showHistory(),
            .context => self.showContext(),
            .profile => self.showProfileStatus(),
            .syncclis => try self.runSyncClis(io),
            .open => try self.runOpen(specialArg(line), io),
            .diff => try self.runDiff(specialArg(line), io),
            .features => self.showFeatures(),
            .learn => self.toggleLearn(),
            .save => try self.saveSession(specialArg(line), io),
            .load => try self.loadSession(specialArg(line), io),
            .sessions => try self.listSessions(io),
            .clear => self.clearScreen(),
            .commit => try self.runCommit(io),
            .unknown => {
                // Check plugin commands
                const token = blk: {
                    const body = line[1..];
                    const end = std.mem.indexOfAny(u8, body, " \t\r") orelse body.len;
                    break :blk body[0..end];
                };
                if (cmds.matchPluginCommandToken(token, self.state.config.plugin_commands)) |plugin_cmd| {
                    return self.runPluginCommand(plugin_cmd, specialArg(line));
                }
                try self.printUnknownCommand(line);
            },
        }
        return .keep_going;
    }

    fn persistedBlockCount(self: *ReplLoop) usize {
        if (comptime build_options.feat_wdbx) return self.store.blockCount();
        return 0;
    }

    fn showStatus(self: *ReplLoop) void {
        var buf: [512]u8 = undefined;
        std.debug.print("{s}\n", .{formatStatusLine(
            buf[0..256],
            self.state.session_id,
            self.state.turn_count,
            self.state.config.model,
            self.state.config.store_turns,
            self.persistedBlockCount(),
        )});
        std.debug.print("  session:  {d}\n", .{self.state.session_id});
        std.debug.print("  model:    {s}\n", .{self.state.config.model});
        std.debug.print("  provider: {s}\n", .{models.providerOf(self.state.config.model).label()});
        std.debug.print("  turns:    {d} in session, {d} in history\n", .{ self.state.turn_count, self.state.turn_history_count });
        std.debug.print("  sea:      {s}\n", .{if (self.state.config.learn_mode) "on (self-learning)" else "off"});
        std.debug.print("  store:    {s}\n", .{if (self.state.config.store_turns) "on" else "off"});
        if (self.state.open_path.len > 0) {
            std.debug.print("  file:     {s} ({d} bytes)\n", .{ self.state.open_path, self.state.open_content.len });
        } else {
            std.debug.print("  file:     (none)\n", .{});
        }
        const context_active = self.state.config.context_snippets.len > 0;
        std.debug.print("  context:  {s}\n", .{if (context_active) "loaded" else "none"});
        std.debug.print("  blocks:   {d}\n", .{self.persistedBlockCount()});
    }

    /// `/sync-clis`: execute the central sync-clis launcher (in-repo canonical
    /// first, then synced copies; never executes a missing script).
    fn runSyncClis(self: *ReplLoop, io: std.Io) !void {
        try git_cmds.runSyncClis(self.allocator, io);
    }

    fn printUnknownCommand(self: *ReplLoop, line: []const u8) !void {
        // Echo the rejected command through the sanitizer: the raw line is
        // attacker-controlled and may carry ESC/control bytes.
        const safe_line = try sanitize.sanitizeControlBytes(self.allocator, line);
        defer self.allocator.free(safe_line);
        std.debug.print("unknown command: {s} (try /help)\n", .{safe_line});
    }

    /// Ordinary (non-slash) input: run the completion path in
    /// `repl_complete.zig` (SEA / local-bridge SSE / in-process streaming).
    fn completePrompt(self: *ReplLoop, line: []const u8, io: std.Io) !LineOutcome {
        return repl_complete.completePrompt(self.allocator, self.store, self.scheduler, &self.state, line, io);
    }

    /// `/reset`: clear the per-session turn counter and start a fresh session
    /// context (a new session id), so subsequent `/history` summaries describe
    /// only turns after the reset.
    fn resetSession(self: *ReplLoop) void {
        self.state.turn_count = 0;
        self.state.session_id = time.unixMs();
        self.state.clearFileContext(self.allocator);
        self.state.clearTurnHistory(self.allocator);
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

    /// `/context`: show current context state (open file, turn history, preview).
    fn showContext(self: *ReplLoop) void {
        const preview = cmds.formatTurnHistoryPreview(
            self.allocator,
            &self.state.turn_history,
            self.state.turn_history_count,
            self.state.turn_history_head,
        ) catch &.{};
        defer self.allocator.free(preview);

        const status = formatContextStatus(
            self.allocator,
            self.state.open_path,
            self.state.open_content,
            self.state.turn_count,
            self.state.turn_history_count,
            preview,
        ) catch {
            std.debug.print("context: status unavailable\n", .{});
            return;
        };
        defer self.allocator.free(status);
        std.debug.print("{s}\n", .{status});
    }

    /// Dispatch a plugin-registered slash-command. Routes through
    /// `config.plugin_dispatch` when set (real plugin execution), otherwise
    /// prints a stub acknowledgment.
    fn runPluginCommand(self: *ReplLoop, cmd: cmds.PluginSlashCommand, arg: []const u8) !LineOutcome {
        if (self.state.config.plugin_dispatch) |dispatch| {
            const output = dispatch(self.allocator, cmd.plugin, cmd.name, arg) catch |err| {
                std.debug.print("plugin command /{s} failed: {s}\n", .{ cmd.name, @errorName(err) });
                return .keep_going;
            };
            defer self.allocator.free(output);
            std.debug.print("{s}\n", .{output});
        } else {
            const ack = try cmds.formatPluginCommandAck(self.allocator, cmd, arg);
            defer self.allocator.free(ack);
            std.debug.print("{s}\n", .{ack});
        }
        return .keep_going;
    }

    fn showProfileStatus(self: *ReplLoop) void {
        std.debug.print("profile: adaptive router active; model={s}; turns={d}\n", .{
            self.state.config.model,
            self.state.turn_count,
        });
    }

    /// `/open <path>`: read a file into the prompt context. The content is stored
    /// and injected before the next completion.
    fn runOpen(self: *ReplLoop, path: []const u8, io: std.Io) !void {
        try git_cmds.runOpen(self.allocator, &self.state, path, io);
    }

    /// `/diff [--stat]`: run `git diff` and print the output. When `--stat` is
    /// passed, shows a summary of changed files. Output is colorized with ANSI
    /// codes for added (+) lines in green and removed (-) lines in red.
    fn runDiff(self: *ReplLoop, arg: []const u8, io: std.Io) !void {
        try git_cmds.runDiff(self.allocator, arg, io);
    }

    /// `/commit`: stage all changes and create a commit. Prompts for a commit
    /// message interactively via the next REPL input line.
    fn runCommit(self: *ReplLoop, io: std.Io) !void {
        try git_cmds.runCommit(self.allocator, io);
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

    /// `/features`: display compile-time feature flags with ✓ or empty markers.
    /// These are build gates (`-Dfeat-*`), not runtime `available`/`native_dispatch`
    /// capability bits — see `abi backends` for honesty on GPU/accelerator stubs.
    fn showFeatures(self: *ReplLoop) void {
        _ = self;
        const Feature = struct { name: []const u8, active: bool, desc: []const u8 };
        const features = [_]Feature{
            .{ .name = "ai", .active = build_options.feat_ai, .desc = "AI profiles, routing, constitution" },
            .{ .name = "wdbx", .active = build_options.feat_wdbx, .desc = "Vector store, HNSW index" },
            .{ .name = "sea", .active = build_options.feat_sea, .desc = "Self-learning evidence loop" },
            .{ .name = "gpu", .active = build_options.feat_gpu, .desc = "GPU feature linked (runtime accel separate)" },
            .{ .name = "tui", .active = build_options.feat_tui, .desc = "Terminal UI / diagnostics dashboard" },
            .{ .name = "accelerator", .active = build_options.feat_accelerator, .desc = "Accelerator selection report (no native dispatch)" },
            .{ .name = "shaders", .active = build_options.feat_shader, .desc = "Shader validate+checksum (no compiler)" },
            .{ .name = "mlir", .active = build_options.feat_mlir, .desc = "Textual MLIR lower (no LLVM toolchain)" },
            .{ .name = "mobile", .active = build_options.feat_mobile, .desc = "Mobile profile report (no runtime)" },
            .{ .name = "os_control", .active = build_options.feat_os_control, .desc = "OS command policy controls" },
            .{ .name = "hash", .active = build_options.feat_hash, .desc = "Portable hashing utilities" },
            .{ .name = "metrics", .active = build_options.feat_metrics, .desc = "In-process metrics" },
            .{ .name = "telemetry", .active = build_options.feat_telemetry, .desc = "Telemetry event emission" },
            .{ .name = "nn", .active = build_options.feat_nn, .desc = "Char-LM neural net trainer" },
            .{ .name = "foundationmodels", .active = build_options.feat_foundationmodels, .desc = "FoundationModels (macOS)" },
        };
        std.debug.print("Build-time feature flags (-Dfeat-*):\n", .{});
        for (features) |f| {
            const marker = if (f.active) "✓" else " ";
            std.debug.print("  {s:<18} [{s}]  {s}\n", .{ f.name, marker, f.desc });
        }
        std.debug.print("(compile-on only — use `abi backends` for runtime GPU/accelerator status)\n", .{});
    }

    /// `/learn`: toggle SEA self-learning mode on/off.
    fn toggleLearn(self: *ReplLoop) void {
        self.state.config.learn_mode = !self.state.config.learn_mode;
        if (!build_options.feat_sea and self.state.config.learn_mode) {
            std.debug.print("SEA learning: on (feature disabled — falling back to plain completion)\n", .{});
        } else {
            std.debug.print("SEA learning: {s}\n", .{if (self.state.config.learn_mode) "on" else "off"});
        }
    }

    fn sessionsDir(self: *ReplLoop) ![]const u8 {
        return try repl_session.sessionsDir(self.allocator);
    }

    /// `/save <name>`: serialize session state to ~/.abi/sessions/<name>.json.
    fn saveSession(self: *ReplLoop, name: []const u8, io: std.Io) !void {
        try repl_session.saveSession(self.allocator, &self.state, name, io);
    }

    /// `/load <name>`: restore session state from ~/.abi/sessions/<name>.json.
    fn loadSession(self: *ReplLoop, name: []const u8, io: std.Io) !void {
        try repl_session.loadSession(self.allocator, &self.state, name, io);
    }

    /// `/sessions`: list saved session files in ~/.abi/sessions/.
    fn listSessions(self: *ReplLoop, io: std.Io) !void {
        try git_cmds.listSessions(self.allocator, io);
    }

    /// `/clear`: clear the terminal screen and redraw the prompt.
    fn clearScreen(self: *ReplLoop) void {
        _ = self;
        std.debug.print("\x1b[2J\x1b[H", .{});
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
    try std.testing.expectEqual(SpecialCommand.context, parseSpecialCommand("/context"));
    try std.testing.expectEqual(SpecialCommand.features, parseSpecialCommand("/features"));
    try std.testing.expectEqual(SpecialCommand.features, parseSpecialCommand("/feat"));
    try std.testing.expectEqual(SpecialCommand.learn, parseSpecialCommand("/learn"));
    try std.testing.expectEqual(SpecialCommand.save, parseSpecialCommand("/save my-session"));
    try std.testing.expectEqual(SpecialCommand.load, parseSpecialCommand("/load my-session"));
    try std.testing.expectEqual(SpecialCommand.sessions, parseSpecialCommand("/sessions"));
    try std.testing.expectEqual(SpecialCommand.sessions, parseSpecialCommand("/ls-sessions"));
    try std.testing.expectEqual(SpecialCommand.clear, parseSpecialCommand("/clear"));
    try std.testing.expectEqual(SpecialCommand.clear, parseSpecialCommand("/cls"));
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

test "tui input hardening: adversarial bytes never corrupt state" {
    const adversarial = [_][]const u8{
        "",
        "\x00",
        "\x1b[A\x1b[B",
        "\xff\xfe",
        "quit\x00extra",
        "/quit\x00extra",
    };
    for (adversarial) |input| {
        switch (parseSpecialCommand(input)) {
            .quit, .reset, .help, .model, .profile, .status, .history, .context, .syncclis, .open, .diff, .commit, .features, .learn, .save, .load, .sessions, .clear, .unknown => {},
        }
    }

    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*b, i| b.* = @intCast(i);
    _ = parseSpecialCommand(&all_bytes);

    const overlong = try std.testing.allocator.alloc(u8, line_editor.MAX_LINE_BYTES * 2);
    defer std.testing.allocator.free(overlong);
    @memset(overlong, '/');
    _ = parseSpecialCommand(overlong);
    try std.testing.expect(overlong.len > line_editor.MAX_LINE_BYTES);

    const dirty = "\x1b[31mred\x00\x07\x7fboom\x1b[0m";
    const clean = try sanitize.sanitizeControlBytes(std.testing.allocator, dirty);
    defer std.testing.allocator.free(clean);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x1b) == null);
    try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x00) == null);
    try std.testing.expectEqual(dirty.len, clean.len);
}

test "save and load session round-trip restores state" {
    if (!build_options.feat_wdbx) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    var scheduler = scheduler_mod.Scheduler.init(allocator);
    defer scheduler.deinit();

    var repl = ReplLoop.init(allocator, &store, &scheduler, .{
        .model = "test-model",
        .learn_mode = true,
        .store_turns = false,
    });
    defer repl.deinit();

    // Seed some state
    repl.state.turn_count = 5;
    repl.state.session_id = 123456;
    repl.state.config.learn_mode = true;
    repl.state.pushTurn(allocator, "hello", "world");
    repl.state.pushTurn(allocator, "foo", "bar");

    // Serialization lives in repl_session; exercise it through the loop state.
    const json = try repl_session.serializeSession(allocator, &repl.state);
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(SessionFile, allocator, json, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 1), parsed.value.version);
    try std.testing.expectEqual(@as(i64, 123456), parsed.value.session_id);
    try std.testing.expectEqualStrings("test-model", parsed.value.model);
    try std.testing.expect(parsed.value.learn_mode);
    try std.testing.expectEqual(@as(usize, 5), parsed.value.turn_count);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.turn_history_count);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.turns.len);
    try std.testing.expectEqualStrings("hello", parsed.value.turns[0].input);
    try std.testing.expectEqualStrings("world", parsed.value.turns[0].response);
    try std.testing.expectEqualStrings("foo", parsed.value.turns[1].input);
    try std.testing.expectEqualStrings("bar", parsed.value.turns[1].response);
}

test {
    std.testing.refAllDecls(@This());
}
