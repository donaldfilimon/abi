//! Shared REPL state types used by `repl.zig` and extracted helpers
//! (`repl_session.zig`). Keeping these in a leaf module avoids circular
//! imports when session serialize/load lives beside the interactive loop.
const std = @import("std");
const models = @import("../ai/models.zig");
const time = @import("../../foundation/time.zig");
const cmds = @import("repl_commands.zig");

pub const MODEL_STORAGE_BYTES = cmds.MODEL_STORAGE_BYTES;

/// Outcome of dispatching one input line. Lets the raw-mode and line-mode
/// loops (`repl_io.zig`) and the completion path (`repl_complete.zig`) share
/// identical handling and quit semantics.
pub const LineOutcome = enum { keep_going, quit };

/// Callback type for dispatching a plugin slash-command. Takes the plugin name,
/// command name, and argument text; returns the output to display. `null` means
/// plugin dispatch is unavailable (e.g. TUI feature disabled at build time).
pub const PluginDispatchFn = *const fn (allocator: std.mem.Allocator, plugin: []const u8, cmd_name: []const u8, arg: []const u8) anyerror![]u8;

pub const ReplConfig = struct {
    model: []const u8 = models.default_model,
    store_turns: bool = true,
    prompt_prefix: []const u8 = "> ",
    /// Slash-commands registered by plugins. Built from the Registry at startup.
    plugin_commands: []const cmds.PluginSlashCommand = &.{},
    /// Optional dispatch callback for executing plugin slash-commands.
    /// When non-null, `runPluginCommand` routes through this instead of
    /// printing a stub acknowledgment.
    plugin_dispatch: ?PluginDispatchFn = null,
    /// Context snippets from plugin context providers, injected before every
    /// completion. Owned by the caller; the REPL reads but does not free this.
    context_snippets: []const u8 = "",
    /// Whether SEA self-learning mode is active. When true, completions route
    /// through `runLearnLoop` for evidence-augmented output.
    learn_mode: bool = false,
};

pub const MAX_TURN_HISTORY: usize = 10;

/// Maximum bytes for file context loaded via '/open' (was 64 KiB, now unified).
pub const OPEN_FILE_BUDGET_BYTES: usize = 32 * 1024;

/// Maximum bytes for '@file' mention resolution (was 16 KiB, now shared).
pub const MENTION_FILE_BUDGET_BYTES: usize = 16 * 1024;

/// Approximate token estimate (4 chars per token) for context budgeting.
pub fn estimateTokens(text: []const u8) usize {
    return (text.len + 3) / 4;
}

pub const TurnEntry = struct {
    input: []const u8,
    response: []const u8,
};

pub const ReplState = struct {
    config: ReplConfig,
    turn_count: usize,
    session_id: i64,
    /// Backing storage for a model id set at runtime via `/model`, so
    /// `config.model` never dangles into the transient stdin read buffer.
    model_storage: [MODEL_STORAGE_BYTES]u8 = undefined,
    /// File context from `/open <path>`: the opened file path (owned by
    /// `file_context_buf`) and the loaded contents (owned by `file_context_buf`).
    open_path: []const u8 = "",
    open_content: []const u8 = "",
    /// Backing allocation for file context strings; freed on next `/open` or reset.
    file_context_buf: ?[]u8 = null,
    /// `/pane` split view: when true, each completed turn renders as a
    /// two-column block (chat left, `git diff --stat` right) instead of the
    /// plain streamed print. `pane_cols` is the terminal width captured at
    /// toggle time; a later resize takes effect on the next `/pane`.
    pane_mode: bool = false,
    pane_cols: usize = 0,
    /// Ring buffer of recent (input, response) pairs for multi-turn context.
    turn_history: [MAX_TURN_HISTORY]TurnEntry = @splat(TurnEntry{ .input = "", .response = "" }),
    turn_history_count: usize = 0,
    turn_history_head: usize = 0,

    pub fn init(config: ReplConfig) ReplState {
        return .{
            .config = config,
            .turn_count = 0,
            .session_id = time.unixMs(),
        };
    }

    pub fn pushTurn(self: *ReplState, allocator: std.mem.Allocator, input: []const u8, response: []const u8) void {
        // Free the oldest entry if we're overwriting it
        if (self.turn_history_count == MAX_TURN_HISTORY) {
            const old = &self.turn_history[self.turn_history_head];
            allocator.free(old.input);
            allocator.free(old.response);
        }
        self.turn_history[self.turn_history_head] = .{
            .input = allocator.dupe(u8, input) catch @panic("OOM"),
            .response = allocator.dupe(u8, response) catch @panic("OOM"),
        };
        self.turn_history_head = (self.turn_history_head + 1) % MAX_TURN_HISTORY;
        if (self.turn_history_count < MAX_TURN_HISTORY) self.turn_history_count += 1;
    }

    pub fn clearFileContext(self: *ReplState, allocator: std.mem.Allocator) void {
        if (self.file_context_buf) |buf| allocator.free(buf);
        self.open_path = "";
        self.open_content = "";
        self.file_context_buf = null;
    }

    pub fn clearTurnHistory(self: *ReplState, allocator: std.mem.Allocator) void {
        var i: usize = 0;
        while (i < self.turn_history_count) : (i += 1) {
            const idx = (self.turn_history_head + MAX_TURN_HISTORY - self.turn_history_count + i) % MAX_TURN_HISTORY;
            allocator.free(self.turn_history[idx].input);
            allocator.free(self.turn_history[idx].response);
        }
        self.turn_history_count = 0;
        self.turn_history_head = 0;
    }
};

test {
    std.testing.refAllDecls(@This());
}
