const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const types = @import("types.zig");
const sanitize = @import("sanitize.zig");

/// Disabled-TUI REPL surface. Mirrors `repl.zig`'s public names so
/// `zig build check-parity` holds; `ReplLoop.run` refuses with
/// `error.FeatureDisabled`.
pub const PluginSlashCommand = struct {
    name: []const u8,
    summary: []const u8,
    plugin: []const u8,
    aliases: []const []const u8 = &.{},
};

pub const PluginDispatchFn = *const fn (allocator: std.mem.Allocator, plugin: []const u8, cmd_name: []const u8, arg: []const u8) anyerror![]u8;

pub fn matchPluginCommandToken(token: []const u8, plugin_cmds: []const PluginSlashCommand) ?PluginSlashCommand {
    for (plugin_cmds) |cmd| {
        if (std.mem.eql(u8, token, cmd.name)) return cmd;
        for (cmd.aliases) |alias| {
            if (std.mem.eql(u8, token, alias)) return cmd;
        }
    }
    return null;
}

pub fn printPluginHelp(plugin_cmds: []const PluginSlashCommand) void {
    _ = plugin_cmds;
}

pub fn formatContextStatus(allocator: std.mem.Allocator, open_path: []const u8, open_content: []const u8, turn_count: usize, history_count: usize, turn_history_preview: []const u8) ![]u8 {
    _ = open_path;
    _ = open_content;
    _ = turn_count;
    _ = history_count;
    _ = turn_history_preview;
    return allocator.dupe(u8, "context: TUI feature is disabled");
}

pub fn printHelpWithPlugins(plugin_cmds: []const PluginSlashCommand) void {
    _ = plugin_cmds;
}

pub const MAX_TURN_HISTORY = 10;

pub const TurnEntry = struct {
    input: []const u8 = "",
    response: []const u8 = "",
};

/// Disabled-TUI session file shape (parity with `repl_session.SessionFile`).
pub const SessionFile = struct {
    version: u32 = 0,
    session_id: i64 = 0,
    model: []const u8 = "",
    learn_mode: bool = false,
    open_path: []const u8 = "",
    open_content: []const u8 = "",
    turn_count: usize = 0,
    turn_history_count: usize = 0,
    turn_history_head: usize = 0,
    turns: []const TurnData = &.{},

    pub const TurnData = struct {
        input: []const u8 = "",
        response: []const u8 = "",
    };
};

pub const ReplConfig = struct {
    model: []const u8 = "abi-local",
    store_turns: bool = true,
    prompt_prefix: []const u8 = "> ",
    plugin_commands: []const PluginSlashCommand = &.{},
    plugin_dispatch: ?PluginDispatchFn = null,
    context_snippets: []const u8 = "",
    learn_mode: bool = false,
};

pub const ReplState = struct {
    config: ReplConfig,
    turn_count: usize = 0,
    session_id: i64 = 0,

    pub fn init(config: ReplConfig) ReplState {
        return .{ .config = config, .turn_count = 0, .session_id = 0 };
    }
};

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

    pub fn run(self: *ReplLoop, io: std.Io) !void {
        _ = self;
        _ = io;
        return error.FeatureDisabled;
    }
};

/// Namespace mirror of `repl.zig` so `tui.repl.*` resolves under the stub too.
/// Leaf modules re-exported by `mod.zig` for discovery; stubs keep parity.
pub const repl_types = struct {};
pub const repl_session = struct {};
pub const repl_git_commands = struct {};
pub const repl_io = struct {};
pub const repl_complete = struct {};

pub const repl = struct {
    pub const SpecialCommand = enum { quit, reset, help, model, profile, status, history, context, syncclis, open, diff, commit, features, learn, save, load, sessions, clear, unknown };

    pub fn parseSpecialCommand(line: []const u8) SpecialCommand {
        _ = line;
        return .unknown;
    }
};

pub const Status = types.Status;
pub const Item = types.Item;
pub const State = types.State;
pub const ScreenState = types.ScreenState;
pub const PaneKind = types.PaneKind;
pub const DiagPane = types.DiagPane;
pub const DashboardPaneMeta = types.DashboardPaneMeta;
pub const DashboardState = types.DashboardState;
pub const FocusedPane = types.FocusedPane;
pub const DASHBOARD_PANES = types.DASHBOARD_PANES;
pub const DASHBOARD_PANE_COUNT = types.DASHBOARD_PANE_COUNT;

pub const DiagnosticRenderOptions = struct {
    color: bool = true,
    refresh_interval_ms: u64 = 1000,
    compact: bool = false,
};

pub fn dashboardPaneIndexForKey(key: u8) ?usize {
    for (DASHBOARD_PANES, 0..) |pane, idx| {
        if (pane.hotkey == key) return idx;
    }
    return null;
}

pub fn dashboardPaneName(kind: PaneKind) []const u8 {
    return switch (kind) {
        .system => "system",
        .plugins => "plugins",
        .storage => "storage",
        .scheduler => "scheduler",
        .memory => "memory",
        .agent_output => "agent_output",
    };
}

pub fn dashboardPaneIndexForToken(token: []const u8) ?usize {
    if (token.len == 1) {
        if (dashboardPaneIndexForKey(token[0])) |idx| return idx;
    }
    for (DASHBOARD_PANES, 0..) |pane, idx| {
        if (std.ascii.eqlIgnoreCase(token, dashboardPaneName(pane.kind))) return idx;
        if (pane.kind == .storage and std.ascii.eqlIgnoreCase(token, "wdbx")) return idx;
    }
    return null;
}

pub fn nextDashboardPane(current: usize, key: u8) ?usize {
    if (dashboardPaneIndexForKey(key)) |idx| return idx;
    if (key == 'l' or key == 'L' or key == '>') return (current + 1) % DASHBOARD_PANE_COUNT;
    if (key == 'h' or key == 'H' or key == '<') return (current + DASHBOARD_PANE_COUNT - 1) % DASHBOARD_PANE_COUNT;
    return null;
}

pub fn stdinFd() std.posix.fd_t {
    return std.Io.File.stdin().handle;
}

pub const InteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !InteractiveTerminal {
        _ = fd;
        return error.NotATerminal;
    }

    pub fn deinit(self: *InteractiveTerminal) void {
        _ = self;
    }

    pub fn readKey(self: *InteractiveTerminal) ?u8 {
        _ = self;
        return null;
    }

    pub fn pollInput(self: *InteractiveTerminal, timeout_ms: i32) bool {
        _ = self;
        _ = timeout_ms;
        return false;
    }
};

pub const ScreenSession = struct {
    term: InteractiveTerminal,
    screen_active: bool = false,

    pub fn init(fd: std.posix.fd_t) !ScreenSession {
        return .{ .term = try InteractiveTerminal.init(fd) };
    }

    pub fn deinit(self: *ScreenSession) void {
        _ = self;
    }
};

pub fn initScreen() !void {}
pub fn initScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearScreen() !void {}
pub fn homeScreen() void {}
pub fn homeScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearToEnd() void {}
pub fn clearToEndWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn render(state: ScreenState) !void {
    _ = state;
}
pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    _ = writer;
    _ = state;
}
pub fn deinitScreen() void {}
pub fn deinitScreenWriter(writer: anytype) !void {
    _ = writer;
}

/// Output sanitizer mirror. The disabled build still neutralizes terminal control
/// characters — a pure safety utility has no feature dependency, so it degrades to
/// identical behavior rather than refusing. The input is walked as UTF-8: a valid
/// sequence whose codepoint is a control (U+0000–U+001F, U+007F DEL, or the
/// U+0080–U+009F C1 range, incl. 0x9B CSI) has every byte replaced with '.';
/// otherwise the sequence is copied verbatim so multi-byte UTF-8 survives. Bytes
/// that do not form a valid UTF-8 sequence (a lone C1 like 0x9B, a stray
/// continuation byte, or a truncated sequence) are replaced one-for-one with '.'.
/// The output length always equals the input length. Caller owns the returned
/// slice.
pub fn sanitizeControlBytes(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return sanitize.sanitizeControlBytes(allocator, input);
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;
    return try allocator.dupe(u8, "TUI feature is disabled");
}

pub fn renderDiagnostics(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    return renderDiagnosticsWithOptions(allocator, ds, .{});
}

pub fn renderDiagnosticsWithOptions(allocator: std.mem.Allocator, ds: DashboardState, options: DiagnosticRenderOptions) ![]u8 {
    _ = ds;
    _ = options;
    return try allocator.dupe(u8, "TUI diagnostics are disabled in this build");
}

pub fn renderDiagnosticsSplit(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    return renderDiagnosticsSplitWithOptions(allocator, ds, .{});
}

pub fn renderDiagnosticsSplitWithOptions(allocator: std.mem.Allocator, ds: DashboardState, options: DiagnosticRenderOptions) ![]u8 {
    _ = ds;
    _ = options;
    return try allocator.dupe(u8, "TUI diagnostics split mode is disabled in this build");
}

pub fn formatAgentStatusDigest(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    _ = ds;
    return try allocator.dupe(u8, "TUI feature is disabled");
}

pub fn writeDashboard(writer: anytype, allocator: std.mem.Allocator, state: State) !void {
    const rendered = try renderDashboard(allocator, state);
    defer allocator.free(rendered);
    try writer.writeAll(rendered);
}

pub fn writeDiagnostics(writer: anytype, allocator: std.mem.Allocator, ds: DashboardState) !void {
    const rendered = try renderDiagnostics(allocator, ds);
    defer allocator.free(rendered);
    try writer.writeAll(rendered);
}

pub fn isQuitKey(byte: u8) bool {
    return byte == 'q' or byte == 'Q' or byte == 0x1b or byte == 0x03;
}

pub fn isRefreshKey(byte: u8) bool {
    return byte == 'r' or byte == 'R';
}

pub fn isTabKey(byte: u8) bool {
    return byte == 0x09;
}

pub fn isScrollUpKey(byte: u8) bool {
    return byte == 'k' or byte == 'K';
}

pub fn isScrollDownKey(byte: u8) bool {
    return byte == 'j' or byte == 'J';
}

test {
    std.testing.refAllDecls(@This());
}

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}

pub fn dashboardHealth(ds: DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (ds.gpu_accelerated and ds.gpu_linked) return "nominal";
    return "cpu";
}
