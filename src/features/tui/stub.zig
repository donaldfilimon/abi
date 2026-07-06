const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const scheduler_mod = @import("../../core/scheduler.zig");
const types = @import("types.zig");
const sanitize = @import("sanitize.zig");

/// Disabled-TUI REPL surface. Mirrors `repl.zig`'s public names so
/// `zig build check-parity` holds; `ReplLoop.run` refuses with
/// `error.FeatureDisabled`.
pub const ReplConfig = struct {
    model: []const u8 = "abi-local",
    store_turns: bool = true,
    prompt_prefix: []const u8 = "> ",
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
pub const repl = struct {
    pub const SpecialCommand = enum { quit, reset, help, model, profile, history, syncclis, unknown };

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
pub const DashboardState = types.DashboardState;

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

pub fn initScreen() !void {}
pub fn initScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearScreen() !void {}
pub fn homeScreen() void {}
pub fn clearToEnd() void {}
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
    _ = ds;
    return try allocator.dupe(u8, "TUI diagnostics are disabled in this build");
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
    return byte == 'q' or byte == 'Q' or byte == 0x1b;
}

pub fn isRefreshKey(byte: u8) bool {
    return byte == 'r' or byte == 'R';
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
