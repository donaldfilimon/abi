const std = @import("std");
const builtin = @import("builtin");

pub const repl = @import("repl.zig");
pub const ReplLoop = repl.ReplLoop;
pub const ReplState = repl.ReplState;
pub const ReplConfig = repl.ReplConfig;

pub const Status = enum {
    ready,
    busy,
    warning,
    disabled,
};

pub const Item = struct {
    label: []const u8,
    value: []const u8,
};

pub const State = struct {
    title: []const u8,
    status: Status = .ready,
    items: []const Item = &.{},
};

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

/// Pane categories for the multi-pane diagnostic dashboard.
pub const PaneKind = enum {
    system,
    plugins,
    storage,
    scheduler,
};

/// A single diagnostic pane with a title and key-value rows.
pub const DiagPane = struct {
    kind: PaneKind,
    title: []const u8,
    items: []const Item,
};

/// Full dashboard state for the interactive TUI.
pub const DashboardState = struct {
    gpu_backend: []const u8 = "unknown",
    gpu_accelerated: bool = false,
    gpu_linked: bool = false,
    plugin_count: usize = 0,
    plugin_names: []const []const u8 = &.{},
    wdbx_blocks: usize = 0,
    wdbx_vectors: usize = 0,
    wdbx_entries: usize = 0,
    wdbx_spatial_records: usize = 0,
    scheduler_source: []const u8 = "not attached",
    scheduler_running: usize = 0,
    scheduler_pending: usize = 0,
    scheduler_completed: usize = 0,
    scheduler_failed: usize = 0,
    memory_source: []const u8 = "not attached",
    memory_peak: usize = 0,
    memory_current: usize = 0,
    memory_leaked: usize = 0,
};

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}

test {
    std.testing.refAllDecls(@This());
}

/// Replace C0 control bytes (0x00–0x1F) and DEL (0x7F) in `input` with a visible
/// '.' so attacker-influenced strings interpolated into ANSI render output cannot
/// inject terminal escape sequences (ESC/CSI/OSC) or embed NUL. Bytes >= 0x80
/// pass through unchanged so legitimate multi-byte UTF-8 (box-drawing glyphs,
/// accented text) survives. The output length always equals the input length.
/// Caller owns the returned slice.
pub fn sanitizeControlBytes(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    for (input, 0..) |byte, i| {
        out[i] = if (byte < 0x20 or byte == 0x7f) '.' else byte;
    }
    return out;
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Sanitize caller-supplied strings before interpolation so control bytes in
    // the title/items cannot inject terminal escapes (see sanitizeControlBytes).
    const title = try sanitizeControlBytes(allocator, state.title);
    defer allocator.free(title);

    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "| {s:<28} |\n", .{title});
    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "status: {s}\n", .{statusText(state.status)});
    for (state.items) |item| {
        const label = try sanitizeControlBytes(allocator, item.label);
        defer allocator.free(label);
        const value = try sanitizeControlBytes(allocator, item.value);
        defer allocator.free(value);
        try out.print(allocator, "- {s}: {s}\n", .{ label, value });
    }
    try out.print(allocator, "\nCommands: abi help | abi agent train all | abi agent os dry-run <cmd>\n", .{});

    return try out.toOwnedSlice(allocator);
}

/// Render the full interactive diagnostics dashboard.
pub fn renderDiagnostics(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Externally-influenced string fields (GPU backend label, plugin names,
    // scheduler/memory source) are sanitized before they are interpolated into
    // the ANSI render stream so they cannot inject terminal escapes. Plugin names
    // are sanitized per-iteration in the loop below.
    const gpu_backend = try sanitizeControlBytes(allocator, ds.gpu_backend);
    defer allocator.free(gpu_backend);
    const scheduler_source = try sanitizeControlBytes(allocator, ds.scheduler_source);
    defer allocator.free(scheduler_source);
    const memory_source = try sanitizeControlBytes(allocator, ds.memory_source);
    defer allocator.free(memory_source);

    // Header
    try out.appendSlice(allocator, "\x1b[1;36m");
    try out.appendSlice(allocator, "╔══════════════════════════════════════════════════════════════╗\n");
    try out.appendSlice(allocator, "║              ABI Diagnostics Dashboard                      ║\n");
    try out.appendSlice(allocator, "╚══════════════════════════════════════════════════════════════╝\n");
    try out.appendSlice(allocator, "\x1b[0m");

    // System pane
    try out.appendSlice(allocator, "\x1b[1;33m┌─ System ────────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ GPU Backend:     \x1b[1m{s:<42}\x1b[0m│\n", .{gpu_backend});
    try out.print(allocator, "│ Accelerated:     \x1b[1m{s:<42}\x1b[0m│\n", .{if (ds.gpu_accelerated) "yes" else "no"});
    try out.print(allocator, "│ Native Linked:   \x1b[1m{s:<42}\x1b[0m│\n", .{if (ds.gpu_linked) "yes" else "no"});
    try out.appendSlice(allocator, "\x1b[1;33m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Plugins pane
    try out.appendSlice(allocator, "\x1b[1;32m┌─ Plugins ───────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Registered:      \x1b[1m{d:<42}\x1b[0m│\n", .{ds.plugin_count});
    for (ds.plugin_names) |name| {
        const safe_name = try sanitizeControlBytes(allocator, name);
        defer allocator.free(safe_name);
        try out.print(allocator, "│   - \x1b[1m{s:<55}\x1b[0m│\n", .{safe_name});
    }
    try out.appendSlice(allocator, "\x1b[1;32m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Storage pane
    try out.appendSlice(allocator, "\x1b[1;35m┌─ WDBX Storage ──────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Block chain:     \x1b[1m{d:<42}\x1b[0m│\n", .{ds.wdbx_blocks});
    try out.print(allocator, "│ Vectors:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.wdbx_vectors});
    try out.print(allocator, "│ KV Entries:      \x1b[1m{d:<42}\x1b[0m│\n", .{ds.wdbx_entries});
    try out.print(allocator, "│ Spatial 3D:      \x1b[1m{d:<42}\x1b[0m│\n", .{ds.wdbx_spatial_records});
    try out.appendSlice(allocator, "\x1b[1;35m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Scheduler pane
    try out.appendSlice(allocator, "\x1b[1;34m┌─ Scheduler ─────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Source:          \x1b[1m{s:<42}\x1b[0m│\n", .{scheduler_source});
    try out.print(allocator, "│ Running:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_running});
    try out.print(allocator, "│ Pending:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_pending});
    try out.print(allocator, "│ Completed:       \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_completed});
    try out.print(allocator, "│ Failed:          \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_failed});
    try out.appendSlice(allocator, "\x1b[1;34m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Memory pane
    try out.appendSlice(allocator, "\x1b[1;31m┌─ Memory ────────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Source:          \x1b[1m{s:<42}\x1b[0m│\n", .{memory_source});
    try out.print(allocator, "│ Peak:            \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_peak});
    try out.print(allocator, "│ Current:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_current});
    try out.print(allocator, "│ Leaked:          \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_leaked});
    try out.appendSlice(allocator, "\x1b[1;31m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Footer
    try out.appendSlice(allocator, "\n\x1b[2m[q/Esc] Quit  [r] Refresh\x1b[0m\n");

    return try out.toOwnedSlice(allocator);
}

// --- Interactive Terminal Helpers ---

/// Portable stdin file descriptor/handle. `std.Io.File.stdin().handle` is a
/// `std.posix.fd_t` on every target (fd 0 on POSIX, the console HANDLE on
/// Windows), avoiding the `STDIN_FILENO` comptime-int vs HANDLE mismatch.
pub fn stdinFd() std.posix.fd_t {
    return std.Io.File.stdin().handle;
}

/// Raw-mode interactive terminal. Comptime-selected per platform so non-POSIX
/// targets never instantiate the termios/poll path. The public API (init/deinit/
/// readKey/pollInput) is identical across platforms and mirrors `stub.zig`.
pub const InteractiveTerminal = if (builtin.os.tag == .windows)
    WindowsInteractiveTerminal
else
    PosixInteractiveTerminal;

/// POSIX (macOS/Linux) raw-mode terminal. libc-free: TTY detection comes from
/// `tcgetattr` failing with ENOTTY on a non-terminal, so no `isatty` extern is
/// needed (which would be an undefined symbol on a no-libc Linux link).
const PosixInteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !PosixInteractiveTerminal {
        // tcgetattr fails with ENOTTY on a non-tty; treat that as the
        // "not a terminal" signal and let callers fall back to line mode.
        const original = std.posix.tcgetattr(fd) catch return error.NotATerminal;
        var raw = original;
        raw.lflag.ICANON = false;
        raw.lflag.ECHO = false;

        const vmin = if (@hasDecl(std.posix, "VMIN")) std.posix.VMIN else std.posix.system.V.MIN;
        const vtime = if (@hasDecl(std.posix, "VTIME")) std.posix.VTIME else std.posix.system.V.TIME;

        raw.cc[@intFromEnum(vmin)] = 1;
        raw.cc[@intFromEnum(vtime)] = 0;

        try std.posix.tcsetattr(fd, .FLUSH, raw);
        return .{ .fd = fd, .original = original, .is_tty = true };
    }

    pub fn deinit(self: *PosixInteractiveTerminal) void {
        std.posix.tcsetattr(self.fd, .FLUSH, self.original) catch |err| {
            std.log.warn("failed to restore terminal: {s}", .{@errorName(err)});
        };
    }

    pub fn readKey(self: *PosixInteractiveTerminal) ?u8 {
        var buf: [1]u8 = undefined;
        const n = std.posix.read(self.fd, &buf) catch |err| {
            std.log.warn("read stdin failed: {s}", .{@errorName(err)});
            return null;
        };
        if (n == 0) return null;
        return buf[0];
    }

    /// Block up to `timeout_ms` for input to become readable (a negative
    /// timeout blocks indefinitely). Returns true if a key is ready to read,
    /// false on timeout or error. This lets the dashboard loop refresh on a
    /// timer while staying responsive to keystrokes.
    pub fn pollInput(self: *PosixInteractiveTerminal, timeout_ms: i32) bool {
        var fds = [_]std.posix.pollfd{.{ .fd = self.fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const n = std.posix.poll(&fds, timeout_ms) catch return false;
        return n > 0 and (fds[0].revents & std.posix.POLL.IN) != 0;
    }
};

/// Windows terminal: minimal first cut (option b). There is no POSIX termios/
/// poll on Windows, so `init` reports `error.NotATerminal` and the REPL/dashboard
/// engage their existing line-mode fallbacks, preserving the `agent tui` clean-
/// exit contract. A real Console-API raw mode (GetConsoleMode/SetConsoleMode +
/// ReadConsoleInput) can replace this later; the ANSI render helpers below are
/// already portable.
const WindowsInteractiveTerminal = struct {
    fd: std.posix.fd_t,
    is_tty: bool = false,

    pub fn init(fd: std.posix.fd_t) !WindowsInteractiveTerminal {
        _ = fd;
        return error.NotATerminal;
    }

    pub fn deinit(self: *WindowsInteractiveTerminal) void {
        _ = self;
    }

    pub fn readKey(self: *WindowsInteractiveTerminal) ?u8 {
        _ = self;
        return null;
    }

    pub fn pollInput(self: *WindowsInteractiveTerminal, timeout_ms: i32) bool {
        _ = self;
        _ = timeout_ms;
        return false;
    }
};

/// Check if a key press is a quit command (q or Escape).
pub fn isQuitKey(byte: u8) bool {
    return byte == 'q' or byte == 'Q' or byte == 0x1b;
}

/// Check if a key press is a refresh command.
pub fn isRefreshKey(byte: u8) bool {
    return byte == 'r' or byte == 'R';
}

pub fn initScreen() !void {
    std.debug.print("\x1b[?1049h\x1b[H", .{});
}

pub fn initScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049h\x1b[H");
}

pub fn clearScreen() !void {
    std.debug.print("\x1b[2J\x1b[H", .{});
}

/// Move the cursor to the top-left without clearing for flicker-free redraws.
/// Pair with `clearToEnd` after writing a frame to wipe any stale trailing rows.
pub fn homeScreen() void {
    std.debug.print("\x1b[H", .{});
}

/// Clear from the cursor to the end of the screen (removes stale trailing rows
/// left by a shorter frame).
pub fn clearToEnd() void {
    std.debug.print("\x1b[0J", .{});
}

pub fn clearScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[2J\x1b[H");
}

pub fn render(state: ScreenState) !void {
    std.debug.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    std.debug.print("Agents: abbey, aviva, abi | WDBX: in-memory training records\n", .{});
}

pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    try writer.print("TUI Rendering at {d}x{d}\n", .{ state.width, state.height });
    try writer.writeAll("Agents: abbey, aviva, abi | WDBX: in-memory training records\n");
}

pub fn deinitScreen() void {
    std.debug.print("\x1b[?1049l", .{});
}

pub fn deinitScreenWriter(writer: anytype) !void {
    try writer.writeAll("\x1b[?1049l");
}

test "dashboard requires a title" {
    try std.testing.expectError(error.InvalidTuiState, renderDashboard(std.testing.allocator, .{ .title = "" }));
}

test "dashboard renders status and items" {
    const rendered = try renderDashboard(std.testing.allocator, .{
        .title = "ABI",
        .status = .warning,
        .items = &.{.{ .label = "AI", .value = "safe" }},
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "status: warning") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "- AI: safe") != null);
}

test "writer render functions are testable" {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(std.testing.allocator);

    const TestWriter = struct {
        allocator: std.mem.Allocator,
        buffer: *std.ArrayListUnmanaged(u8),

        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, bytes);
        }

        pub fn print(self: *@This(), comptime fmt: []const u8, args: anytype) !void {
            try self.buffer.print(self.allocator, fmt, args);
        }
    };

    var writer = TestWriter{ .allocator = std.testing.allocator, .buffer = &buf };

    try renderWriter(&writer, .{ .width = 80, .height = 24 });
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "80x24") != null);
}

test "diagnostics dashboard renders all panes" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "metal",
        .gpu_accelerated = true,
        .gpu_linked = true,
        .plugin_count = 2,
        .plugin_names = &.{ "core", "wdbx" },
        .wdbx_blocks = 5,
        .wdbx_vectors = 10,
        .wdbx_entries = 3,
        .wdbx_spatial_records = 4,
        .scheduler_source = "test snapshot",
        .scheduler_running = 1,
        .scheduler_pending = 2,
        .scheduler_completed = 7,
        .scheduler_failed = 1,
        .memory_source = "MemoryTracker",
        .memory_peak = 4096,
        .memory_current = 2048,
        .memory_leaked = 0,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "Diagnostics Dashboard") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "metal") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "System") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Plugins") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "WDBX Storage") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spatial 3D") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Scheduler") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Memory") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Failed") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "test snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "MemoryTracker") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Peak") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Leaked") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Refresh") != null);
}

test "InteractiveTerminal struct layout" {
    const term = InteractiveTerminal{
        .fd = 0,
        .original = undefined,
        .is_tty = false,
    };
    try std.testing.expect(!term.is_tty);
    try std.testing.expectEqual(@as(std.posix.fd_t, 0), term.fd);
}

test "quit and refresh key detection" {
    try std.testing.expect(isQuitKey('q'));
    try std.testing.expect(isQuitKey('Q'));
    try std.testing.expect(isQuitKey(0x1b));
    try std.testing.expect(!isQuitKey('r'));
    try std.testing.expect(isRefreshKey('r'));
    try std.testing.expect(isRefreshKey('R'));
    try std.testing.expect(!isRefreshKey('q'));
}
