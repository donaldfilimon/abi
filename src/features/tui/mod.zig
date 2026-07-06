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
    /// 0-based index of focused pane for interactive navigation (System=0, Plugins=1, etc.)
    selected_pane: usize = 0,
};
// goal-turn-79df3a4a516d this-turn-edit

const DIAG_WIDTH: usize = 68;
const LABEL_WIDTH: usize = 25;
const VALUE_WIDTH: usize = 40;
const MAX_PLUGIN_ROWS: usize = 6;

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}

fn dashboardHealth(ds: DashboardState) []const u8 {
    if (ds.scheduler_failed > 0 or ds.memory_leaked > 0) return "attention";
    if (!ds.gpu_accelerated or !ds.gpu_linked) return "degraded";
    return "nominal";
}

fn boolText(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

fn appendRepeated(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, byte: u8, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.append(allocator, byte);
}

fn appendRule(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) try out.appendSlice(allocator, "─");
}

fn appendFitted(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, raw: []const u8, width: usize) !void {
    const safe = try sanitizeControlBytes(allocator, raw);
    defer allocator.free(safe);

    if (safe.len <= width) {
        try out.appendSlice(allocator, safe);
        try appendRepeated(out, allocator, ' ', width - safe.len);
        return;
    }

    if (width == 0) return;
    if (width == 1) {
        try out.append(allocator, '~');
        return;
    }

    try out.appendSlice(allocator, safe[0 .. width - 1]);
    try out.append(allocator, '~');
}

fn appendBorder(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, left: []const u8, title: []const u8, right: []const u8) !void {
    try out.appendSlice(allocator, left);
    if (title.len > 0) {
        try out.appendSlice(allocator, " ");
        try appendFitted(out, allocator, title, @min(title.len, DIAG_WIDTH - 4));
        try out.appendSlice(allocator, " ");
        const used = @min(title.len, DIAG_WIDTH - 4) + 2;
        if (used < DIAG_WIDTH) try appendRule(out, allocator, DIAG_WIDTH - used);
    } else {
        try appendRule(out, allocator, DIAG_WIDTH);
    }
    try out.appendSlice(allocator, right);
    try out.append(allocator, '\n');
}

fn appendRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: []const u8) !void {
    try out.appendSlice(allocator, "│ ");
    try appendFitted(out, allocator, label, LABEL_WIDTH);
    try out.appendSlice(allocator, " ");
    try appendFitted(out, allocator, value, VALUE_WIDTH);
    try out.appendSlice(allocator, " │\n");
}

fn appendMetricRow(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, label: []const u8, value: usize) !void {
    var buf: [32]u8 = undefined;
    const rendered = try std.fmt.bufPrint(&buf, "{d}", .{value});
    try appendRow(out, allocator, label, rendered);
}

fn appendPanelHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, title: []const u8) !void {
    try appendBorder(out, allocator, "┌", title, "┐");
}

fn appendPanelFooter(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator) !void {
    try appendBorder(out, allocator, "└", "", "┘");
}

fn appendPluginRows(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, plugin_names: []const []const u8) !void {
    const shown = @min(plugin_names.len, MAX_PLUGIN_ROWS);
    var i: usize = 0;
    while (i < shown) : (i += 1) {
        try appendRow(out, allocator, "plugin", plugin_names[i]);
    }
    if (plugin_names.len > shown) {
        var buf: [48]u8 = undefined;
        const more = try std.fmt.bufPrint(&buf, "+{d} more registered", .{plugin_names.len - shown});
        try appendRow(out, allocator, "plugin", more);
    }
}

fn appendDashboardHeader(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: DashboardState) !void {
    try out.appendSlice(allocator, "\x1b[1;36m");
    try appendBorder(out, allocator, "╔", "", "╗");
    try appendRow(out, allocator, "ABI Diagnostics Dashboard", "operational snapshot");
    try appendRow(out, allocator, "health", dashboardHealth(ds));
    try appendBorder(out, allocator, "╚", "", "╝");
    try out.appendSlice(allocator, "\x1b[0m");
}

test {
    std.testing.refAllDecls(@This());
}

/// Neutralize terminal control characters in `input` while preserving legitimate
/// UTF-8, so attacker-influenced strings interpolated into ANSI render output
/// cannot inject terminal escape sequences (ESC/CSI/OSC), embed NUL, or smuggle a
/// C1 control. The input is walked as UTF-8: each valid sequence is decoded and,
/// if the codepoint is a control (U+0000–U+001F, U+007F DEL, or the U+0080–U+009F
/// C1 range — which includes 0x9B CSI), every byte of that sequence is replaced
/// with a visible '.'; otherwise the sequence is copied verbatim so box-drawing
/// glyphs and accented text survive. Bytes that do not form a valid UTF-8 sequence
/// (a lone C1 like 0x9B, a stray continuation byte, or a truncated sequence) are
/// replaced one-for-one with '.'. The output length always equals the input
/// length. Behavior is defined for UTF-8 input; non-UTF-8 lone bytes are replaced
/// 1:1. Caller owns the returned slice.
pub fn sanitizeControlBytes(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    var i: usize = 0;
    while (i < input.len) {
        const seq_len: usize = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            // Invalid lead byte (lone continuation or lone C1 like 0x9B): drop one.
            out[i] = '.';
            i += 1;
            continue;
        };
        if (i + seq_len > input.len) {
            // Truncated multi-byte sequence at end of input: drop one byte.
            out[i] = '.';
            i += 1;
            continue;
        }
        const seq = input[i .. i + seq_len];
        const cp = std.unicode.utf8Decode(seq) catch {
            // Overlong/invalid continuation bytes: drop one byte and advance.
            out[i] = '.';
            i += 1;
            continue;
        };
        if (cp < 0x20 or cp == 0x7f or (cp >= 0x80 and cp <= 0x9f)) {
            // C0, DEL, or C1 control: neutralize every byte of the sequence so an
            // encoded C1 (e.g. 0xC2 0x9B) collapses to "..".
            @memset(out[i .. i + seq_len], '.');
        } else {
            @memcpy(out[i .. i + seq_len], seq);
        }
        i += seq_len;
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

    try appendDashboardHeader(&out, allocator, ds);

    const highlight = "\x1b[7m";
    const no_highlight = "\x1b[27m";

    // System pane (0)
    if (ds.selected_pane == 0) try out.appendSlice(allocator, highlight);
    try out.appendSlice(allocator, "\x1b[1;33m");
    try appendPanelHeader(&out, allocator, "System");
    try appendRow(&out, allocator, "GPU backend", ds.gpu_backend);
    try appendRow(&out, allocator, "accelerated", boolText(ds.gpu_accelerated));
    try appendRow(&out, allocator, "native linked", boolText(ds.gpu_linked));
    try appendPanelFooter(&out, allocator);
    try out.appendSlice(allocator, "\x1b[0m");
    if (ds.selected_pane == 0) try out.appendSlice(allocator, no_highlight);

    // Plugins pane (1)
    if (ds.selected_pane == 1) try out.appendSlice(allocator, highlight);
    try out.appendSlice(allocator, "\x1b[1;32m");
    try appendPanelHeader(&out, allocator, "Plugins");
    try appendMetricRow(&out, allocator, "Registered", ds.plugin_count);
    try appendPluginRows(&out, allocator, ds.plugin_names);
    try appendPanelFooter(&out, allocator);
    try out.appendSlice(allocator, "\x1b[0m");
    if (ds.selected_pane == 1) try out.appendSlice(allocator, no_highlight);

    // WDBX Storage pane (2)
    if (ds.selected_pane == 2) try out.appendSlice(allocator, highlight);
    try out.appendSlice(allocator, "\x1b[1;35m");
    try appendPanelHeader(&out, allocator, "WDBX Storage");
    try appendMetricRow(&out, allocator, "Block chain", ds.wdbx_blocks);
    try appendMetricRow(&out, allocator, "Vectors", ds.wdbx_vectors);
    try appendMetricRow(&out, allocator, "KV Entries", ds.wdbx_entries);
    try appendMetricRow(&out, allocator, "Spatial 3D", ds.wdbx_spatial_records);
    try appendPanelFooter(&out, allocator);
    try out.appendSlice(allocator, "\x1b[0m");
    if (ds.selected_pane == 2) try out.appendSlice(allocator, no_highlight);

    // Scheduler pane (3)
    if (ds.selected_pane == 3) try out.appendSlice(allocator, highlight);
    try out.appendSlice(allocator, "\x1b[1;34m");
    try appendPanelHeader(&out, allocator, "Scheduler");
    try appendRow(&out, allocator, "source", ds.scheduler_source);
    try appendMetricRow(&out, allocator, "Running", ds.scheduler_running);
    try appendMetricRow(&out, allocator, "Pending", ds.scheduler_pending);
    try appendMetricRow(&out, allocator, "Completed", ds.scheduler_completed);
    try appendMetricRow(&out, allocator, "Failed", ds.scheduler_failed);
    try appendPanelFooter(&out, allocator);
    try out.appendSlice(allocator, "\x1b[0m");
    if (ds.selected_pane == 3) try out.appendSlice(allocator, no_highlight);

    // Memory pane (4)
    if (ds.selected_pane == 4) try out.appendSlice(allocator, highlight);
    try out.appendSlice(allocator, "\x1b[1;31m");
    try appendPanelHeader(&out, allocator, "Memory");
    try appendRow(&out, allocator, "source", ds.memory_source);
    try appendMetricRow(&out, allocator, "Peak bytes", ds.memory_peak);
    try appendMetricRow(&out, allocator, "Current bytes", ds.memory_current);
    try appendMetricRow(&out, allocator, "Leaked bytes", ds.memory_leaked);
    try appendPanelFooter(&out, allocator);
    try out.appendSlice(allocator, "\x1b[0m");
    if (ds.selected_pane == 4) try out.appendSlice(allocator, no_highlight);

    try out.appendSlice(allocator, "\n\x1b[2m[q/Esc] Quit  [r] Refresh  [1-5/h/l] Select pane  live snapshot every 1s\x1b[0m\n");

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

/// POSIX (macOS/Linux) raw-mode terminal. libc-free: use std.posix TTY probing
/// before termios so non-terminals fall back cleanly without noisy diagnostics.
const PosixInteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !PosixInteractiveTerminal {
        if (@hasDecl(std.posix.system, "isatty") and std.posix.system.isatty(fd) == 0) return error.NotATerminal;

        // tcgetattr can still fail if stdin stops being a terminal between the
        // probe and termios setup; treat that as a normal fallback signal.
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

test "tui sanitizeControlBytes neutralizes C1 controls and preserves UTF-8" {
    const allocator = std.testing.allocator;

    // Lone 0x9B (raw CSI on a non-UTF-8 terminal): not a valid UTF-8 start byte,
    // replaced 1:1 with '.'. No 0x9B survives; length preserved.
    {
        const clean = try sanitizeControlBytes(allocator, "x\x9by");
        defer allocator.free(clean);
        try std.testing.expectEqualStrings("x.y", clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x9b) == null);
        try std.testing.expectEqual(@as(usize, 3), clean.len);
    }

    // Encoded C1 "\xc2\x9b" decodes to U+009B (CSI): both bytes collapse to "..".
    // Neither the 0x9b payload nor the 0xc2 lead byte survives; length preserved.
    {
        const clean = try sanitizeControlBytes(allocator, "a\xc2\x9bb");
        defer allocator.free(clean);
        try std.testing.expectEqualStrings("a..b", clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x9b) == null);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0xc2) == null);
        try std.testing.expectEqual(@as(usize, 4), clean.len);
    }

    // Valid multibyte UTF-8 (accented "aéb" and a box-drawing "─") is preserved
    // byte-for-byte.
    {
        const accented = "a\xc3\xa9b"; // a é b
        const clean = try sanitizeControlBytes(allocator, accented);
        defer allocator.free(clean);
        try std.testing.expectEqualStrings(accented, clean);

        const box = "\xe2\x94\x80"; // ─ (U+2500)
        const clean_box = try sanitizeControlBytes(allocator, box);
        defer allocator.free(clean_box);
        try std.testing.expectEqualStrings(box, clean_box);
    }

    // ESC (0x1b) and NUL (0x00) are still stripped; output length == input length.
    {
        const dirty = "\x1b[2J\x00ok";
        const clean = try sanitizeControlBytes(allocator, dirty);
        defer allocator.free(clean);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x1b) == null);
        try std.testing.expect(std.mem.indexOfScalar(u8, clean, 0x00) == null);
        try std.testing.expectEqual(dirty.len, clean.len);
    }
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
    try std.testing.expect(std.mem.indexOf(u8, rendered, "operational snapshot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "health") != null);
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

test "diagnostics dashboard summarizes attention state and bounds plugin rows" {
    const rendered = try renderDiagnostics(std.testing.allocator, .{
        .gpu_backend = "cpu",
        .gpu_accelerated = false,
        .gpu_linked = false,
        .plugin_count = 8,
        .plugin_names = &.{ "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7" },
        .scheduler_failed = 1,
    });
    defer std.testing.allocator.free(rendered);

    try std.testing.expect(std.mem.indexOf(u8, rendered, "attention") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "+2 more registered") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "p7") == null);
}

test "InteractiveTerminal struct layout" {
    const term = if (builtin.os.tag == .windows) InteractiveTerminal{
        .fd = 0,
        .is_tty = false,
    } else InteractiveTerminal{
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
