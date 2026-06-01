const std = @import("std");

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

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "| {s:<28} |\n", .{state.title});
    try out.print(allocator, "+------------------------------+\n", .{});
    try out.print(allocator, "status: {s}\n", .{statusText(state.status)});
    for (state.items) |item| {
        try out.print(allocator, "- {s}: {s}\n", .{ item.label, item.value });
    }
    try out.print(allocator, "\nCommands: abi help | abi agent train all | abi agent os dry-run <cmd>\n", .{});

    return try out.toOwnedSlice(allocator);
}

/// Render the full interactive diagnostics dashboard.
pub fn renderDiagnostics(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    // Header
    try out.appendSlice(allocator, "\x1b[1;36m");
    try out.appendSlice(allocator, "╔══════════════════════════════════════════════════════════════╗\n");
    try out.appendSlice(allocator, "║              ABI Diagnostics Dashboard                      ║\n");
    try out.appendSlice(allocator, "╚══════════════════════════════════════════════════════════════╝\n");
    try out.appendSlice(allocator, "\x1b[0m");

    // System pane
    try out.appendSlice(allocator, "\x1b[1;33m┌─ System ────────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ GPU Backend:     \x1b[1m{s:<42}\x1b[0m│\n", .{ds.gpu_backend});
    try out.print(allocator, "│ Accelerated:     \x1b[1m{s:<42}\x1b[0m│\n", .{if (ds.gpu_accelerated) "yes" else "no"});
    try out.print(allocator, "│ Native Linked:   \x1b[1m{s:<42}\x1b[0m│\n", .{if (ds.gpu_linked) "yes" else "no"});
    try out.appendSlice(allocator, "\x1b[1;33m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Plugins pane
    try out.appendSlice(allocator, "\x1b[1;32m┌─ Plugins ───────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Registered:      \x1b[1m{d:<42}\x1b[0m│\n", .{ds.plugin_count});
    for (ds.plugin_names) |name| {
        try out.print(allocator, "│   - \x1b[1m{s:<55}\x1b[0m│\n", .{name});
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
    try out.print(allocator, "│ Source:          \x1b[1m{s:<42}\x1b[0m│\n", .{ds.scheduler_source});
    try out.print(allocator, "│ Running:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_running});
    try out.print(allocator, "│ Pending:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_pending});
    try out.print(allocator, "│ Completed:       \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_completed});
    try out.print(allocator, "│ Failed:          \x1b[1m{d:<42}\x1b[0m│\n", .{ds.scheduler_failed});
    try out.appendSlice(allocator, "\x1b[1;34m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Memory pane
    try out.appendSlice(allocator, "\x1b[1;31m┌─ Memory ────────────────────────────────────────────────────┐\x1b[0m\n");
    try out.print(allocator, "│ Source:          \x1b[1m{s:<42}\x1b[0m│\n", .{ds.memory_source});
    try out.print(allocator, "│ Peak:            \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_peak});
    try out.print(allocator, "│ Current:         \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_current});
    try out.print(allocator, "│ Leaked:          \x1b[1m{d:<42}\x1b[0m│\n", .{ds.memory_leaked});
    try out.appendSlice(allocator, "\x1b[1;31m└─────────────────────────────────────────────────────────────┘\x1b[0m\n");

    // Footer
    try out.appendSlice(allocator, "\n\x1b[2m[q/Esc] Quit  [r] Refresh\x1b[0m\n");

    return try out.toOwnedSlice(allocator);
}

// --- Interactive Terminal Helpers ---

extern fn isatty(fd: std.posix.fd_t) callconv(.c) c_int;

pub const InteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !InteractiveTerminal {
        const is_tty = isatty(fd) != 0;
        if (!is_tty) return error.NotATerminal;

        const original = try std.posix.tcgetattr(fd);
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

    pub fn deinit(self: *InteractiveTerminal) void {
        std.posix.tcsetattr(self.fd, .FLUSH, self.original) catch |err| {
            std.log.warn("failed to restore terminal: {s}", .{@errorName(err)});
        };
    }

    pub fn readKey(self: *InteractiveTerminal) ?u8 {
        var buf: [1]u8 = undefined;
        const n = std.posix.read(self.fd, &buf) catch |err| {
            std.log.warn("read stdin failed: {s}", .{@errorName(err)});
            return null;
        };
        if (n == 0) return null;
        return buf[0];
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
