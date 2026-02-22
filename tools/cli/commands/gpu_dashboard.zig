//! GPU Dashboard Command
//!
//! Interactive TUI dashboard combining GPU monitoring
//! and agent learning visualization in a split-panel layout.
//! Uses AsyncLoop for timer-driven refresh (10 FPS) independent of input.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const tui = @import("../tui/mod.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "gpu-dashboard",
    .description = "Interactive GPU + Agent monitoring dashboard",
    .aliases = &.{"dashboard"},
    .io_mode = .io,
    .forward = .{
        .target = "ui",
        .prepend_args = &[_][:0]const u8{"gpu"},
        .warning = "'abi gpu-dashboard' is deprecated; use 'abi ui gpu'.",
    },
};

// ===============================================================================
// Types
// ===============================================================================

/// Dashboard state combining GPU monitor and agent panel
const DashboardState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme_manager: tui.ThemeManager,
    gpu_monitor: tui.GpuMonitor,
    agent_panel: tui.AgentPanel,
    term_size: tui.TerminalSize,
    paused: bool,
    show_help: bool,
    frame_count: u64,
    notification: ?[]const u8,
    notification_time: i64,

    pub fn init(allocator: std.mem.Allocator, terminal: *tui.Terminal) DashboardState {
        const theme_manager = tui.ThemeManager.init();
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme_manager = theme_manager,
            .gpu_monitor = tui.GpuMonitor.init(allocator, terminal, theme_manager.current),
            .agent_panel = tui.AgentPanel.init(allocator, terminal, theme_manager.current),
            .term_size = terminal.size(),
            .paused = false,
            .show_help = false,
            .frame_count = 0,
            .notification = null,
            .notification_time = 0,
        };
    }

    pub fn deinit(self: *DashboardState) void {
        self.gpu_monitor.deinit();
        self.agent_panel.deinit();
    }

    pub fn theme(self: *const DashboardState) *const tui.Theme {
        return self.theme_manager.current;
    }

    pub fn showNotification(self: *DashboardState, msg: []const u8) void {
        self.notification = msg;
        self.notification_time = abi.shared.utils.unixMs();
    }

    pub fn clearExpiredNotification(self: *DashboardState) void {
        if (self.notification != null) {
            const elapsed = abi.shared.utils.unixMs() - self.notification_time;
            if (elapsed > 2000) {
                self.notification = null;
            }
        }
    }

    pub fn updateTheme(self: *DashboardState) void {
        self.gpu_monitor.theme = self.theme_manager.current;
        self.agent_panel.theme = self.theme_manager.current;
    }
};

// ===============================================================================
// Box Drawing Characters
// ===============================================================================

const box = struct {
    const tl = "╭";
    const tr = "╮";
    const bl = "╰";
    const br = "╯";
    const h = "─";
    const v = "│";
    const lsep = "├";
    const rsep = "┤";
    const cross = "┼";
    const tsep = "┬";
    const bsep = "┴";
};

// ===============================================================================
// Entry Point
// ===============================================================================

/// Entry point for the GPU dashboard command.
pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    _ = io; // Not needed for TUI operations

    try runDashboard(allocator);
}

fn runDashboard(allocator: std.mem.Allocator) !void {
    // Check platform support
    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("GPU Dashboard requires a terminal. Platform: {s}", .{caps.platform_name});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    // Enter TUI mode
    terminal.enter() catch |err| {
        utils.output.printError("Failed to start GPU Dashboard: {t}", .{err});
        utils.output.printInfo("Falling back to non-interactive GPU info.", .{});
        try printFallbackInfo();
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI GPU Dashboard") catch {};

    var state = DashboardState.init(allocator, &terminal);
    defer state.deinit();

    // Use AsyncLoop for timer-driven refresh instead of blocking readEvent.
    // This gives real 10 FPS auto-refresh regardless of user input.
    var loop = tui.AsyncLoop.init(allocator, &terminal, .{
        .refresh_rate_ms = 100, // 10 FPS
        .input_poll_ms = 16, // ~60 Hz input polling
        .auto_resize = true,
    });
    defer loop.deinit();

    loop.setRenderCallback(&dashboardRender);
    loop.setTickCallback(&dashboardTick);
    loop.setUpdateCallback(&dashboardUpdate);
    loop.setUserData(@ptrCast(&state));

    try loop.run();
}

// ===============================================================================
// AsyncLoop Callbacks
// ===============================================================================

/// Render callback — clears screen and draws the full dashboard.
fn dashboardRender(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    try state.terminal.clear();
    try renderDashboard(state);
    state.frame_count = loop.getFrameCount();
}

/// Tick callback — updates widget data and handles timed state.
/// Called at refresh_rate_ms (100ms) intervals.
fn dashboardTick(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    state.term_size = state.terminal.size();
    state.clearExpiredNotification();

    if (!state.paused) {
        try state.gpu_monitor.update();
        try state.agent_panel.update();
    }
}

/// Update callback — handles input events. Returns true to quit.
fn dashboardUpdate(loop: *tui.AsyncLoop, event: tui.AsyncEvent) anyerror!bool {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    return switch (event) {
        .input => |ev| switch (ev) {
            .key => |key| try handleKeyEvent(state, key),
            .mouse => false,
        },
        .resize => |size| blk: {
            state.term_size = .{
                .rows = size.rows,
                .cols = size.cols,
            };
            break :blk false;
        },
        .quit => true,
        else => false,
    };
}

// ===============================================================================
// Input Handling
// ===============================================================================

fn handleKeyEvent(state: *DashboardState, key: tui.Key) !bool {
    // Help overlay handles its own keys
    if (state.show_help) {
        switch (key.code) {
            .escape, .enter => state.show_help = false,
            .character => {
                if (key.char) |ch| {
                    if (ch == 'h' or ch == 'q') state.show_help = false;
                }
            },
            else => {},
        }
        return false;
    }

    switch (key.code) {
        .ctrl_c => return true,
        .escape => return true,
        .character => {
            if (key.char) |ch| {
                switch (ch) {
                    'q' => return true,
                    'p' => {
                        state.paused = !state.paused;
                        state.showNotification(if (state.paused) "Paused" else "Resumed");
                    },
                    't' => {
                        state.theme_manager.nextTheme();
                        state.updateTheme();
                        state.showNotification("Theme changed");
                    },
                    'T' => {
                        state.theme_manager.prevTheme();
                        state.updateTheme();
                        state.showNotification("Theme changed");
                    },
                    'h', '?' => {
                        state.show_help = true;
                    },
                    'r' => {
                        // Reset statistics
                        state.gpu_monitor.clearDevices();
                        state.gpu_monitor.update_counter = 0;
                        state.agent_panel.episode_count = 0;
                        state.agent_panel.total_reward = 0;
                        state.agent_panel.exploration_rate = 1.0;
                        state.agent_panel.phase = .exploration;
                        state.agent_panel.update_counter = 0;
                        state.showNotification("Stats reset");
                    },
                    else => {},
                }
            }
        },
        else => {},
    }
    return false;
}

// ===============================================================================
// Rendering
// ===============================================================================

const MIN_COLS: u16 = 40;
const MIN_ROWS: u16 = 10;

fn renderDashboard(state: *DashboardState) !void {
    const term = state.terminal;
    const theme_val = state.theme();
    const width = state.term_size.cols;
    const height = state.term_size.rows;

    // Require minimum terminal size to avoid layout overflow
    if (width < MIN_COLS or height < MIN_ROWS) {
        try renderResizeMessage(term, theme_val, width, height);
        return;
    }

    // Render help overlay if active
    if (state.show_help) {
        try renderHelpOverlay(term, theme_val, width, height);
        return;
    }

    // Title bar
    try renderTitleBar(term, theme_val, state, width);

    // Calculate panel dimensions
    const panel_start_row: u16 = 3;
    const panel_height: u16 = @max(8, height -| 6);
    const half_width = width / 2;

    // GPU Monitor (left panel)
    try state.gpu_monitor.render(panel_start_row, 0, half_width, panel_height);

    // Agent Panel (right panel)
    try state.agent_panel.render(panel_start_row, half_width, half_width, panel_height);

    // Notification (if any)
    if (state.notification) |msg| {
        try renderNotification(term, theme_val, msg, height - 3, width);
    }

    // Status bar
    try renderStatusBar(term, theme_val, state, height - 2, width);

    // Help bar
    try renderHelpBar(term, theme_val, height - 1, width);
}

fn renderResizeMessage(term: *tui.Terminal, theme_val: *const tui.Theme, width: u16, height: u16) !void {
    const msg = "Resize terminal to at least 40×10";
    const row: u16 = if (height >= 2) (height - 1) / 2 else 0;
    const msg_len_u16: u16 = @intCast(msg.len);
    const col: u16 = if (width >= msg_len_u16 + 2) (width - msg_len_u16) / 2 else 0;
    try setCursorPosition(term, row, col);
    try term.write(theme_val.warning);
    try term.write(msg);
    try term.write(theme_val.reset);
    try setCursorPosition(term, row + 1, 0);
    try term.write(theme_val.text_dim);
    try term.write("Current: ");
    var buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&buf, "{d}×{d}", .{ width, height }) catch "?×?";
    try term.write(size_str);
    try term.write("  [q] Quit");
    try term.write(theme_val.reset);
}

fn renderTitleBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const DashboardState, width: u16) !void {
    // Top border
    try term.write(theme_val.border);
    try term.write(box.tl);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.tr);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Title line
    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);

    const title = " ABI GPU Dashboard ";
    const status = if (state.paused) "[PAUSED]" else "[LIVE]";
    const theme_name = state.theme_manager.current.name;
    const inner = if (width >= 2) width - 2 else 0;

    // Build center content; truncate theme name if needed so we don't overflow
    const max_theme_len = if (inner > title.len + status.len + 8)
        @min(theme_name.len, inner - title.len - status.len - 8)
    else
        0;
    const theme_display = if (max_theme_len > 0) theme_name[0..max_theme_len] else "";
    const title_area = title.len + status.len + (if (max_theme_len > 0) theme_display.len + 4 else 0);
    const left_pad = if (inner >= title_area) (inner - title_area) / 2 else 0;
    const right_pad = if (inner >= title_area) inner - title_area - left_pad else 0;

    try writeRepeat(term, " ", left_pad);
    try term.write(theme_val.bold);
    try term.write(theme_val.primary);
    try term.write(title);
    try term.write(theme_val.reset);

    try term.write(" ");
    if (state.paused) {
        try term.write(theme_val.warning);
    } else {
        try term.write(theme_val.success);
    }
    try term.write(status);
    try term.write(theme_val.reset);

    if (max_theme_len > 0) {
        try term.write(" [");
        try term.write(theme_val.text_muted);
        try term.write(theme_display);
        try term.write(theme_val.reset);
        try term.write("]");
    }

    try writeRepeat(term, " ", right_pad);
    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Separator with center cross
    try term.write(theme_val.border);
    try term.write(box.lsep);
    const sep_half = (@as(usize, width) - 2) / 2;
    try writeRepeat(term, box.h, sep_half - 1);
    try term.write(box.tsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2 - sep_half);
    try term.write(box.rsep);
    try term.write(theme_val.reset);
    try term.write("\n");
}

fn renderNotification(term: *tui.Terminal, theme_val: *const tui.Theme, msg: []const u8, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(theme_val.info);
    try term.write("ℹ ");
    try term.write(msg);
    try term.write(theme_val.reset);

    const used = 4 + msg.len;
    if (used < @as(usize, width) - 1) {
        try writeRepeat(term, " ", @as(usize, width) - 1 - used);
    }

    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);
}

fn renderStatusBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const DashboardState, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    // Bottom separator
    try term.write(theme_val.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.rsep);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Status line
    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);
    try term.write(" ");

    // Frame counter
    try term.write(theme_val.text_dim);
    try term.write("Frame: ");
    try term.write(theme_val.reset);
    var buf: [32]u8 = undefined;
    const frame_str = std.fmt.bufPrint(&buf, "{d}", .{state.frame_count}) catch "?";
    try term.write(frame_str);

    // GPU count
    try term.write(theme_val.text_dim);
    try term.write(" │ GPUs: ");
    try term.write(theme_val.reset);
    const gpu_count = std.fmt.bufPrint(&buf, "{d}", .{state.gpu_monitor.devices.items.len}) catch "?";
    try term.write(gpu_count);

    // Agent episode
    try term.write(theme_val.text_dim);
    try term.write(" │ Episodes: ");
    try term.write(theme_val.reset);
    const ep_str = std.fmt.bufPrint(&buf, "{d}", .{state.agent_panel.episode_count}) catch "?";
    try term.write(ep_str);

    // Exploration rate
    try term.write(theme_val.text_dim);
    try term.write(" │ ε: ");
    try term.write(theme_val.reset);
    const eps_str = std.fmt.bufPrint(&buf, "{d:.2}", .{state.agent_panel.exploration_rate}) catch "?";
    try term.write(eps_str);

    // Pad to width (safe for narrow terminals)
    const status_inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    const status_len = 8 + frame_str.len + 9 + gpu_count.len + 13 + ep_str.len + 6 + eps_str.len;
    if (status_len < status_inner) {
        try writeRepeat(term, " ", status_inner - status_len);
    }

    try term.write(theme_val.border);
    try term.write(box.v);
    try term.write(theme_val.reset);
}

fn renderHelpBar(term: *tui.Terminal, theme_val: *const tui.Theme, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    const inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    try term.write(theme_val.border);
    try term.write(box.bl);
    try writeRepeat(term, box.h, inner);
    try term.write(box.br);
    try term.write(theme_val.reset);
    try term.write("\n");

    const full_help_len = 45;
    if (inner >= full_help_len) {
        try term.write(" ");
        try term.write(theme_val.accent);
        try term.write("q");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Quit ");
        try term.write(box.v);
        try term.write(" ");
        try term.write(theme_val.reset);
        try term.write(theme_val.accent);
        try term.write("p");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Pause ");
        try term.write(box.v);
        try term.write(" ");
        try term.write(theme_val.reset);
        try term.write(theme_val.accent);
        try term.write("t");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Theme ");
        try term.write(box.v);
        try term.write(" ");
        try term.write(theme_val.reset);
        try term.write(theme_val.accent);
        try term.write("r");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Reset ");
        try term.write(box.v);
        try term.write(" ");
        try term.write(theme_val.reset);
        try term.write(theme_val.accent);
        try term.write("h");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Help");
        try term.write(theme_val.reset);
    } else if (inner >= 28) {
        try term.write(" ");
        try term.write(theme_val.accent);
        try term.write("q");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Quit ");
        try term.write(theme_val.reset);
        try term.write(theme_val.accent);
        try term.write("p");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Pause ");
        try term.write(theme_val.accent);
        try term.write("t");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Theme ");
        try term.write(theme_val.accent);
        try term.write("h");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Help");
        try term.write(theme_val.reset);
    } else if (inner > 8) {
        try term.write(" ");
        try term.write(theme_val.accent);
        try term.write("q");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Quit ");
        try term.write(theme_val.accent);
        try term.write("h");
        try term.write(theme_val.reset);
        try term.write(theme_val.text_dim);
        try term.write(" Help");
        try term.write(theme_val.reset);
    }
}

fn renderHelpOverlay(term: *tui.Terminal, theme_val: *const tui.Theme, width: u16, height: u16) !void {
    // Center the help box
    const box_width: u16 = @min(50, width - 4);
    const box_height: u16 = 15;
    const start_col = (width - box_width) / 2;
    const start_row = (height - box_height) / 2;

    // Draw help box
    try setCursorPosition(term, start_row, start_col);
    try term.write(theme_val.primary);
    try term.write("╔");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╗");
    try term.write(theme_val.reset);

    const help_lines = [_][]const u8{
        "",
        "  GPU Dashboard Help",
        "",
        "  [q] / [Esc]   Quit dashboard",
        "  [p]           Pause/Resume updates",
        "  [t] / [T]     Cycle theme forward/back",
        "  [r]           Reset statistics",
        "  [h] / [?]     Toggle this help",
        "",
        "  GPU Monitor shows backend status,",
        "  memory usage, and scheduler stats.",
        "",
        "  Agent Panel shows learning progress",
        "  and recent scheduling decisions.",
        "",
    };

    for (help_lines, 0..) |line, i| {
        try setCursorPosition(term, start_row + 1 + @as(u16, @intCast(i)), start_col);
        try term.write(theme_val.primary);
        try term.write("║");
        try term.write(theme_val.reset);
        try term.write(line);
        const pad = @as(usize, box_width) - 2 - line.len;
        try writeRepeat(term, " ", pad);
        try term.write(theme_val.primary);
        try term.write("║");
        try term.write(theme_val.reset);
    }

    try setCursorPosition(term, start_row + box_height - 1, start_col);
    try term.write(theme_val.primary);
    try term.write("╚");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╝");
    try term.write(theme_val.reset);

    // Footer hint
    try setCursorPosition(term, start_row + box_height, start_col + 5);
    try term.write(theme_val.text_dim);
    try term.write("Press any key to close");
    try term.write(theme_val.reset);
}

// ===============================================================================
// Utilities
// ===============================================================================

fn writeRepeat(term: *tui.Terminal, char: []const u8, count: usize) !void {
    for (0..count) |_| {
        try term.write(char);
    }
}

fn setCursorPosition(term: *tui.Terminal, row: u16, col: u16) !void {
    var buf: [16]u8 = undefined;
    const seq = std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{ row + 1, col + 1 }) catch return;
    try term.write(seq);
}

fn printFallbackInfo() !void {
    std.debug.print("\nGPU Backend Information:\n", .{});
    std.debug.print("  Use 'abi gpu backends' for backend details\n", .{});
    std.debug.print("  Use 'abi gpu devices' for device listing\n", .{});
    std.debug.print("\nAgent Information:\n", .{});
    std.debug.print("  Use 'abi agent' for agent interaction\n", .{});
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi gpu-dashboard [OPTIONS]
        \\
        \\Launch an interactive GPU monitoring dashboard with real-time
        \\visualization of GPU status and agent learning progress.
        \\
        \\Options:
        \\  -h, --help      Show this help message
        \\
        \\Keyboard Controls:
        \\  q, Esc          Quit the dashboard
        \\  p               Pause/Resume updates
        \\  t / T           Cycle through themes
        \\  r               Reset statistics
        \\  h, ?            Show help overlay
        \\
        \\The dashboard displays:
        \\  - Left panel: GPU devices, memory usage, scheduler stats
        \\  - Right panel: Agent learning phase, reward history, decisions
        \\
    , .{});
}
