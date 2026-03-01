//! GPU Dashboard Command
//!
//! Interactive TUI dashboard combining GPU monitoring
//! and agent learning visualization in a split-panel layout.
//! Uses AsyncLoop for timer-driven refresh (10 FPS) independent of input.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");
const style_adapter = @import("../tui/style_adapter.zig");

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

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        initial_theme: *const tui.Theme,
    ) DashboardState {
        var theme_manager = tui.ThemeManager.init();
        theme_manager.current = initial_theme;
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
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;
    var parsed = try theme_options.parseThemeArgs(allocator, args);
    defer parsed.deinit();

    if (parsed.list_themes) {
        theme_options.printAvailableThemes();
        return;
    }

    if (parsed.wants_help) {
        printHelp();
        return;
    }

    if (parsed.remaining_args.len > 0) {
        utils.output.printError("Unknown argument for ui gpu: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        return error.InvalidArgument;
    }

    _ = io; // Not needed for TUI operations

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runDashboard(allocator, initial_theme);
}

fn runDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme) !void {
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

    var state = DashboardState.init(allocator, &terminal, initial_theme);
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
                        state.showNotification(theme_options.themeNotificationMessage(state.theme_manager.current.name));
                    },
                    'T' => {
                        state.theme_manager.prevTheme();
                        state.updateTheme();
                        state.showNotification(theme_options.themeNotificationMessage(state.theme_manager.current.name));
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
    const chrome = style_adapter.gpu(theme_val);
    const msg = "Resize terminal to at least 40x10";
    const row: u16 = if (height >= 2) (height - 1) / 2 else 0;
    const msg_len_u16: u16 = @intCast(msg.len);
    const col: u16 = if (width >= msg_len_u16 + 2) (width - msg_len_u16) / 2 else 0;
    try setCursorPosition(term, row, col);
    try term.write(chrome.warning);
    try term.write(msg);
    try term.write(theme_val.reset);
    try setCursorPosition(term, row + 1, 0);
    try term.write(theme_val.text_dim);
    try term.write("Current: ");
    var buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&buf, "{d}x{d}", .{ width, height }) catch "?x?";
    try term.write(size_str);
    try term.write("  [q] Quit");
    try term.write(theme_val.reset);
}

fn renderTitleBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const DashboardState, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    const inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    const title = " ABI GPU CONTROL PLANE ";
    const mode_label = if (state.paused) "PAUSED" else "LIVE";
    const theme_name = state.theme_manager.current.name;
    const theme_display = if (theme_name.len > 14) theme_name[0..14] else theme_name;

    const right_width = mode_label.len + theme_display.len + 8; // [mode] [theme]
    const left_max = inner -| right_width -| 1;
    const left_display = if (title.len > left_max) title[0..left_max] else title;
    const gap = inner -| left_display.len -| right_width;

    // Top border
    try term.write(chrome.frame);
    try term.write(box.tl);
    try writeRepeat(term, box.h, inner);
    try term.write(box.tr);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Title line with mode + theme chips
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);

    try term.write(theme_val.bold);
    try term.write(chrome.title);
    try term.write(left_display);
    try term.write(theme_val.reset);

    try writeRepeat(term, " ", gap);

    try term.write(chrome.chip_bg);
    try term.write(if (state.paused) chrome.paused else chrome.live);
    try term.write("[");
    try term.write(mode_label);
    try term.write("]");
    try term.write(theme_val.reset);
    try term.write(" ");

    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write("[");
    try term.write(theme_display);
    try term.write("]");
    try term.write(theme_val.reset);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Separator
    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(theme_val.reset);
    try term.write("\n");
}

fn renderNotification(term: *tui.Terminal, theme_val: *const tui.Theme, msg: []const u8, row: u16, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    try setCursorPosition(term, row, 0);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);

    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.info);
    try term.write(" INFO ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(theme_val.text);
    try term.write(msg);
    try term.write(theme_val.reset);

    const used = 10 + msg.len;
    if (used < @as(usize, width) - 1) {
        try writeRepeat(term, " ", @as(usize, width) - 1 - used);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);
}

fn renderStatusBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const DashboardState, row: u16, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    try setCursorPosition(term, row, 0);

    // Bottom separator
    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.rsep);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Status line with compact cyber chips
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);

    var buf: [64]u8 = undefined;
    const frame_str = std.fmt.bufPrint(&buf, "{d}", .{state.frame_count}) catch "?";
    const gpu_count = std.fmt.bufPrint(&buf, "{d}", .{state.gpu_monitor.devices.items.len}) catch "?";
    const ep_str = std.fmt.bufPrint(&buf, "{d}", .{state.agent_panel.episode_count}) catch "?";
    const eps_str = std.fmt.bufPrint(&buf, "{d:.2}", .{state.agent_panel.exploration_rate}) catch "?";

    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" frame ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(frame_str);

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" gpus ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(gpu_count);

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" episodes ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(ep_str);

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" epsilon ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(eps_str);

    const status_inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    const status_len = 48 + frame_str.len + gpu_count.len + ep_str.len + eps_str.len;
    if (status_len < status_inner) try writeRepeat(term, " ", status_inner - status_len);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);
}

fn renderHelpBar(term: *tui.Terminal, theme_val: *const tui.Theme, row: u16, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    try setCursorPosition(term, row, 0);

    const inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    try term.write(chrome.frame);
    try term.write(box.bl);
    try writeRepeat(term, box.h, inner);
    try term.write(box.br);
    try term.write(theme_val.reset);
    try term.write("\n");

    try term.write(" ");
    if (inner >= 56) {
        try writeKeyHint(term, chrome, theme_val, "q", "quit");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "p", "pause");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "t/T", "theme");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "r", "reset");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "h", "help");
    } else if (inner >= 30) {
        try writeKeyHint(term, chrome, theme_val, "q", "quit");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "t", "theme");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "h", "help");
    } else if (inner >= 14) {
        try writeKeyHint(term, chrome, theme_val, "q", "quit");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "h", "help");
    }
}

fn renderHelpOverlay(term: *tui.Terminal, theme_val: *const tui.Theme, width: u16, height: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    // Center the help box
    const box_width: u16 = @min(56, width - 4);
    const box_height: u16 = @min(@as(u16, 17), height - 2);
    const start_col = (width - box_width) / 2;
    const start_row = (height - box_height) / 2;

    // Draw help box
    try setCursorPosition(term, start_row, start_col);
    try term.write(chrome.frame);
    try term.write("╔");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╗");
    try term.write(theme_val.reset);

    const help_lines = [_][]const u8{
        "",
        "  ABI GPU Dashboard - Help",
        "",
        "  [q] / [Esc]   Quit dashboard",
        "  [p]           Pause/Resume updates",
        "  [t] / [T]     Cycle themes",
        "  [r]           Reset runtime stats",
        "  [h] / [?]     Toggle help overlay",
        "  --theme <name> Set initial theme at startup",
        "  --list-themes Print all supported themes",
        "",
        "  Left panel: GPU devices, memory, scheduler",
        "  Right panel: Agent phase, rewards, decisions",
        "",
    };

    const max_lines = @min(help_lines.len, @as(usize, box_height - 2));
    for (help_lines[0..max_lines], 0..) |line, i| {
        try setCursorPosition(term, start_row + 1 + @as(u16, @intCast(i)), start_col);
        try term.write(chrome.frame);
        try term.write("║");
        try term.write(theme_val.reset);
        try term.write(line);
        const pad = @as(usize, box_width) - 2 - line.len;
        try writeRepeat(term, " ", pad);
        try term.write(chrome.frame);
        try term.write("║");
        try term.write(theme_val.reset);
    }

    try setCursorPosition(term, start_row + box_height - 1, start_col);
    try term.write(chrome.frame);
    try term.write("╚");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╝");
    try term.write(theme_val.reset);

    // Footer hint
    try setCursorPosition(term, start_row + box_height, start_col + 4);
    try term.write(theme_val.text_dim);
    try term.write("Press q, h, Esc, or Enter to close");
    try term.write(theme_val.reset);
}

// ===============================================================================
// Utilities
// ===============================================================================

fn writeKeyHint(
    term: *tui.Terminal,
    chrome: style_adapter.ChromeStyle,
    theme_val: *const tui.Theme,
    key: []const u8,
    label: []const u8,
) !void {
    try term.write(chrome.keycap_bg);
    try term.write(chrome.keycap_fg);
    try term.write(" ");
    try term.write(key);
    try term.write(" ");
    try term.write(theme_val.reset);
    try term.write(theme_val.text_dim);
    try term.write(" ");
    try term.write(label);
    try term.write(theme_val.reset);
}

const writeRepeat = tui.render_utils.writeRepeat;

fn setCursorPosition(term: *tui.Terminal, row: u16, col: u16) !void {
    // Delegate to terminal.moveTo (already 0-indexed with +1 internally).
    try term.moveTo(row, col);
}

fn printFallbackInfo() !void {
    utils.output.println("\nGPU Backend Information:", .{});
    utils.output.println("  Use 'abi gpu backends' for backend details", .{});
    utils.output.println("  Use 'abi gpu devices' for device listing", .{});
    utils.output.println("\nAgent Information:", .{});
    utils.output.println("  Use 'abi agent' for agent interaction", .{});
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui gpu [OPTIONS]
        \\
        \\Launch an interactive GPU monitoring dashboard with real-time
        \\visualization of GPU status and agent learning progress.
        \\
        \\Options:
        \\  --theme <name>  Set initial theme (exact lowercase name)
        \\  --list-themes   Print available themes and exit
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

test {
    std.testing.refAllDecls(@This());
}
