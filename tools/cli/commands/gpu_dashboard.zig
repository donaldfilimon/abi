//! GPU Dashboard Command
//!
//! Interactive TUI dashboard combining GPU monitoring
//! and agent learning visualization in a split-panel layout.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const tui = @import("../tui/mod.zig");
const utils = @import("../utils/mod.zig");

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
    last_update_ms: i64,
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
            .last_update_ms = 0,
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
            if (elapsed > 2000) { // 2 second display
                self.notification = null;
            }
        }
    }

    /// Update theme reference in widgets after theme change
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

    var state = DashboardState.init(allocator, &terminal);
    defer state.deinit();

    // Main event loop
    while (true) {
        // Refresh terminal size
        state.term_size = terminal.size();
        state.clearExpiredNotification();

        // Update widgets if not paused
        if (!state.paused) {
            const now = abi.shared.utils.unixMs();
            // Update at ~10 FPS
            if (now - state.last_update_ms >= 100) {
                try state.gpu_monitor.update();
                try state.agent_panel.update();
                state.last_update_ms = now;
            }
        }

        // Clear and render
        try terminal.clear();
        try renderDashboard(&state);
        state.frame_count += 1;

        // Read input with short timeout for animation
        // Note: readEvent blocks, so we rely on update rate for animation
        const event = try terminal.readEvent();

        switch (event) {
            .key => |key| {
                if (try handleKeyEvent(&state, key)) break;
            },
            .mouse => |mouse| {
                // Handle mouse scroll for future enhancements
                _ = mouse;
            },
        }
    }
}

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

fn renderDashboard(state: *DashboardState) !void {
    const term = state.terminal;
    const theme = state.theme();
    const width = state.term_size.cols;
    const height = state.term_size.rows;

    // Render help overlay if active
    if (state.show_help) {
        try renderHelpOverlay(term, theme, width, height);
        return;
    }

    // Title bar
    try renderTitleBar(term, theme, state, width);

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
        try renderNotification(term, theme, msg, height - 3, width);
    }

    // Status bar
    try renderStatusBar(term, theme, state, height - 2, width);

    // Help bar
    try renderHelpBar(term, theme, height - 1, width);
}

fn renderTitleBar(term: *tui.Terminal, theme: *const tui.Theme, state: *const DashboardState, width: u16) !void {
    // Top border
    try term.write(theme.border);
    try term.write(box.tl);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.tr);
    try term.write(theme.reset);
    try term.write("\n");

    // Title line
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);

    const title = " ABI GPU Dashboard ";
    const status = if (state.paused) "[PAUSED]" else "[LIVE]";
    const theme_name = state.theme_manager.current.name;
    const frame_str_buf: [32]u8 = undefined;
    _ = frame_str_buf;

    // Center title
    const title_area = title.len + status.len + theme_name.len + 6;
    const left_pad = (@as(usize, width) - 2 - title_area) / 2;
    const right_pad = @as(usize, width) - 2 - title_area - left_pad;

    try writeRepeat(term, " ", left_pad);
    try term.write(theme.bold);
    try term.write(theme.primary);
    try term.write(title);
    try term.write(theme.reset);

    // Status indicator
    try term.write(" ");
    if (state.paused) {
        try term.write(theme.warning);
    } else {
        try term.write(theme.success);
    }
    try term.write(status);
    try term.write(theme.reset);

    // Theme indicator
    try term.write(" [");
    try term.write(theme.text_muted);
    try term.write(theme_name);
    try term.write(theme.reset);
    try term.write("]");

    try writeRepeat(term, " ", right_pad);
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");

    // Separator with center cross
    try term.write(theme.border);
    try term.write(box.lsep);
    const sep_half = (@as(usize, width) - 2) / 2;
    try writeRepeat(term, box.h, sep_half - 1);
    try term.write(box.tsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2 - sep_half);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderNotification(term: *tui.Terminal, theme: *const tui.Theme, msg: []const u8, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.info);
    try term.write("ℹ ");
    try term.write(msg);
    try term.write(theme.reset);

    const used = 4 + msg.len;
    if (used < @as(usize, width) - 1) {
        try writeRepeat(term, " ", @as(usize, width) - 1 - used);
    }

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
}

fn renderStatusBar(term: *tui.Terminal, theme: *const tui.Theme, state: *const DashboardState, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    // Bottom separator
    try term.write(theme.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");

    // Status line
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write(" ");

    // Frame counter
    try term.write(theme.text_dim);
    try term.write("Frame: ");
    try term.write(theme.reset);
    var buf: [32]u8 = undefined;
    const frame_str = std.fmt.bufPrint(&buf, "{d}", .{state.frame_count}) catch "?";
    try term.write(frame_str);

    // GPU count
    try term.write(theme.text_dim);
    try term.write(" │ GPUs: ");
    try term.write(theme.reset);
    const gpu_count = std.fmt.bufPrint(&buf, "{d}", .{state.gpu_monitor.devices.items.len}) catch "?";
    try term.write(gpu_count);

    // Agent episode
    try term.write(theme.text_dim);
    try term.write(" │ Episodes: ");
    try term.write(theme.reset);
    const ep_str = std.fmt.bufPrint(&buf, "{d}", .{state.agent_panel.episode_count}) catch "?";
    try term.write(ep_str);

    // Exploration rate
    try term.write(theme.text_dim);
    try term.write(" │ ε: ");
    try term.write(theme.reset);
    const eps_str = std.fmt.bufPrint(&buf, "{d:.2}", .{state.agent_panel.exploration_rate}) catch "?";
    try term.write(eps_str);

    // Pad to width
    const status_len = 8 + frame_str.len + 9 + gpu_count.len + 13 + ep_str.len + 6 + eps_str.len;
    if (status_len < @as(usize, width) - 2) {
        try writeRepeat(term, " ", @as(usize, width) - 2 - status_len);
    }

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
}

fn renderHelpBar(term: *tui.Terminal, theme: *const tui.Theme, row: u16, width: u16) !void {
    try setCursorPosition(term, row, 0);

    // Bottom border with help
    try term.write(theme.border);
    try term.write(box.bl);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.br);
    try term.write(theme.reset);
    try term.write("\n");

    // Help text
    try term.write(" ");
    try term.write(theme.accent);
    try term.write("q");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" Quit │ ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("p");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" Pause │ ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("t");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" Theme │ ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("r");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" Reset │ ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("h");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" Help");
    try term.write(theme.reset);
}

fn renderHelpOverlay(term: *tui.Terminal, theme: *const tui.Theme, width: u16, height: u16) !void {
    // Center the help box
    const box_width: u16 = @min(50, width - 4);
    const box_height: u16 = 15;
    const start_col = (width - box_width) / 2;
    const start_row = (height - box_height) / 2;

    // Draw help box
    try setCursorPosition(term, start_row, start_col);
    try term.write(theme.primary);
    try term.write("╔");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╗");
    try term.write(theme.reset);

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
        try term.write(theme.primary);
        try term.write("║");
        try term.write(theme.reset);
        try term.write(line);
        const pad = @as(usize, box_width) - 2 - line.len;
        try writeRepeat(term, " ", pad);
        try term.write(theme.primary);
        try term.write("║");
        try term.write(theme.reset);
    }

    try setCursorPosition(term, start_row + box_height - 1, start_col);
    try term.write(theme.primary);
    try term.write("╚");
    try writeRepeat(term, "═", @as(usize, box_width) - 2);
    try term.write("╝");
    try term.write(theme.reset);

    // Footer hint
    try setCursorPosition(term, start_row + box_height, start_col + 5);
    try term.write(theme.text_dim);
    try term.write("Press any key to close");
    try term.write(theme.reset);
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
