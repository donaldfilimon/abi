//! Unified Dashboard Command
//!
//! Opens a tabbed TUI dashboard combining all monitoring panels
//! in a single interface. Individual panels remain available
//! via `abi ui gpu`, `abi ui brain`, etc.
//! Uses AsyncLoop for timer-driven refresh (10 FPS).

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");
const render_utils = tui.render_utils;

// ===============================================================================
// Constants
// ===============================================================================

const panel_count = 12;

const help_title = "Dashboard Keyboard Shortcuts";

const help_lines = [_][]const u8{
    "  [1-9,0]  Switch to tab",
    "  [Tab]    Next tab",
    "  [S-Tab]  Previous tab",
    "  [t/T]    Next/Previous theme",
    "  [p]      Pause/Resume",
    "  [?/h]    Toggle this help",
    "  [q/Esc]  Quit",
};

// ===============================================================================
// Types
// ===============================================================================

const ErrorBoundaryPanel = tui.Panel.ErrorBoundaryPanel;

const DashboardState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme_manager: tui.ThemeManager,
    tab: tui.TabBar,
    paused: bool,
    term_size: tui.TerminalSize,
    frame_count: u64,
    panels: [panel_count]tui.Panel,
    help: tui.HelpOverlay,

    /// Error boundary wrappers for each panel slot. Panels are wrapped
    /// so that a single panel crashing does not kill the entire dashboard.
    boundaries: [panel_count]ErrorBoundaryPanel,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal_ptr: *tui.Terminal,
        initial_theme: *const tui.Theme,
        tab_labels: []const []const u8,
    ) DashboardState {
        var theme_manager = tui.ThemeManager.init();
        theme_manager.current = initial_theme;

        // Initialize all boundary slots wrapping noop panels.
        // Actual adapter wiring uses wrapPanel() after init.
        var boundaries: [panel_count]ErrorBoundaryPanel = undefined;
        var panels: [panel_count]tui.Panel = undefined;
        for (&boundaries, &panels) |*b, *p| {
            b.* = tui.Panel.withErrorBoundary(tui.Panel.noop_panel);
            p.* = tui.Panel.noop_panel; // placeholder; overwritten below
        }

        var state: DashboardState = .{
            .allocator = allocator,
            .terminal = terminal_ptr,
            .theme_manager = theme_manager,
            .tab = tui.TabBar.init(tab_labels),
            .paused = false,
            .term_size = terminal_ptr.size(),
            .frame_count = 0,
            .panels = panels,
            .help = tui.HelpOverlay.init(help_title, &help_lines),
            .boundaries = boundaries,
        };

        // Wire panel slots to point at the boundary wrappers.
        // This must happen after the struct is at its final memory location
        // in the caller's stack frame, so we do it via a separate method.
        // (See wrapAllPanels below.)
        _ = &state;

        return state;
    }

    /// Finalize error boundary wiring after the struct has stable memory.
    /// Must be called once after init, before the event loop starts.
    pub fn wrapAllPanels(self: *DashboardState) void {
        for (&self.boundaries, &self.panels) |*b, *p| {
            p.* = b.asPanel();
        }
    }

    /// Set a panel slot, wrapping it in an error boundary automatically.
    /// The inner panel is stored in the boundary array for stable memory.
    pub fn setPanel(self: *DashboardState, index: usize, inner: tui.Panel) void {
        self.boundaries[index] = tui.Panel.withErrorBoundary(inner);
        self.panels[index] = self.boundaries[index].asPanel();
    }

    pub fn theme(self: *const DashboardState) *const tui.Theme {
        return self.theme_manager.current;
    }

    /// Get the active panel.
    pub fn activePanel(self: *DashboardState) tui.Panel {
        return self.panels[self.tab.active];
    }
};

// ===============================================================================
// Entry Point
// ===============================================================================

/// Entry point for the unified dashboard command.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
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

    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("Dashboard requires a terminal. Platform: {s}", .{caps.platform_name});
        return;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;

    // Tab labels for the unified view
    const tab_labels = [_][]const u8{
        "GPU", "Agent", "Train", "Model", "Stream", "DB", "Net", "Bench", "Brain", "Security", "Connectors", "Ralph",
    };

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        utils.output.printError("Failed to start dashboard: {t}", .{err});
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Dashboard") catch {};

    var state = DashboardState.init(allocator, &terminal, initial_theme, &tab_labels);

    // Finalize error boundary wiring now that state has stable stack memory.
    // Each panel slot is wrapped in an ErrorBoundaryPanel so that a single
    // panel crashing renders a fallback error message instead of killing
    // the entire dashboard.
    state.wrapAllPanels();

    // Use AsyncLoop for timer-driven refresh
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

/// Render callback -- clears screen and draws the full dashboard.
fn dashboardRender(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    try state.terminal.clear();

    const sz = state.term_size;
    const th = state.theme();

    // Tab bar (row 0)
    try state.tab.render(state.terminal, th, 0, sz.cols);

    // Status bar (row 1)
    try renderStatusBar(state, th, sz);

    // Content area: starts at row 2, ends at rows-2 (leaving room for help bar)
    const content_rect = tui.Rect{
        .x = 0,
        .y = 2,
        .width = sz.cols,
        .height = sz.rows -| 4, // 2 rows top (tab + status) + 2 rows bottom margin
    };

    if (!content_rect.isEmpty()) {
        try state.panels[state.tab.active].render(state.terminal, content_rect, th);
    }

    // Help bar (bottom row)
    try state.terminal.moveTo(sz.rows -| 1, 1);
    try state.terminal.write(th.text_dim);
    try state.terminal.write("[1-9,0] tabs  [Tab/S-Tab] next/prev  [t] theme  [p] pause  [?] help  [q] quit");
    try state.terminal.write(th.reset);

    // Help overlay rendered last (on top of everything) if visible
    try state.help.render(state.terminal, th, sz.cols, sz.rows);

    try state.terminal.flush();
    state.frame_count = loop.getFrameCount();
}

/// Tick callback -- updates state on each timer interval.
fn dashboardTick(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    state.term_size = state.terminal.size();

    // Tick the active panel unless paused
    if (!state.paused) {
        try state.panels[state.tab.active].tick();
    }
}

/// Update callback -- handles input events. Returns true to quit.
fn dashboardUpdate(loop: *tui.AsyncLoop, event: tui.AsyncEvent) anyerror!bool {
    const state = loop.getUserData(DashboardState) orelse
        return error.UserDataNotSet;
    return switch (event) {
        .input => |ev| switch (ev) {
            .key => |key| handleKeyEvent(state, key),
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
// Status Bar
// ===============================================================================

/// Render the status bar at row 1 showing panel info, frame count, and theme.
fn renderStatusBar(state: *const DashboardState, th: *const tui.Theme, sz: tui.TerminalSize) !void {
    try state.terminal.moveTo(1, 0);

    // Background fill for status bar
    try state.terminal.write(th.text_dim);

    // Left: panel name + shortcut hint
    const panel_name = state.panels[state.tab.active].getName();
    const hint = state.panels[state.tab.active].shortcutHint();

    try state.terminal.write(" ");
    try state.terminal.write(th.accent);
    try state.terminal.write(panel_name);
    try state.terminal.write(th.reset);

    if (hint.len > 0) {
        try state.terminal.write(th.text_muted);
        try state.terminal.write(" (");
        try state.terminal.write(hint);
        try state.terminal.write(")");
        try state.terminal.write(th.reset);
    }

    // Center: frame count + paused indicator
    // Position roughly at center of terminal
    const center_col = sz.cols / 2 -| 10;
    try state.terminal.moveTo(1, center_col);
    try state.terminal.write(th.text_dim);

    var frame_buf: [32]u8 = undefined;
    const frame_str = std.fmt.bufPrint(&frame_buf, "Frame: {d}", .{state.frame_count}) catch "Frame: ?";
    try state.terminal.write(frame_str);

    if (state.paused) {
        try state.terminal.write("  ");
        try state.terminal.write(th.warning);
        try state.terminal.write("PAUSED");
        try state.terminal.write(th.reset);
    }

    try state.terminal.write(th.reset);

    // Right: theme name
    const theme_name = state.theme_manager.current.name;
    const right_col = sz.cols -| @as(u16, @intCast(@min(theme_name.len + 3, @as(usize, sz.cols))));
    try state.terminal.moveTo(1, right_col);
    try state.terminal.write(th.text_muted);
    try state.terminal.write("[");
    try state.terminal.write(theme_name);
    try state.terminal.write("]");
    try state.terminal.write(th.reset);
}

// ===============================================================================
// Input Handling
// ===============================================================================

fn handleKeyEvent(state: *DashboardState, key: tui.Key) bool {
    // If help overlay is visible, any key closes it
    if (state.help.visible) {
        state.help.hide();
        return false;
    }

    switch (key.code) {
        .ctrl_c, .escape => return true,
        .tab => {
            if (key.mods.shift) {
                state.tab.prev();
            } else {
                state.tab.next();
            }
        },
        .character => {
            if (key.char) |ch| {
                switch (ch) {
                    'q' => return true,
                    'p' => state.paused = !state.paused,
                    't' => state.theme_manager.nextTheme(),
                    'T' => state.theme_manager.prevTheme(),
                    '?' => state.help.toggle(),
                    'h' => state.help.toggle(),
                    '1'...'9' => state.tab.setActive(@as(usize, ch - '1')),
                    '0' => state.tab.setActive(9), // '0' maps to tab 10 (index 9)
                    else => {
                        // Forward unhandled character events to the active panel
                        const consumed = state.panels[state.tab.active].handleEvent(.{
                            .key = key,
                        }) catch false;
                        return consumed;
                    },
                }
            }
        },
        else => {
            // Forward unhandled non-character events to the active panel
            const consumed = state.panels[state.tab.active].handleEvent(.{
                .key = key,
            }) catch false;
            return consumed;
        },
    }
    return false;
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui dashboard [OPTIONS]
        \\
        \\Unified tabbed TUI dashboard combining all monitoring panels.
        \\
        \\Options:
        \\  --theme <name>    Set initial theme
        \\  --list-themes     Print available themes and exit
        \\  -h, --help        Show this help message
        \\
        \\Keyboard Controls:
        \\  1-9, 0            Switch to tab N (0 = tab 10)
        \\  Tab / Shift+Tab   Cycle tabs forward/back
        \\  t / T             Cycle themes forward/back
        \\  p                 Pause/Resume updates
        \\  ? / h             Toggle help overlay
        \\  q, Esc            Quit
        \\
        \\Individual panels are still available via:
        \\  abi ui gpu, abi ui brain, abi ui train, etc.
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
