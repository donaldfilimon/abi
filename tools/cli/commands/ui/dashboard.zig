//! Unified Dashboard Command
//!
//! Opens a tabbed TUI dashboard combining all monitoring panels
//! in a single interface. Individual panels remain available
//! via `abi ui gpu`, `abi ui brain`, etc.
//! Uses AsyncLoop for timer-driven refresh (10 FPS).

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const panel_registry = @import("../../tui/panels/registry.zig");
const security_panel_mod = @import("../../tui/panels/security_panel.zig");
const connectors_panel_mod = @import("../../tui/panels/connectors_panel.zig");
const ralph_panel_mod = @import("../../tui/panels/ralph_panel.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");
const render_utils = tui.render_utils;

// ===============================================================================
// Constants
// ===============================================================================

const panel_count = 12;
const tab_gpu = 0;
const tab_agent = 1;
const tab_train = 2;
const tab_model = 3;
const tab_stream = 4;
const tab_db = 5;
const tab_net = 6;
const tab_bench = 7;
const tab_brain = 8;
const tab_security = 9;
const tab_connectors = 10;
const tab_ralph = 11;

inline fn panelLabel(comptime idx: usize) []const u8 {
    return panel_registry.panel_specs[idx].label;
}

inline fn panelShortcut(comptime idx: usize) []const u8 {
    return panel_registry.panel_specs[idx].shortcut_hint;
}

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
const SecurityPanel = security_panel_mod.SecurityPanel;
const ConnectorsPanel = connectors_panel_mod.ConnectorsPanel;
const RalphPanel = ralph_panel_mod.RalphPanel;

const GpuAdapter = struct {
    inner: tui.GpuMonitor,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) GpuAdapter {
        return .{ .inner = tui.GpuMonitor.init(allocator, term, theme) };
    }

    pub fn render(self: *GpuAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term;
    }

    pub fn tick(self: *GpuAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *GpuAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *GpuAdapter) []const u8 {
        return panelLabel(tab_gpu);
    }

    pub fn shortcutHint(_: *GpuAdapter) []const u8 {
        return panelShortcut(tab_gpu);
    }

    pub fn deinit(self: *GpuAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *GpuAdapter) tui.Panel {
        return tui.Panel.from(GpuAdapter, self);
    }
};

const AgentAdapter = struct {
    inner: tui.AgentPanel,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) AgentAdapter {
        return .{ .inner = tui.AgentPanel.init(allocator, term, theme) };
    }

    pub fn render(self: *AgentAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term;
    }

    pub fn tick(self: *AgentAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *AgentAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *AgentAdapter) []const u8 {
        return panelLabel(tab_agent);
    }

    pub fn shortcutHint(_: *AgentAdapter) []const u8 {
        return panelShortcut(tab_agent);
    }

    pub fn deinit(self: *AgentAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *AgentAdapter) tui.Panel {
        return tui.Panel.from(AgentAdapter, self);
    }
};

const TrainingAdapter = struct {
    inner: tui.TrainingPanel,

    pub fn init(allocator: std.mem.Allocator, theme: *const tui.Theme) TrainingAdapter {
        return .{ .inner = tui.TrainingPanel.init(allocator, theme, .{}) };
    }

    pub fn render(self: *TrainingAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        self.inner.width = rect.width;
        try renderTrainingPanelInRect(self.inner.allocator, &self.inner, term, rect);
    }

    pub fn tick(self: *TrainingAdapter) anyerror!void {
        _ = try self.inner.pollMetrics();
    }

    pub fn handleEvent(_: *TrainingAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *TrainingAdapter) []const u8 {
        return panelLabel(tab_train);
    }

    pub fn shortcutHint(_: *TrainingAdapter) []const u8 {
        return panelShortcut(tab_train);
    }

    pub fn deinit(self: *TrainingAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *TrainingAdapter) tui.Panel {
        return tui.Panel.from(TrainingAdapter, self);
    }
};

fn renderTrainingPanelInRect(
    allocator: std.mem.Allocator,
    panel: *tui.TrainingPanel,
    term: *tui.Terminal,
    rect: tui.Rect,
) !void {
    if (rect.isEmpty()) return;

    var render_buf = std.ArrayListUnmanaged(u8).empty;
    defer render_buf.deinit(allocator);

    const BufferWriter = struct {
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged(u8),

        pub const Error = anyerror;

        pub fn print(self: @This(), comptime fmt: []const u8, args: anytype) anyerror!void {
            const text = try std.fmt.allocPrint(self.allocator, fmt, args);
            defer self.allocator.free(text);
            try self.out.appendSlice(self.allocator, text);
        }
    };
    const writer = BufferWriter{
        .allocator = allocator,
        .out = &render_buf,
    };
    try panel.render(writer);

    var row: usize = 0;
    var lines = std.mem.splitScalar(u8, render_buf.items, '\n');
    while (lines.next()) |line| : (row += 1) {
        if (row >= @as(usize, rect.height)) break;
        const trimmed = if (line.len > 0 and line[line.len - 1] == '\r')
            line[0 .. line.len - 1]
        else
            line;

        try term.moveTo(rect.y + @as(u16, @intCast(row)), rect.x);
        _ = try render_utils.writeClipped(term, trimmed, rect.width);
    }
}

const ModelAdapter = struct {
    inner: tui.ModelManagementPanel,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) ModelAdapter {
        return .{ .inner = tui.ModelManagementPanel.init(allocator, term, theme) };
    }

    pub fn render(self: *ModelAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(@intCast(rect.y), @intCast(rect.x), @intCast(rect.width), @intCast(rect.height));
        _ = term;
    }

    pub fn tick(self: *ModelAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *ModelAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *ModelAdapter) []const u8 {
        return panelLabel(tab_model);
    }

    pub fn shortcutHint(_: *ModelAdapter) []const u8 {
        return panelShortcut(tab_model);
    }

    pub fn deinit(self: *ModelAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *ModelAdapter) tui.Panel {
        return tui.Panel.from(ModelAdapter, self);
    }
};

const StreamingAdapter = struct {
    inner: tui.StreamingDashboard,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) !StreamingAdapter {
        return .{ .inner = try tui.StreamingDashboard.init(allocator, term, theme, "localhost:8080") };
    }

    pub fn render(self: *StreamingAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(@intCast(rect.y), @intCast(rect.x), @intCast(rect.width), @intCast(rect.height));
        _ = term;
    }

    pub fn tick(self: *StreamingAdapter) anyerror!void {
        try self.inner.pollMetrics();
    }

    pub fn handleEvent(_: *StreamingAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *StreamingAdapter) []const u8 {
        return panelLabel(tab_stream);
    }

    pub fn shortcutHint(_: *StreamingAdapter) []const u8 {
        return panelShortcut(tab_stream);
    }

    pub fn deinit(self: *StreamingAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *StreamingAdapter) tui.Panel {
        return tui.Panel.from(StreamingAdapter, self);
    }
};

const DbAdapter = struct {
    inner: tui.DatabasePanel,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) DbAdapter {
        return .{ .inner = tui.DatabasePanel.init(allocator, term, theme) };
    }

    pub fn render(self: *DbAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term;
    }

    pub fn tick(self: *DbAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *DbAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *DbAdapter) []const u8 {
        return panelLabel(tab_db);
    }

    pub fn shortcutHint(_: *DbAdapter) []const u8 {
        return panelShortcut(tab_db);
    }

    pub fn deinit(self: *DbAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *DbAdapter) tui.Panel {
        return tui.Panel.from(DbAdapter, self);
    }
};

const NetworkAdapter = struct {
    inner: tui.NetworkPanel,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) NetworkAdapter {
        return .{ .inner = tui.NetworkPanel.init(allocator, term, theme) };
    }

    pub fn render(self: *NetworkAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term;
    }

    pub fn tick(self: *NetworkAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *NetworkAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *NetworkAdapter) []const u8 {
        return panelLabel(tab_net);
    }

    pub fn shortcutHint(_: *NetworkAdapter) []const u8 {
        return panelShortcut(tab_net);
    }

    pub fn deinit(self: *NetworkAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *NetworkAdapter) tui.Panel {
        return tui.Panel.from(NetworkAdapter, self);
    }
};

const BenchAdapter = struct {
    inner: tui.BenchmarkPanel,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const tui.Theme) BenchAdapter {
        return .{ .inner = tui.BenchmarkPanel.init(allocator, term, theme) };
    }

    pub fn render(self: *BenchAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term;
    }

    pub fn tick(self: *BenchAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *BenchAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *BenchAdapter) []const u8 {
        return panelLabel(tab_bench);
    }

    pub fn shortcutHint(_: *BenchAdapter) []const u8 {
        return panelShortcut(tab_bench);
    }

    pub fn deinit(self: *BenchAdapter) void {
        self.inner.deinit();
    }

    pub fn panel(self: *BenchAdapter) tui.Panel {
        return tui.Panel.from(BenchAdapter, self);
    }
};

const BrainAdapter = struct {
    inner: tui.BrainDashboardPanel,
    data: tui.BrainDashboardData,

    pub fn init(term: *tui.Terminal, theme: *const tui.Theme) BrainAdapter {
        return .{
            .inner = tui.BrainDashboardPanel.init(term, theme),
            .data = tui.BrainDashboardData.init(),
        };
    }

    pub fn render(self: *BrainAdapter, term: *tui.Terminal, rect: tui.Rect, theme: *const tui.Theme) anyerror!void {
        self.inner.theme = theme;
        self.inner.term = term;
        try self.inner.render(&self.data, rect.y, rect.x, rect.width, rect.height);
    }

    pub fn tick(_: *BrainAdapter) anyerror!void {}

    pub fn handleEvent(_: *BrainAdapter, _: tui.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *BrainAdapter) []const u8 {
        return panelLabel(tab_brain);
    }

    pub fn shortcutHint(_: *BrainAdapter) []const u8 {
        return panelShortcut(tab_brain);
    }

    pub fn deinit(_: *BrainAdapter) void {}

    pub fn panel(self: *BrainAdapter) tui.Panel {
        return tui.Panel.from(BrainAdapter, self);
    }
};

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

    // Owned panel instances with stable lifetime for Panel vtable pointers.
    gpu_adapter: GpuAdapter,
    agent_adapter: AgentAdapter,
    training_adapter: TrainingAdapter,
    model_adapter: ModelAdapter,
    streaming_adapter: StreamingAdapter,
    db_adapter: DbAdapter,
    network_adapter: NetworkAdapter,
    bench_adapter: BenchAdapter,
    brain_adapter: BrainAdapter,
    security_panel: SecurityPanel,
    connectors_panel: ConnectorsPanel,
    ralph_panel: RalphPanel,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal_ptr: *tui.Terminal,
        initial_theme: *const tui.Theme,
        tab_labels: []const []const u8,
    ) !DashboardState {
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

        const streaming_adapter = try StreamingAdapter.init(
            allocator,
            terminal_ptr,
            initial_theme,
        );

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
            .gpu_adapter = GpuAdapter.init(allocator, terminal_ptr, initial_theme),
            .agent_adapter = AgentAdapter.init(allocator, terminal_ptr, initial_theme),
            .training_adapter = TrainingAdapter.init(allocator, initial_theme),
            .model_adapter = ModelAdapter.init(allocator, terminal_ptr, initial_theme),
            .streaming_adapter = streaming_adapter,
            .db_adapter = DbAdapter.init(allocator, terminal_ptr, initial_theme),
            .network_adapter = NetworkAdapter.init(allocator, terminal_ptr, initial_theme),
            .bench_adapter = BenchAdapter.init(allocator, terminal_ptr, initial_theme),
            .brain_adapter = BrainAdapter.init(terminal_ptr, initial_theme),
            .security_panel = SecurityPanel.init(allocator),
            .connectors_panel = ConnectorsPanel.init(allocator),
            .ralph_panel = RalphPanel.init(allocator),
        };

        // Panel pointers are wired after init once the struct has stable
        // memory in the caller's stack frame.
        _ = &state;

        return state;
    }

    pub fn deinit(self: *DashboardState) void {
        for (&self.boundaries) |*boundary| {
            boundary.deinit();
        }
    }

    /// Finalize panel wiring after the struct has stable memory.
    /// Must be called once after init, before the event loop starts.
    pub fn wirePanels(self: *DashboardState) void {
        self.setPanel(tab_gpu, self.gpu_adapter.panel());
        self.setPanel(tab_agent, self.agent_adapter.panel());
        self.setPanel(tab_train, self.training_adapter.panel());
        self.setPanel(tab_model, self.model_adapter.panel());
        self.setPanel(tab_stream, self.streaming_adapter.panel());
        self.setPanel(tab_db, self.db_adapter.panel());
        self.setPanel(tab_net, self.network_adapter.panel());
        self.setPanel(tab_bench, self.bench_adapter.panel());
        self.setPanel(tab_brain, self.brain_adapter.panel());
        self.setPanel(tab_security, self.security_panel.asPanel());
        self.setPanel(tab_connectors, self.connectors_panel.asPanel());
        self.setPanel(tab_ralph, self.ralph_panel.asPanel());
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

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        utils.output.printError("Failed to start dashboard: {t}", .{err});
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Dashboard") catch {};

    var state = try DashboardState.init(allocator, &terminal, initial_theme, &panel_registry.tab_labels);
    defer state.deinit();

    // Finalize panel wiring now that state has stable stack memory.
    // Each panel slot is wrapped in an ErrorBoundaryPanel so that a single
    // panel crashing renders a fallback error message instead of killing
    // the entire dashboard.
    state.wirePanels();

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
