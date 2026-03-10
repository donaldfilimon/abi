//! Shared UI shell command.
//!
//! Uses the generic dashboard runtime as the only shell host and renders a
//! tabbed panel stack with an in-shell command palette.

const std = @import("std");
const command = @import("../../../command");
const context_mod = @import("../../../framework/context");
const tui = @import("../../../terminal/mod.zig");

pub const meta: command.Meta = .{
    .name = "dashboard",
    .description = "Shared UI shell with tabbed panel stack and command palette",
};

pub fn forwardToView(
    ctx: *const context_mod.CommandContext,
    args: []const [:0]const u8,
    view_name: [:0]const u8,
) !void {
    const allocator = ctx.allocator;
    var forwarded = try std.ArrayList([:0]const u8).initCapacity(allocator, args.len + 2);
    defer forwarded.deinit(allocator);

    try forwarded.append(allocator, "--view");
    try forwarded.append(allocator, view_name);
    for (args) |arg| {
        try forwarded.append(allocator, arg);
    }

    try run(ctx, forwarded.items);
}

const launcher_actions = @import("../../../terminal/launcher/actions");
const launcher_palette = @import("../../../terminal/launcher/palette");
const panel_mod = @import("../../../terminal/panels/mod.zig");
const panel_registry = @import("../../../terminal/panels/registry");
const session_runner = @import("./session_runner");
const theme_options = @import("./theme_options");
const utils = @import("../../../utils/mod.zig");
const database_alloc = @import("abi").features.database.core.alloc;

const panel_count = panel_registry.panel_specs.len;

const help_lines = [_][]const u8{
    "  [1-9,0]  Jump to a shell tab",
    "  [Tab]    Next tab",
    "  [S-Tab]  Previous tab",
    "  [/]      Open command palette",
    "  [p]      Pause or resume updates",
    "  [t/T]    Next or previous theme",
    "  [?/h]    Toggle help",
    "  [q/Esc]  Quit",
};

const ShellPanel = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme: *const tui.Theme,
    tab: tui.TabBar,
    palette: launcher_palette.CommandPalette,
    panels: [panel_count]tui.Panel,
    boundaries: [panel_count]tui.Panel.ErrorBoundaryPanel,

    gpu_panel: panel_mod.gpu_monitor.GpuMonitor,
    agent_adapter: panel_mod.agent_adapter.AgentAdapter,
    training_adapter: panel_mod.training_adapter.TrainingAdapter,
    model_adapter: panel_mod.model_adapter.ModelAdapter,
    streaming_adapter: panel_mod.streaming_adapter.StreamingAdapter,
    db_adapter: panel_mod.db_adapter.DbAdapter,
    network_adapter: panel_mod.network_adapter.NetworkAdapter,
    bench_adapter: panel_mod.bench_adapter.BenchAdapter,
    brain_panel: panel_mod.brain_panel.BrainDashboardPanel,
    security_panel: panel_mod.SecurityPanel,
    connectors_panel: panel_mod.ConnectorsPanel,
    ralph_panel: panel_mod.RalphPanel,
    chat_adapter: panel_mod.chat_adapter.ChatAdapter,
    memory_panel: panel_mod.MemoryPanel,
    create_subagent_panel: panel_mod.CreateSubagentPanel,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        theme: *const tui.Theme,
    ) !ShellPanel {
        var boundaries: [panel_count]tui.Panel.ErrorBoundaryPanel = undefined;
        var panels: [panel_count]tui.Panel = undefined;
        for (&boundaries, &panels) |*boundary, *panel| {
            boundary.* = tui.Panel.withErrorBoundary(tui.Panel.noop_panel);
            panel.* = tui.Panel.noop_panel;
        }

        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme = theme,
            .tab = tui.TabBar.init(&panel_registry.tab_labels),
            .palette = try launcher_palette.CommandPalette.init(allocator),
            .panels = panels,
            .boundaries = boundaries,
            .gpu_panel = panel_mod.gpu_monitor.GpuMonitor.init(allocator, terminal, theme),
            .agent_adapter = panel_mod.agent_adapter.AgentAdapter.init(allocator, terminal, theme),
            .training_adapter = panel_mod.training_adapter.TrainingAdapter.init(allocator, terminal, theme),
            .model_adapter = panel_mod.model_adapter.ModelAdapter.init(allocator, terminal, theme),
            .streaming_adapter = try panel_mod.streaming_adapter.StreamingAdapter.init(allocator, terminal, theme),
            .db_adapter = panel_mod.db_adapter.DbAdapter.init(allocator, terminal, theme),
            .network_adapter = panel_mod.network_adapter.NetworkAdapter.init(allocator, terminal, theme),
            .bench_adapter = panel_mod.bench_adapter.BenchAdapter.init(allocator, terminal, theme),
            .brain_panel = panel_mod.brain_panel.BrainDashboardPanel.init(terminal, theme),
            .security_panel = panel_mod.SecurityPanel.init(allocator),
            .connectors_panel = panel_mod.ConnectorsPanel.init(allocator),
            .ralph_panel = panel_mod.RalphPanel.init(allocator),
            .chat_adapter = panel_mod.chat_adapter.ChatAdapter.init(allocator, terminal, theme),
            .memory_panel = panel_mod.MemoryPanel.init(allocator),
            .create_subagent_panel = panel_mod.CreateSubagentPanel.init(allocator),
        };
    }

    /// Connect the database allocator to the Memory panel for live stats.
    pub fn connectDatabaseTracker(self: *ShellPanel, tracker: *database_alloc.TrackingAllocator) void {
        self.memory_panel.connectDatabaseTracker(tracker);
    }

    pub fn wirePanels(self: *ShellPanel) void {
        self.setPanel(0, self.gpu_panel.panel());
        self.setPanel(1, self.agent_adapter.panel());
        self.setPanel(2, self.training_adapter.panel());
        self.setPanel(3, self.model_adapter.panel());
        self.setPanel(4, self.streaming_adapter.panel());
        self.setPanel(5, self.db_adapter.panel());
        self.setPanel(6, self.network_adapter.panel());
        self.setPanel(7, self.bench_adapter.panel());
        self.setPanel(8, self.brain_panel.panel());
        self.setPanel(9, self.security_panel.asPanel());
        self.setPanel(10, self.connectors_panel.asPanel());
        self.setPanel(11, self.ralph_panel.asPanel());
        self.setPanel(12, self.chat_adapter.panel());
        self.setPanel(13, self.memory_panel.asPanel());
        self.setPanel(14, self.create_subagent_panel.asPanel());
    }

    fn setPanel(self: *ShellPanel, index: usize, inner: tui.Panel) void {
        self.boundaries[index] = tui.Panel.withErrorBoundary(inner);
        self.panels[index] = self.boundaries[index].asPanel();
    }

    fn activePanel(self: *ShellPanel) tui.Panel {
        return self.panels[self.tab.active];
    }

    pub fn deinit(self: *ShellPanel) void {
        self.palette.deinit();
        for (&self.boundaries) |*boundary| {
            boundary.deinit();
        }
    }

    pub fn update(self: *ShellPanel) !void {
        try self.activePanel().tick();
    }

    pub fn render(
        self: *ShellPanel,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        const row: u16 = @intCast(start_row);
        const col: u16 = @intCast(start_col);
        const cols: u16 = @intCast(width);
        const rows: u16 = @intCast(height);

        if (cols < 24 or rows < 6) {
            try self.terminal.moveTo(row, col);
            try self.terminal.write(self.theme.warning);
            try self.terminal.write("Resize the shell pane to at least 24x6");
            try self.terminal.write(self.theme.reset);
            return;
        }

        try self.tab.render(self.terminal, self.theme, row, cols);

        const status_row = row + 1;
        try self.terminal.moveTo(status_row, col);
        try self.terminal.write(self.theme.text_dim);
        try self.terminal.write(" Active: ");
        try self.terminal.write(self.activePanel().getName());
        try self.terminal.write("  [/] palette  [Tab/S-Tab] views  [1-9,0] jump");
        try self.terminal.write(self.theme.reset);

        const content_rect = tui.Rect{
            .x = col,
            .y = row + 2,
            .width = cols,
            .height = rows -| 2,
        };
        if (!content_rect.isEmpty()) {
            try self.activePanel().render(self.terminal, content_rect, self.theme);
        }

        try self.palette.render(self.terminal, self.theme, self.terminal.size());
    }
};

const Dash = tui.dashboard.Dashboard(ShellPanel);

fn handleShellKeys(dash: *Dash, key: tui.Key) bool {
    if (dash.panel.palette.active) {
        const outcome = dash.panel.palette.handleKey(key) catch return false;
        switch (outcome) {
            .none, .close => return false,
            .quit => return true,
            .submit => |action| {
                return launcher_actions.executeWithTerminal(dash.allocator, dash.terminal, action, .{
                    .help_callback = printHelp,
                    .return_prompt = "Press Enter to return to the shared shell...",
                }) catch false;
            },
        }
    }

    switch (key.code) {
        .tab => {
            if (key.mods.shift) {
                dash.panel.tab.prev();
            } else {
                dash.panel.tab.next();
            }
            return false;
        },
        .character => {
            if (key.char) |ch| {
                switch (ch) {
                    '/' => {
                        dash.panel.palette.open() catch {};
                        return false;
                    },
                    '1'...'9' => {
                        dash.panel.tab.setActive(@as(usize, ch - '1'));
                        return false;
                    },
                    '0' => {
                        dash.panel.tab.setActive(9);
                        return false;
                    },
                    else => {},
                }
            }
        },
        else => {},
    }

    return dash.panel.activePanel().handleEvent(.{ .key = key }) catch false;
}

/// Entry point for the shared UI shell.
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

    if (parsed.remaining_args.len > 0) {
        utils.output.printError("Unknown argument for ui shell: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        return error.InvalidArgument;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    var session = session_runner.startSimpleDashboard(allocator, .{
        .dashboard_name = "UI Shell",
        .terminal_title = "ABI UI Shell",
    }) orelse return;
    defer session.deinit();

    const shell = try ShellPanel.init(allocator, &session.terminal, initial_theme);
    var dash = tui.dashboard.Dashboard(ShellPanel).init(allocator, &session.terminal, initial_theme, shell, .{
        .title = "ABI UI SHELL",
        .refresh_rate_ms = 100,
        .min_width = 60,
        .min_height = 16,
        .help_keys = " [q]uit  [Tab]views  [/]palette  [1-9,0]jump  [p]ause  [t]heme  [?]help",
        .help_title = "UI Shell Help",
        .help_lines = &help_lines,
    });
    dash.panel.wirePanels();
    dash.extra_key_handler = handleShellKeys;
    defer dash.deinit();

    try dash.run();
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui [OPTIONS]
        \\
        \\Open the shared terminal shell with tabbed views and a command palette.
        \\
        \\Options:
        \\  --theme <name>    Set initial theme
        \\  --list-themes     Print available themes and exit
        \\  -h, --help        Show this help message
        \\
        \\Keyboard Controls:
        \\  1-9, 0            Jump to a tab
        \\  /                 Open the command palette
        \\  Tab / Shift+Tab   Cycle tabs forward or backward
        \\  t / T             Cycle themes
        \\  p                 Pause or resume panel updates
        \\  ? / h             Toggle help
        \\  q / Esc           Quit
        \\
        \\Focused views remain available via:
        \\  abi ui gpu, abi ui brain, abi ui model, abi ui editor, ...
        \\
    , .{});
}

test "shell panel tab metadata matches the registry" {
    try std.testing.expectEqual(panel_count, panel_registry.tab_labels.len);
    try std.testing.expectEqual(panel_count, panel_registry.panel_specs.len);
}

test {
    std.testing.refAllDecls(@This());
}
