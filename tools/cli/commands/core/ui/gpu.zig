//! GPU dashboard command on the shared dashboard runtime.

const std = @import("std");
const context_mod = @import("../../../framework/context.zig");
const tui = @import("../../../terminal/mod.zig");
const dsl = @import("../../../terminal/dsl/mod.zig");
const utils = @import("../../../utils/mod.zig");

const Dash = tui.dashboard.Dashboard(GpuPanel);

const GpuPanel = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme: *const tui.Theme,
    gpu_monitor: tui.GpuMonitor,
    agent_panel: tui.AgentPanel,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        theme: *const tui.Theme,
    ) GpuPanel {
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme = theme,
            .gpu_monitor = tui.GpuMonitor.init(allocator, terminal, theme),
            .agent_panel = tui.AgentPanel.init(allocator, terminal, theme),
        };
    }

    pub fn deinit(self: *GpuPanel) void {
        self.gpu_monitor.deinit();
        self.agent_panel.deinit();
    }

    pub fn update(self: *GpuPanel) !void {
        try self.gpu_monitor.update();
        try self.agent_panel.update();
    }

    pub fn resetStats(self: *GpuPanel) void {
        self.gpu_monitor.clearDevices();
        self.gpu_monitor.update_counter = 0;
        self.agent_panel.episode_count = 0;
        self.agent_panel.total_reward = 0;
        self.agent_panel.exploration_rate = 1.0;
        self.agent_panel.phase = .exploration;
        self.agent_panel.update_counter = 0;
    }

    pub fn render(
        self: *GpuPanel,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        const row: u16 = @intCast(start_row);
        const col: u16 = @intCast(start_col);
        const cols: u16 = @intCast(width);
        const rows: u16 = @intCast(height);

        if (cols < 40 or rows < 8) {
            try self.terminal.moveTo(row, col);
            try self.terminal.write(self.theme.warning);
            try self.terminal.write("Resize terminal to at least 40x8");
            try self.terminal.write(self.theme.reset);
            return;
        }

        self.gpu_monitor.theme = self.theme;
        self.agent_panel.theme = self.theme;

        const left_width = cols / 2;
        const right_width = cols - left_width;
        try self.gpu_monitor.render(row, col, left_width, rows);
        try self.agent_panel.render(row, col + left_width, right_width, rows);
    }
};

fn handleGpuKeys(dash: *Dash, key: tui.Key) bool {
    if (key.code == .character) {
        if (key.char) |ch| {
            if (ch == 'r') {
                dash.panel.resetStats();
                dash.showNotification("Stats reset");
                return false;
            }
        }
    }
    return false;
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    _: []const [:0]const u8,
) !GpuPanel {
    return GpuPanel.init(allocator, terminal, initial_theme);
}

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(GpuPanel, ctx, args, .{
        .dashboard_name = "GPU Dashboard",
        .terminal_title = "ABI GPU Dashboard",
        .title = "ABI GPU CONTROL PLANE",
        .refresh_rate_ms = 100,
        .min_width = 40,
        .min_height = 10,
        .help_keys = " [q]uit  [p]ause  [t]heme  [r]eset  [?]help",
        .print_help = printHelp,
        .init_panel = initPanel,
        .extra_key_handler = handleGpuKeys,
    });
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui gpu [OPTIONS]
        \\
        \\Launch the shared GPU dashboard with a split GPU and agent view.
        \\
        \\Options:
        \\  --theme <name>    Set initial theme
        \\  --list-themes     Print available themes and exit
        \\  -h, --help        Show this help message
        \\
        \\Keyboard Controls:
        \\  q / Esc           Quit
        \\  p                 Pause or resume updates
        \\  t / T             Cycle themes
        \\  r                 Reset GPU and agent stats
        \\  ? / h             Toggle help
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
