//! Brain dashboard command on the shared dashboard runtime.

const std = @import("std");
const context_mod = @import("../../../framework/context.zig");
const tui = @import("../../../terminal/mod.zig");
const dsl = @import("../../../terminal/dsl/mod.zig");
const utils = @import("../../../utils/mod.zig");

const Dash = tui.dashboard.Dashboard(BrainPanelState);

const ViewMode = enum {
    animation,
    dashboard,

    fn label(self: ViewMode) []const u8 {
        return switch (self) {
            .animation => "Animation",
            .dashboard => "Dashboard",
        };
    }
};

const DataSource = enum {
    simulated,
    live,
    training,

    fn label(self: DataSource) []const u8 {
        return switch (self) {
            .simulated => "Simulated",
            .live => "Live",
            .training => "Training",
        };
    }
};

const BrainPanelState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme: *const tui.Theme,
    view_mode: ViewMode,
    data_source: DataSource,
    tick_count: u64,
    anim: tui.BrainAnimation,
    panel: tui.BrainDashboardPanel,
    data: tui.BrainDashboardData,
    metrics_reader: ?tui.MetricsFileReader,
    brain_mapper: ?tui.TrainingBrainMapper,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        theme: *const tui.Theme,
        data_source: DataSource,
        training_path: ?[]const u8,
    ) BrainPanelState {
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme = theme,
            .view_mode = .animation,
            .data_source = data_source,
            .tick_count = 0,
            .anim = tui.BrainAnimation.init(),
            .panel = tui.BrainDashboardPanel.init(terminal, theme),
            .data = tui.BrainDashboardData.init(),
            .metrics_reader = if (training_path) |path| tui.MetricsFileReader.init(path) else null,
            .brain_mapper = if (data_source == .training) tui.TrainingBrainMapper.init() else null,
        };
    }

    pub fn deinit(_: *BrainPanelState) void {}

    pub fn update(self: *BrainPanelState) !void {
        self.tick_count +%= 1;

        switch (self.data_source) {
            .training => {
                if (self.metrics_reader) |*reader| {
                    _ = reader.poll();
                    if (self.brain_mapper) |*mapper| {
                        mapper.updateDashboardData(&self.data, reader.getMetrics(), null);
                        self.anim.updateTrainingDynamics(
                            self.data.train_loss,
                            self.data.train_accuracy,
                            self.data.learning_rate_current,
                        );
                    }
                }
            },
            else => {
                self.data.updateSimulated(self.tick_count);
            },
        }

        self.anim.updateFromData(&self.data.node_activity);
    }

    pub fn toggleView(self: *BrainPanelState) []const u8 {
        self.view_mode = switch (self.view_mode) {
            .animation => .dashboard,
            .dashboard => .animation,
        };
        return switch (self.view_mode) {
            .animation => "Animation View",
            .dashboard => "Dashboard View",
        };
    }

    pub fn render(
        self: *BrainPanelState,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        const row: u16 = @intCast(start_row);
        const col: u16 = @intCast(start_col);
        const cols: u16 = @intCast(width);
        const rows: u16 = @intCast(height);

        if (cols < 40 or rows < 12) {
            try self.terminal.moveTo(row, col);
            try self.terminal.write(self.theme.warning);
            try self.terminal.write("Resize terminal to at least 40x12");
            try self.terminal.write(self.theme.reset);
            return;
        }

        self.panel.term = self.terminal;
        self.panel.theme = self.theme;

        try self.terminal.moveTo(row, col);
        try self.terminal.write(self.theme.text_dim);
        try self.terminal.write(" View: ");
        try self.terminal.write(self.view_mode.label());
        try self.terminal.write("  Data: ");
        try self.terminal.write(self.data_source.label());
        try self.terminal.write("  [Tab] switch view");
        try self.terminal.write(self.theme.reset);

        switch (self.view_mode) {
            .animation => try self.anim.render(self.terminal, self.theme, row + 1, col, cols, rows -| 1),
            .dashboard => try self.panel.render(&self.data, row + 1, col, cols, rows -| 1),
        }
    }
};

fn handleBrainKeys(dash: *Dash, key: tui.Key) bool {
    if (key.code == .tab) {
        dash.showNotification(dash.panel.toggleView());
        return false;
    }
    return false;
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    remaining_args: []const [:0]const u8,
) !BrainPanelState {
    var data_source: DataSource = .simulated;
    var training_path: ?[]const u8 = null;

    var arg_idx: usize = 0;
    while (arg_idx < remaining_args.len) : (arg_idx += 1) {
        const arg = std.mem.sliceTo(remaining_args[arg_idx], 0);
        if (std.mem.eql(u8, arg, "--db") or std.mem.eql(u8, arg, "--live")) {
            data_source = .live;
        } else if (std.mem.eql(u8, arg, "--training")) {
            data_source = .training;
            if (arg_idx + 1 >= remaining_args.len) {
                utils.output.printError("--training requires a metrics JSONL file path.", .{});
                return error.InvalidArgument;
            }
            arg_idx += 1;
            training_path = std.mem.sliceTo(remaining_args[arg_idx], 0);
        } else {
            utils.output.printError("Unknown argument for ui brain: {s}", .{arg});
            @import("./theme_options.zig").printThemeHint();
            return error.InvalidArgument;
        }
    }

    if (data_source == .live) {
        utils.output.printInfo("Live WDBX mode is not wired yet; using simulated data.", .{});
        data_source = .simulated;
    }

    return BrainPanelState.init(allocator, terminal, initial_theme, data_source, training_path);
}

fn validateArgs(_: []const [:0]const u8) !void {
    // Args are validated inside initPanel where we have context to parse them
}

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(BrainPanelState, ctx, args, .{
        .dashboard_name = "Brain Dashboard",
        .terminal_title = "ABI Brain Dashboard",
        .title = "ABI BRAIN VISUALIZER",
        .refresh_rate_ms = 100,
        .min_width = 40,
        .min_height = 12,
        .help_keys = " [q]uit  [Tab]view  [p]ause  [t]heme  [?]help",
        .print_help = printHelp,
        .init_panel = initPanel,
        .validate_args = validateArgs,
        .extra_key_handler = handleBrainKeys,
    });
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui brain [OPTIONS]
        \\
        \\Launch the shared brain dashboard with animation and metric views.
        \\
        \\Options:
        \\  --theme <name>       Set initial theme
        \\  --list-themes        Print available themes and exit
        \\  --db, --live         Reserved live data mode (currently simulated)
        \\  --training <path>    Use a training metrics JSONL file
        \\  -h, --help           Show this help message
        \\
        \\Keyboard Controls:
        \\  Tab             Switch between animation and dashboard views
        \\  q / Esc         Quit
        \\  p               Pause or resume updates
        \\  t / T           Cycle themes
        \\  ? / h           Toggle help
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
