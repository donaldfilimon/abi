//! Brain Dashboard Command
//!
//! Interactive TUI combining a 3D brain animation with a 6-panel
//! data dashboard showing WDBX vector database stats and Abbey
//! learning metrics. Toggle views with Tab.
//! Uses AsyncLoop for timer-driven refresh (10 FPS).

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");
const style_adapter = @import("../tui/style_adapter.zig");
const brain_animation = @import("../../tui/brain_animation.zig");
const brain_panel = @import("../../tui/brain_panel.zig");
const metrics_file_reader = @import("../../tui/metrics_file_reader.zig");
const training_brain_mapper = @import("../../tui/training_brain_mapper.zig");

// ===============================================================================
// Types
// ===============================================================================

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

const BrainState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme_manager: tui.ThemeManager,
    view_mode: ViewMode,
    data_source: DataSource,
    paused: bool,
    show_help: bool,
    frame_count: u64,
    tick_count: u64,
    notification: ?[]const u8,
    notification_time: i64,
    term_size: tui.TerminalSize,

    // Components
    anim: brain_animation.BrainAnimation,
    panel: brain_panel.BrainDashboardPanel,
    data: brain_panel.DashboardData,

    // Training mode components (only used when data_source == .training)
    metrics_reader: ?metrics_file_reader.MetricsFileReader,
    brain_mapper: ?training_brain_mapper.TrainingBrainMapper,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        initial_theme: *const tui.Theme,
        data_source: DataSource,
        training_path: ?[]const u8,
    ) BrainState {
        var theme_manager = tui.ThemeManager.init();
        theme_manager.current = initial_theme;
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme_manager = theme_manager,
            .view_mode = .animation,
            .data_source = data_source,
            .paused = false,
            .show_help = false,
            .frame_count = 0,
            .tick_count = 0,
            .notification = null,
            .notification_time = 0,
            .term_size = terminal.size(),
            .anim = brain_animation.BrainAnimation.init(),
            .panel = brain_panel.BrainDashboardPanel.init(terminal, initial_theme),
            .data = brain_panel.DashboardData.init(),
            .metrics_reader = if (training_path) |p| metrics_file_reader.MetricsFileReader.init(p) else null,
            .brain_mapper = if (data_source == .training) training_brain_mapper.TrainingBrainMapper.init() else null,
        };
    }

    pub fn theme(self: *const BrainState) *const tui.Theme {
        return self.theme_manager.current;
    }

    pub fn showNotification(self: *BrainState, msg: []const u8) void {
        self.notification = msg;
        self.notification_time = abi.shared.utils.unixMs();
    }

    pub fn clearExpiredNotification(self: *BrainState) void {
        if (self.notification != null) {
            const elapsed = abi.shared.utils.unixMs() - self.notification_time;
            if (elapsed > 2000) {
                self.notification = null;
            }
        }
    }

    pub fn updateTheme(self: *BrainState) void {
        self.panel.theme = self.theme_manager.current;
    }
};

// ===============================================================================
// Box Drawing
// ===============================================================================

const box = tui.render_utils.boxChars(.rounded);

// ===============================================================================
// Entry Point
// ===============================================================================

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

    // Parse brain-specific args
    var data_source: DataSource = .simulated;
    var training_path: ?[]const u8 = null;
    var arg_idx: usize = 0;
    while (arg_idx < parsed.remaining_args.len) : (arg_idx += 1) {
        const a = std.mem.sliceTo(parsed.remaining_args[arg_idx], 0);
        if (std.mem.eql(u8, a, "--db") or std.mem.eql(u8, a, "--live")) {
            data_source = .live;
        } else if (std.mem.eql(u8, a, "--training")) {
            data_source = .training;
            if (arg_idx + 1 < parsed.remaining_args.len) {
                arg_idx += 1;
                training_path = std.mem.sliceTo(parsed.remaining_args[arg_idx], 0);
            }
        }
    }

    // Live mode requires database feature and a running WDBX instance.
    // For now, fall back to simulated data until WDBX polling is wired.
    if (data_source == .live) {
        utils.output.printInfo("Live WDBX mode not yet connected; using simulated data.", .{});
        data_source = .simulated;
    }

    // Training mode requires a metrics file path
    if (data_source == .training and training_path == null) {
        utils.output.printError("--training requires a metrics JSONL file path.", .{});
        return;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runBrainDashboard(allocator, initial_theme, data_source, training_path);
}

fn runBrainDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme, data_source: DataSource, training_path: ?[]const u8) !void {
    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("Brain dashboard requires a terminal. Platform: {s}", .{caps.platform_name});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        utils.output.printError("Failed to start Brain Dashboard: {t}", .{err});
        printFallbackInfo();
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Brain Dashboard") catch {};

    var state = BrainState.init(allocator, &terminal, initial_theme, data_source, training_path);

    var loop = tui.AsyncLoop.init(allocator, &terminal, .{
        .refresh_rate_ms = 100,
        .input_poll_ms = 16,
        .auto_resize = true,
    });
    defer loop.deinit();

    loop.setRenderCallback(&brainRender);
    loop.setTickCallback(&brainTick);
    loop.setUpdateCallback(&brainUpdate);
    loop.setUserData(@ptrCast(&state));

    try loop.run();
}

// ===============================================================================
// AsyncLoop Callbacks
// ===============================================================================

fn brainRender(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(BrainState) orelse return error.UserDataNotSet;
    try state.terminal.clear();
    try renderBrain(state);
    state.frame_count = loop.getFrameCount();
}

fn brainTick(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(BrainState) orelse return error.UserDataNotSet;
    state.term_size = state.terminal.size();
    state.clearExpiredNotification();

    if (!state.paused) {
        state.tick_count +%= 1;

        switch (state.data_source) {
            .training => {
                // Poll JSONL file for new metrics, map into dashboard data
                if (state.metrics_reader) |*reader| {
                    _ = reader.poll();
                    if (state.brain_mapper) |*mapper| {
                        mapper.updateDashboardData(&state.data, reader.getMetrics(), null);
                        // Drive animation dynamics from training state
                        state.anim.updateTrainingDynamics(
                            state.data.train_loss,
                            state.data.train_accuracy,
                            state.data.learning_rate_current,
                        );
                    }
                }
            },
            else => {
                state.data.updateSimulated(state.tick_count);
            },
        }

        state.anim.updateFromData(&state.data.node_activity);
    }
}

fn brainUpdate(loop: *tui.AsyncLoop, event: tui.AsyncEvent) anyerror!bool {
    const state = loop.getUserData(BrainState) orelse return error.UserDataNotSet;
    return switch (event) {
        .input => |ev| switch (ev) {
            .key => |key| handleKeyEvent(state, key),
            .mouse => false,
        },
        .resize => |size| blk: {
            state.term_size = .{ .rows = size.rows, .cols = size.cols };
            break :blk false;
        },
        .quit => true,
        else => false,
    };
}

// ===============================================================================
// Input Handling
// ===============================================================================

fn handleKeyEvent(state: *BrainState, key: tui.Key) bool {
    if (state.show_help) {
        switch (key.code) {
            .escape, .enter => state.show_help = false,
            .character => {
                if (key.char) |ch| {
                    if (ch == '?' or ch == 'q' or ch == 'h') state.show_help = false;
                }
            },
            else => {},
        }
        return false;
    }

    switch (key.code) {
        .ctrl_c, .escape => return true,
        .tab => {
            state.view_mode = switch (state.view_mode) {
                .animation => .dashboard,
                .dashboard => .animation,
            };
            state.showNotification(switch (state.view_mode) {
                .animation => "Animation View",
                .dashboard => "Dashboard View",
            });
        },
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
                    '?' => state.show_help = true,
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
const MIN_ROWS: u16 = 12;

fn renderBrain(state: *BrainState) !void {
    const term = state.terminal;
    const theme_val = state.theme();
    const width = state.term_size.cols;
    const height = state.term_size.rows;

    if (width < MIN_COLS or height < MIN_ROWS) {
        try renderResizeMessage(term, theme_val, width, height);
        return;
    }

    if (state.show_help) {
        try renderHelpOverlay(term, theme_val, width, height);
        return;
    }

    // Title bar (3 rows)
    try renderTitleBar(term, theme_val, state, width);

    // Main content area
    const content_start: u16 = 3;
    const content_height: u16 = @max(8, height -| 6);

    switch (state.view_mode) {
        .animation => {
            try state.anim.render(term, theme_val, content_start, 0, width, content_height);
        },
        .dashboard => {
            try state.panel.render(&state.data, content_start, 0, width, content_height);
        },
    }

    // Notification
    if (state.notification) |msg| {
        try renderNotification(term, theme_val, msg, height - 3, width);
    }

    // Status bar
    try renderStatusBar(term, theme_val, state, height - 2, width);

    // Help bar
    try renderHelpBar(term, theme_val, height - 1, width);
}

fn renderTitleBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const BrainState, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    const inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    const title = " ABI BRAIN VISUALIZER ";
    const mode_label = if (state.paused) "PAUSED" else "LIVE";

    const right_width = mode_label.len + state.view_mode.label().len + 10;
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

    // Title line
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);
    try term.write(theme_val.bold);
    try term.write(chrome.title);
    try term.write(left_display);
    try term.write(theme_val.reset);
    try writeRepeat(term, " ", gap);

    // Mode chip
    try term.write(chrome.chip_bg);
    try term.write(if (state.paused) chrome.paused else chrome.live);
    try term.write("[");
    try term.write(mode_label);
    try term.write("]");
    try term.write(theme_val.reset);
    try term.write(" ");

    // View chip
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write("[");
    try term.write(state.view_mode.label());
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

fn renderStatusBar(term: *tui.Terminal, theme_val: *const tui.Theme, state: *const BrainState, row: u16, width: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    try setCursorPosition(term, row, 0);

    // Separator
    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, @as(usize, width) - 2);
    try term.write(box.rsep);
    try term.write(theme_val.reset);
    try term.write("\n");

    // Status line
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(theme_val.reset);

    var buf: [64]u8 = undefined;

    try term.write(" ");
    try writeChip(term, chrome, theme_val, "view", state.view_mode.label());

    try term.write("  ");
    const frame_str = std.fmt.bufPrint(&buf, "{d}", .{state.frame_count}) catch "?";
    try writeChip(term, chrome, theme_val, "frame", frame_str);

    try term.write("  ");
    const vec_str = std.fmt.bufPrint(&buf, "{d}", .{state.data.vector_count}) catch "?";
    try writeChip(term, chrome, theme_val, "vecs", vec_str);

    try term.write("  ");
    try writeChip(term, chrome, theme_val, "data", state.data_source.label());

    // Pad and close
    const status_inner: usize = if (width >= 2) @as(usize, width) - 2 else 0;
    const approx_used: usize = 60 + frame_str.len + vec_str.len;
    if (approx_used < status_inner) try writeRepeat(term, " ", status_inner - approx_used);

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
    if (inner >= 60) {
        try writeKeyHint(term, chrome, theme_val, "Tab", "switch");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "q", "quit");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "p", "pause");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "t", "theme");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "?", "help");
    } else if (inner >= 30) {
        try writeKeyHint(term, chrome, theme_val, "Tab", "switch");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "q", "quit");
        try term.write(" ");
        try writeKeyHint(term, chrome, theme_val, "?", "help");
    }
}

fn renderResizeMessage(term: *tui.Terminal, theme_val: *const tui.Theme, width: u16, height: u16) !void {
    const msg = "Resize terminal to at least 40x12";
    const row: u16 = if (height >= 2) (height - 1) / 2 else 0;
    const msg_len_u16: u16 = @intCast(msg.len);
    const col: u16 = if (width >= msg_len_u16 + 2) (width - msg_len_u16) / 2 else 0;
    try setCursorPosition(term, row, col);
    try term.write(theme_val.warning);
    try term.write(msg);
    try term.write(theme_val.reset);
    try setCursorPosition(term, row + 1, 0);
    try term.write(theme_val.text_dim);
    var buf: [32]u8 = undefined;
    const size_str = std.fmt.bufPrint(&buf, "Current: {d}x{d}  [q] Quit", .{ width, height }) catch "?";
    try term.write(size_str);
    try term.write(theme_val.reset);
}

fn renderHelpOverlay(term: *tui.Terminal, theme_val: *const tui.Theme, width: u16, height: u16) !void {
    const chrome = style_adapter.gpu(theme_val);
    const box_width: u16 = @min(56, width - 4);
    const box_height: u16 = @min(@as(u16, 16), height - 2);
    const start_col = (width - box_width) / 2;
    const start_row = (height - box_height) / 2;

    try setCursorPosition(term, start_row, start_col);
    try term.write(chrome.frame);
    try term.write("\u{2554}");
    try writeRepeat(term, "\u{2550}", @as(usize, box_width) - 2);
    try term.write("\u{2557}");
    try term.write(theme_val.reset);

    const help_lines = [_][]const u8{
        "",
        "  ABI Brain Dashboard - Help",
        "",
        "  [Tab]         Switch Animation/Dashboard",
        "  [q] / [Esc]   Quit",
        "  [p]           Pause/Resume data updates",
        "  [t] / [T]     Cycle themes forward/back",
        "  [?]           Toggle this help overlay",
        "",
        "  Animation: 3D brain with data-driven nodes",
        "  Dashboard: WDBX stats + Abbey learning",
        "",
        "  --theme <name>  Set initial theme",
        "",
    };

    const max_lines = @min(help_lines.len, @as(usize, box_height - 2));
    for (help_lines[0..max_lines], 0..) |line, i| {
        try setCursorPosition(term, start_row + 1 + @as(u16, @intCast(i)), start_col);
        try term.write(chrome.frame);
        try term.write("\u{2551}");
        try term.write(theme_val.reset);
        try term.write(line);
        const pad = @as(usize, box_width) - 2 - line.len;
        try writeRepeat(term, " ", pad);
        try term.write(chrome.frame);
        try term.write("\u{2551}");
        try term.write(theme_val.reset);
    }

    try setCursorPosition(term, start_row + box_height - 1, start_col);
    try term.write(chrome.frame);
    try term.write("\u{255A}");
    try writeRepeat(term, "\u{2550}", @as(usize, box_width) - 2);
    try term.write("\u{255D}");
    try term.write(theme_val.reset);

    try setCursorPosition(term, start_row + box_height, start_col + 4);
    try term.write(theme_val.text_dim);
    try term.write("Press q, ?, Esc, or Enter to close");
    try term.write(theme_val.reset);
}

// ===============================================================================
// Utilities
// ===============================================================================

fn writeChip(
    term: *tui.Terminal,
    chrome: style_adapter.ChromeStyle,
    theme_val: *const tui.Theme,
    label: []const u8,
    value: []const u8,
) !void {
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" ");
    try term.write(label);
    try term.write(" ");
    try term.write(theme_val.reset);
    try term.write(" ");
    try term.write(value);
}

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
    try term.moveTo(row, col);
}

fn printFallbackInfo() void {
    utils.output.println("\nBrain Dashboard requires a terminal.", .{});
    utils.output.println("  Use 'abi system-info' for system status", .{});
    utils.output.println("  Use 'abi ui gpu' for GPU monitoring", .{});
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi ui brain [OPTIONS]
        \\
        \\Interactive brain visualization dashboard combining a 3D neural
        \\network animation with WDBX vector database and Abbey learning
        \\metrics in a dual-view TUI.
        \\
        \\Options:
        \\  --theme <name>       Set initial theme (e.g., nord, monokai)
        \\  --list-themes        Print available themes and exit
        \\  --db <path>          Use live WDBX database (requires -Denable-database)
        \\  --training <path>    Use training metrics JSONL file as data source
        \\  -h, --help           Show this help message
        \\
        \\Keyboard Controls:
        \\  Tab             Switch between Animation and Dashboard views
        \\  q, Esc          Quit
        \\  p               Pause/Resume data updates
        \\  t / T           Cycle themes forward/back
        \\  ?               Show help overlay
        \\
        \\Views:
        \\  Animation: 3D rotating brain with data-driven node activity
        \\  Dashboard: 6-panel grid showing WDBX stats, throughput,
        \\             similarity, learning phase, rewards, attention
        \\
    , .{});
}

test {
    std.testing.refAllDecls(@This());
}
