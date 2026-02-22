//! Streaming Inference Dashboard Command
//!
//! Interactive TUI dashboard for monitoring a streaming LLM server.
//! Uses AsyncLoop for timer-driven refresh.

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");

const DashboardState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme_manager: tui.ThemeManager,
    dashboard: tui.StreamingDashboard,
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
        endpoint: []const u8,
    ) !DashboardState {
        var theme_manager = tui.ThemeManager.init();
        theme_manager.current = initial_theme;
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme_manager = theme_manager,
            .dashboard = try tui.StreamingDashboard.init(allocator, terminal, theme_manager.current, endpoint),
            .term_size = terminal.size(),
            .paused = false,
            .show_help = false,
            .frame_count = 0,
            .notification = null,
            .notification_time = 0,
        };
    }

    pub fn deinit(self: *DashboardState) void {
        self.dashboard.deinit();
    }

    pub fn theme(self: *const DashboardState) *const tui.Theme {
        return self.theme_manager.current;
    }

    pub fn showNotification(self: *DashboardState, msg: []const u8) void {
        self.notification = msg;
        self.notification_time = abi.shared.utils.unixMs();
    }

    pub fn clearExpiredNotification(self: *DashboardState) void {
        if (self.notification_time > 0) {
            const elapsed = abi.shared.utils.unixMs() - self.notification_time;
            if (elapsed > 3000) {
                self.notification = null;
                self.notification_time = 0;
            }
        }
    }

    pub fn updateTheme(self: *DashboardState) void {
        self.dashboard.theme = self.theme_manager.current;
    }
};

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

    // Parse endpoint from remaining args or use default
    var endpoint: []const u8 = "http://127.0.0.1:8080";
    for (parsed.remaining_args) |arg| {
        const a = std.mem.sliceTo(arg, 0);
        if (std.mem.startsWith(u8, a, "http")) {
            endpoint = a;
        }
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runDashboard(allocator, initial_theme, endpoint);
}

fn runDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme, endpoint: []const u8) !void {
    if (!tui.Terminal.isSupported()) {
        utils.output.printError("Streaming Dashboard requires a terminal.", .{});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        utils.output.printError("Failed to start Streaming Dashboard: {t}", .{err});
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Streaming Dashboard") catch {};

    var state = try DashboardState.init(allocator, &terminal, initial_theme, endpoint);
    defer state.deinit();

    var loop = tui.AsyncLoop.init(allocator, &terminal, .{
        .refresh_rate_ms = 200, // 5 FPS
        .input_poll_ms = 16,
        .auto_resize = true,
    });
    defer loop.deinit();

    loop.setRenderCallback(&dashboardRender);
    loop.setTickCallback(&dashboardTick);
    loop.setUpdateCallback(&dashboardUpdate);
    loop.setUserData(@ptrCast(&state));

    try loop.run();
}

fn dashboardRender(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse return error.UserDataNotSet;
    try state.terminal.clear();
    try renderDashboard(state);
    state.frame_count = loop.getFrameCount();
}

fn dashboardTick(loop: *tui.AsyncLoop) anyerror!void {
    const state = loop.getUserData(DashboardState) orelse return error.UserDataNotSet;
    state.term_size = state.terminal.size();
    state.clearExpiredNotification();
    if (!state.paused) {
        try state.dashboard.pollMetrics();
    }
}

fn dashboardUpdate(loop: *tui.AsyncLoop, event: tui.AsyncEvent) anyerror!bool {
    const state = loop.getUserData(DashboardState) orelse return error.UserDataNotSet;
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

fn handleKeyEvent(state: *DashboardState, key: tui.Key) bool {
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
        .ctrl_c, .escape => return true,
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
                    'h', '?' => state.show_help = true,
                    else => {},
                }
            }
        },
        else => {},
    }
    return false;
}

fn renderDashboard(state: *DashboardState) !void {
    const term = state.terminal;
    const theme_val = state.theme();
    const width = state.term_size.cols;
    const height = state.term_size.rows;

    if (width < 50 or height < 12) {
        try term.moveTo(height / 2, 0);
        try term.write(theme_val.warning);
        try term.write("Resize terminal to at least 50x12");
        try term.write(theme_val.reset);
        return;
    }

    // Title bar
    try term.moveTo(0, 0);
    try term.write(theme_val.bold);
    try term.write(theme_val.primary);
    try term.write(" ABI STREAMING DASHBOARD ");
    try term.write(theme_val.reset);

    // Main dashboard
    try state.dashboard.render(2, 0, width, height -| 4);

    // Notification
    if (state.notification) |msg| {
        try term.moveTo(height -| 2, 2);
        try term.write(theme_val.info);
        try term.write(msg);
        try term.write(theme_val.reset);
    }

    // Help bar
    try term.moveTo(height -| 1, 0);
    try term.write(theme_val.text_dim);
    try term.write(" [q]uit  [p]ause  [t]heme  [?]help");
    try term.write(theme_val.reset);
}

fn printHelp() void {
    const help =
        \\Usage: abi ui streaming [endpoint] [options]
        \\
        \\Monitor a streaming LLM inference server in real time.
        \\
        \\Arguments:
        \\  endpoint           Server URL (default: http://127.0.0.1:8080)
        \\
        \\Options:
        \\  --theme <name>     Set initial theme
        \\  --list-themes      Print available themes
        \\  --help             Show this help
        \\
        \\Keys:
        \\  q / Esc            Quit
        \\  p                  Pause/resume polling
        \\  t / T              Cycle themes
        \\
        \\Examples:
        \\  abi ui streaming
        \\  abi ui streaming http://192.168.1.100:8080
        \\
    ;
    std.debug.print("{s}", .{help});
}
