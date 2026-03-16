//! Chat Dashboard Command
//!
//! Interactive TUI chat panel with multi-persona support.

const std = @import("std");
const command = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const tui = @import("../../../terminal/mod.zig");
const utils = @import("../../../utils/mod.zig");
const dsl = @import("../../../terminal/dsl/mod.zig");
const chat_panel = @import("../../../terminal/panels/chat_panel.zig");

pub const meta: command.Meta = .{
    .name = "chat",
    .description = "Interactive TUI chat panel with multi-persona support",
};

const PanelType = ChatDashPanel;
const Dash = tui.dashboard.Dashboard(PanelType);

const ChatDashPanel = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    theme: *const tui.Theme,
    inner: ?chat_panel.ChatPanel,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        theme: *const tui.Theme,
    ) ChatDashPanel {
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .theme = theme,
            .inner = chat_panel.ChatPanel.init(allocator, terminal, theme) catch null,
        };
    }

    pub fn deinit(self: *ChatDashPanel) void {
        if (self.inner) |*inner| {
            inner.deinit();
        }
    }

    pub fn update(self: *ChatDashPanel) !void {
        if (self.inner) |*inner| {
            try inner.update();
        }
    }

    pub fn render(
        self: *ChatDashPanel,
        start_row: usize,
        start_col: usize,
        width: usize,
        height: usize,
    ) !void {
        if (self.inner) |*inner| {
            inner.theme = self.theme;
            inner.term = self.terminal;
            try inner.render(start_row, start_col, width, height);
        } else {
            const row: u16 = @intCast(start_row);
            const col: u16 = @intCast(start_col);
            try self.terminal.moveTo(row, col);
            try self.terminal.write(self.theme.warning);
            try self.terminal.write("Chat panel failed to initialize");
            try self.terminal.write(self.theme.reset);
        }
    }
};

fn handleChatKeys(dash: *Dash, key: tui.Key) bool {
    if (dash.panel.inner) |*inner| {
        return inner.handleEvent(.{ .key = key }) catch false;
    }
    return false;
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    _: []const [:0]const u8,
) !PanelType {
    return ChatDashPanel.init(allocator, terminal, initial_theme);
}

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(PanelType, ctx, args, .{
        .dashboard_name = "Chat Dashboard",
        .terminal_title = "ABI Chat",
        .title = "ABI CHAT",
        .refresh_rate_ms = 100,
        .min_width = 50,
        .min_height = 16,
        .help_keys = " [q]uit  [Tab]persona  [Enter]send  [p]ause  [t]heme  [?]help",
        .print_help = printHelp,
        .init_panel = initPanel,
        .extra_key_handler = handleChatKeys,
    });
}

fn printHelp() void {
    const help =
        \\Usage: abi ui chat [options]
        \\
        \\Interactive multi-persona chat dashboard.
        \\
        \\Options:
        \\  --theme <name>     Set initial theme
        \\  --list-themes      Print available themes
        \\  --help             Show this help
        \\
        \\Keys:
        \\  q / Esc            Quit
        \\  Tab                Switch persona
        \\  Enter              Send message
        \\  t / T              Cycle themes
        \\  p                  Pause/resume
        \\
    ;
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
