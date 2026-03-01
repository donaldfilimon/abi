//! Model Management Dashboard
//!
//! Interactive TUI dashboard for managing AI models.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const dsl = @import("../../ui/dsl/mod.zig");

const PanelType = tui.ModelManagementPanel;
const Dash = tui.dashboard.Dashboard(PanelType);

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(PanelType, ctx, args, .{
        .dashboard_name = "Model Dashboard",
        .terminal_title = "ABI Model Manager",
        .title = "ABI MODEL MANAGER",
        .refresh_rate_ms = 500,
        .help_keys = " [q]uit  [↑↓]navigate  [enter]select  [t]heme  [?]help",
        .print_help = printHelp,
        .init_panel = initPanel,
        .extra_key_handler = handleModelKeys,
    });
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    _: []const [:0]const u8,
) !PanelType {
    return tui.ModelManagementPanel.init(allocator, terminal, initial_theme);
}

fn handleModelKeys(self: *Dash, key: tui.Key) bool {
    switch (key.code) {
        .up => self.panel.moveUp(),
        .down => self.panel.moveDown(),
        .enter => {
            if (self.panel.getSelectedModel()) |model| {
                self.panel.setActiveModel(model.id);
                self.showNotification("Active model set");
            }
        },
        else => {},
    }
    return false;
}

fn printHelp() void {
    const help =
        \\Usage: abi ui model [options]
        \\
        \\Interactive model management dashboard.
        \\
        \\Options:
        \\  --theme <name>     Set initial theme
        \\  --list-themes      Print available themes
        \\  --help             Show this help
        \\
        \\Keys:
        \\  q / Esc            Quit
        \\  Up/Down            Navigate models
        \\  Enter              Select/activate model
        \\  t / T              Cycle themes
        \\  p                  Pause/resume refresh
        \\
    ;
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
