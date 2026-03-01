//! Database Dashboard Command
//!
//! Interactive TUI dashboard for database monitoring.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const dsl = @import("../../ui/dsl/mod.zig");

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(tui.DatabasePanel, ctx, args, .{
        .dashboard_name = "Database Dashboard",
        .terminal_title = "ABI Database Dashboard",
        .title = "ABI DATABASE DASHBOARD",
        .print_help = printHelp,
        .init_panel = initPanel,
    });
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    _: []const [:0]const u8,
) !tui.DatabasePanel {
    return tui.DatabasePanel.init(allocator, terminal, initial_theme);
}

fn printHelp() void {
    const help =
        \\Usage: abi ui db [options]
        \\
        \\Interactive database monitoring dashboard.
        \\
        \\Options:
        \\  --theme <name>     Set initial theme
        \\  --list-themes      Print available themes
        \\  --help             Show this help
        \\
        \\Keys:
        \\  q / Esc            Quit
        \\  p                  Pause/resume refresh
        \\  t / T              Cycle themes
        \\
    ;
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
