//! Database Dashboard Command
//!
//! Interactive TUI dashboard for database monitoring.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const session_runner = @import("session_runner.zig");
const theme_options = @import("theme_options.zig");

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
        utils.output.printError("Unknown argument for ui db: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        return error.InvalidArgument;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runDashboard(allocator, initial_theme);
}

fn runDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme) !void {
    var session = session_runner.startSimpleDashboard(allocator, .{
        .dashboard_name = "Database Dashboard",
        .terminal_title = "ABI Database Dashboard",
    }) orelse return;
    defer session.deinit();

    const panel = tui.DatabasePanel.init(allocator, &session.terminal, initial_theme);
    var dash = tui.dashboard.Dashboard(tui.DatabasePanel).init(allocator, &session.terminal, initial_theme, panel, .{
        .title = "ABI DATABASE DASHBOARD",
    });
    defer dash.deinit();
    try dash.run();
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
