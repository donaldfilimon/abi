//! Shared DSL helpers for CLI-backed TUI dashboards.
//!
//! Unifies repetitive wiring across `abi ui <subcommand>` dashboard commands.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const session_runner = @import("../../commands/ui/session_runner.zig");
const theme_options = @import("../../commands/ui/theme_options.zig");

pub fn RunOptions(comptime PanelType: type) type {
    return struct {
        dashboard_name: []const u8,
        terminal_title: []const u8,
        title: []const u8,
        refresh_rate_ms: u32 = 250,
        min_width: u16 = 40,
        min_height: u16 = 10,
        help_keys: []const u8 = " [q]uit  [p]ause  [t]heme  [?]help",
        print_help: *const fn () void,
        init_panel: *const fn (
            allocator: std.mem.Allocator,
            terminal: *tui.Terminal,
            initial_theme: *const tui.Theme,
            remaining_args: []const [:0]const u8,
        ) anyerror!PanelType,
        validate_args: ?*const fn (remaining_args: []const [:0]const u8) anyerror!void = null,
        extra_key_handler: ?*const fn (*tui.dashboard.Dashboard(PanelType), tui.Key) bool = null,
    };
}

pub fn runSimpleDashboard(
    comptime PanelType: type,
    ctx: *const context_mod.CommandContext,
    args: []const [:0]const u8,
    options: RunOptions(PanelType),
) !void {
    const allocator = ctx.allocator;
    var parsed = try theme_options.parseThemeArgs(allocator, args);
    defer parsed.deinit();

    if (parsed.list_themes) {
        theme_options.printAvailableThemes();
        return;
    }
    if (parsed.wants_help) {
        options.print_help();
        return;
    }
    if (options.validate_args) |validate_args| {
        try validate_args(parsed.remaining_args);
    } else if (parsed.remaining_args.len > 0) {
        utils.output.printError("Unknown argument: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        return error.InvalidArgument;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;

    var session = session_runner.startSimpleDashboard(allocator, .{
        .dashboard_name = options.dashboard_name,
        .terminal_title = options.terminal_title,
    }) orelse return;
    defer session.deinit();

    const panel = try options.init_panel(allocator, &session.terminal, initial_theme, parsed.remaining_args);
    var dash = tui.dashboard.Dashboard(PanelType).init(allocator, &session.terminal, initial_theme, panel, .{
        .title = options.title,
        .refresh_rate_ms = options.refresh_rate_ms,
        .min_width = options.min_width,
        .min_height = options.min_height,
        .help_keys = options.help_keys,
    });
    if (options.extra_key_handler) |extra_key_handler| {
        dash.extra_key_handler = extra_key_handler;
    }
    defer dash.deinit();
    try dash.run();
}
