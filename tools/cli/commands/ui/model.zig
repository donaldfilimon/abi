//! Model Management Dashboard
//!
//! Interactive TUI dashboard for managing AI models.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");

const Dash = tui.dashboard.Dashboard(tui.ModelManagementPanel);

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

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runDashboard(allocator, initial_theme);
}

fn runDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme) !void {
    if (!tui.Terminal.isSupported()) {
        utils.output.printError("Model Dashboard requires a terminal.", .{});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();
    terminal.enter() catch |err| {
        utils.output.printError("Failed to start Model Dashboard: {t}", .{err});
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Model Manager") catch {};

    const panel = tui.ModelManagementPanel.init(allocator, &terminal, initial_theme);
    var dash = Dash.init(allocator, &terminal, initial_theme, panel, .{
        .title = "ABI MODEL MANAGER",
        .refresh_rate_ms = 500,
        .help_keys = " [q]uit  [\xe2\x86\x91\xe2\x86\x93]navigate  [enter]select  [t]heme  [?]help",
    });
    dash.extra_key_handler = &handleModelKeys;
    defer dash.deinit();
    try dash.run();
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
