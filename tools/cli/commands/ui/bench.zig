//! Benchmark Dashboard Command
//!
//! Interactive TUI dashboard for viewing benchmark suite status and throughput.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const theme_options = @import("theme_options.zig");

const Dash = tui.dashboard.Dashboard(tui.BenchmarkPanel);

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
        utils.output.printError("Unknown argument for ui bench: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        return error.InvalidArgument;
    }

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runDashboard(allocator, initial_theme);
}

fn runDashboard(allocator: std.mem.Allocator, initial_theme: *const tui.Theme) !void {
    if (!tui.Terminal.isSupported()) {
        utils.output.printError("Benchmark Dashboard requires a terminal.", .{});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();
    terminal.enter() catch |err| {
        utils.output.printError("Failed to start Benchmark Dashboard: {t}", .{err});
        return;
    };
    defer terminal.exit() catch {};
    terminal.setTitle("ABI Benchmark Dashboard") catch {};

    const panel = tui.BenchmarkPanel.init(allocator, &terminal, initial_theme);
    var dash = Dash.init(allocator, &terminal, initial_theme, panel, .{
        .title = "ABI BENCHMARK DASHBOARD",
        .refresh_rate_ms = 300,
        .help_keys = " [q]uit  [p]ause  [up/down]select  [t]heme  [?]help",
    });
    dash.extra_key_handler = &handleBenchKeys;
    defer dash.deinit();
    try dash.run();
}

fn handleBenchKeys(self: *Dash, key: tui.Key) bool {
    switch (key.code) {
        .up => self.panel.moveUp(),
        .down => self.panel.moveDown(),
        else => {},
    }
    return false;
}

fn printHelp() void {
    const help =
        \\Usage: abi ui bench [options]
        \\
        \\Interactive benchmark dashboard.
        \\
        \\Options:
        \\  --theme <name>     Set initial theme
        \\  --list-themes      Print available themes
        \\  --help             Show this help
        \\
        \\Keys:
        \\  q / Esc            Quit
        \\  p                  Pause/resume refresh
        \\  Up/Down            Select benchmark suite
        \\  t / T              Cycle themes
        \\
    ;
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
