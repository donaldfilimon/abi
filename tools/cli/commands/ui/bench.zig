//! Benchmark Dashboard Command
//!
//! Interactive TUI dashboard for viewing benchmark suite status and throughput.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const dsl = @import("../../ui/dsl/mod.zig");

const PanelType = tui.BenchmarkPanel;
const Dash = tui.dashboard.Dashboard(PanelType);

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(PanelType, ctx, args, .{
        .dashboard_name = "Benchmark Dashboard",
        .terminal_title = "ABI Benchmark Dashboard",
        .title = "ABI BENCHMARK DASHBOARD",
        .refresh_rate_ms = 300,
        .help_keys = " [q]uit  [p]ause  [up/down]select  [t]heme  [?]help",
        .print_help = printHelp,
        .init_panel = initPanel,
        .validate_args = validateArgs,
        .extra_key_handler = handleBenchKeys,
    });
}

fn validateArgs(remaining_args: []const [:0]const u8) !void {
    if (remaining_args.len == 0) return;
    utils.output.printError("Unknown argument for ui bench: {s}", .{remaining_args[0]});
    @import("theme_options.zig").printThemeHint();
    return error.InvalidArgument;
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    _: []const [:0]const u8,
) !PanelType {
    return tui.BenchmarkPanel.init(allocator, terminal, initial_theme);
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
