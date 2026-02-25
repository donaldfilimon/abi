//! Streaming Inference Dashboard Command
//!
//! Interactive TUI dashboard for monitoring a streaming LLM server.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
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

    const panel = try tui.StreamingDashboard.init(allocator, &terminal, initial_theme, endpoint);
    var dash = tui.dashboard.Dashboard(tui.StreamingDashboard).init(allocator, &terminal, initial_theme, panel, .{
        .title = "ABI STREAMING DASHBOARD",
        .refresh_rate_ms = 200,
        .min_width = 50,
        .min_height = 12,
    });
    defer dash.deinit();
    try dash.run();
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
    utils.output.print("{s}", .{help});
}

test {
    std.testing.refAllDecls(@This());
}
