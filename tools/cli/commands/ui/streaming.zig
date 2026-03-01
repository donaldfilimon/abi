//! Streaming Inference Dashboard Command
//!
//! Interactive TUI dashboard for monitoring a streaming LLM server.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");
const dsl = @import("../../ui/dsl/mod.zig");

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try dsl.runSimpleDashboard(tui.StreamingDashboard, ctx, args, .{
        .dashboard_name = "Streaming Dashboard",
        .terminal_title = "ABI Streaming Dashboard",
        .title = "ABI STREAMING DASHBOARD",
        .refresh_rate_ms = 200,
        .min_width = 50,
        .min_height = 12,
        .print_help = printHelp,
        .init_panel = initPanel,
        .validate_args = validateArgs,
    });
}

fn validateArgs(remaining_args: []const [:0]const u8) !void {
    if (remaining_args.len <= 1) return;
    utils.output.printError("Expected at most one endpoint argument for ui streaming.", .{});
    return error.InvalidArgument;
}

fn initPanel(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    initial_theme: *const tui.Theme,
    remaining_args: []const [:0]const u8,
) !tui.StreamingDashboard {
    var endpoint: []const u8 = "http://127.0.0.1:8080";
    if (remaining_args.len > 0) {
        endpoint = std.mem.sliceTo(remaining_args[0], 0);
    }
    return tui.StreamingDashboard.init(allocator, terminal, initial_theme, endpoint);
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
