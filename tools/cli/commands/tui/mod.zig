//! Interactive TUI command launcher.
//!
//! Provides a terminal-based interface for selecting and running ABI CLI commands.
//! Features: categories, search/filter, quick launch (1-9), mouse support, status bar.
//!
//! This is the thin orchestrator; domain logic lives in sibling modules:
//!   types.zig, menu.zig, completion.zig, state.zig, render.zig, input.zig

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");

const types = @import("types.zig");
const menu = @import("menu.zig");
const completion = @import("completion.zig");
const state_mod = @import("state.zig");
const render = @import("render.zig");
const input = @import("input.zig");
const tui_layout = @import("layout.zig");
const theme_options = @import("../ui/theme_options.zig");

pub const meta: command_mod.Meta = .{
    .name = "tui",
    .description = "Launch interactive TUI command menu",
    .forward = .{
        .target = "ui",
        .prepend_args = &[_][:0]const u8{"launch"},
        .warning = "'abi tui' is deprecated; use 'abi ui launch'.",
    },
};

const TuiState = state_mod.TuiState;

// ═══════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════

/// Entry point for the TUI command.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;
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
        utils.output.printError("Unknown launcher argument: {s}", .{parsed.remaining_args[0]});
        theme_options.printThemeHint();
        utils.output.printInfo("Run 'abi ui launch --help' for launcher usage.", .{});
        return error.InvalidArgument;
    }

    const fw_config = abi.Config.defaults();
    var framework = try abi.Framework.initWithIo(allocator, fw_config, io);
    defer framework.deinit();

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runInteractive(allocator, &framework, initial_theme);
}

fn runInteractive(
    allocator: std.mem.Allocator,
    framework: *abi.Framework,
    initial_theme: *const tui.Theme,
) !void {
    // Check platform support before initializing
    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("TUI is not supported on {s}.", .{caps.platform_name});
        utils.output.printInfo("This platform lacks terminal control capabilities required for the interactive UI.", .{});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    // Attempt to enter the TUI.
    terminal.enter() catch |err| {
        utils.output.printError("Failed to start interactive TUI: {t}", .{err});
        utils.output.printInfo("Falling back to command list display.", .{});
        std.debug.print("\nAvailable commands (run individually):\n", .{});
        std.debug.print("  abi llm providers               - Show provider availability\n", .{});
        std.debug.print("  abi llm run --model llama3 --prompt \"hi\" --backend ollama --strict-backend\n", .{});
        std.debug.print("  abi bench all                   - Run all benchmarks\n", .{});
        std.debug.print("  abi system-info                 - Show system information\n", .{});
        std.debug.print("  abi config show                 - Show current configuration\n", .{});
        std.debug.print("  abi db stats                    - Show database statistics\n", .{});
        std.debug.print("  abi gpu backends                - List GPU backends\n", .{});
        std.debug.print("  abi task list                   - List tasks\n", .{});
        std.debug.print("  abi --list-features             - Show available features\n", .{});
        return;
    };
    defer terminal.exit() catch {};

    var state = try TuiState.init(allocator, &terminal, framework, initial_theme);
    defer state.deinit();

    // Shared async loop aligns launcher behavior with dashboards/monitors.
    var loop = tui.AsyncLoop.init(allocator, &terminal, .{
        .refresh_rate_ms = 120,
        .input_poll_ms = 16,
        .auto_resize = true,
        .max_events_per_frame = 32,
    });
    defer loop.deinit();

    state.term_size = terminal.size();
    state.visible_rows = tui_layout.computeVisibleRows(state.term_size.rows);

    loop.setUserData(&state);
    loop.setRenderCallback(launcherRender);
    loop.setUpdateCallback(launcherUpdate);
    try loop.run();
}

fn launcherRender(loop: *tui.AsyncLoop) !void {
    const state = loop.getUserData(TuiState) orelse return;
    state.term_size = state.terminal.size();
    state.visible_rows = tui_layout.computeVisibleRows(state.term_size.rows);
    try state.terminal.clear();
    try render.renderFrame(state);
}

fn launcherUpdate(loop: *tui.AsyncLoop, event: tui.AsyncEvent) !bool {
    const state = loop.getUserData(TuiState) orelse return false;

    switch (event) {
        .resize => |size| {
            state.term_size.rows = size.rows;
            state.term_size.cols = size.cols;
            state.visible_rows = tui_layout.computeVisibleRows(size.rows);
            return false;
        },
        .input => |input_event| switch (input_event) {
            .key => |key| return input.handleKeyEvent(state, key),
            .mouse => |mouse| {
                if (mouse.pressed and mouse.button == .left) {
                    if (state.handleMouseClick(mouse.row, tui_layout.menuStartRow(state))) {
                        if (state.selectedItem()) |item| {
                            return input.executeActionFromState(state, item.action);
                        }
                    }
                } else if (mouse.button == .wheel_up) {
                    state.moveUp();
                } else if (mouse.button == .wheel_down) {
                    state.moveDown();
                }
                return false;
            },
        },
        else => return false,
    }
}

pub fn printHelp() void {
    input.printHelp();
    std.debug.print(
        \\
        \\Launcher options:
        \\  --theme <name>     Set initial theme (exact lowercase name)
        \\  --list-themes      Print available themes and exit
        \\
    , .{});
}

// ═══════════════════════════════════════════════════════════════════
// Test discovery — pull in sub-module tests
// ═══════════════════════════════════════════════════════════════════
test {
    _ = @import("types.zig");
    _ = @import("menu.zig");
    _ = @import("completion.zig");
    _ = @import("state.zig");
    _ = @import("layout.zig");
}
