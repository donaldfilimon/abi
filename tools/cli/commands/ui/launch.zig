//! Interactive TUI command launcher.
//!
//! Provides a terminal-based interface for selecting and running ABI CLI commands.
//! Features: categories, search/filter, quick launch (1-9), mouse support, status bar.
//!
//! This is the thin orchestrator; domain logic lives in sibling modules:
//!   types.zig, menu.zig, completion.zig, state.zig, render.zig, input.zig

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const tui = @import("../../tui/mod.zig");
const utils = @import("../../utils/mod.zig");

const types = @import("../tui/types.zig");
const menu = @import("../tui/menu.zig");
const completion = @import("../tui/completion.zig");
const state_mod = @import("../tui/state.zig");
const render = @import("../tui/render.zig");
const input = @import("../tui/input.zig");
const tui_layout = @import("../tui/layout.zig");
const theme_options = @import("theme_options.zig");

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
    var framework = try abi.App.initWithIo(allocator, fw_config, io);
    defer framework.deinit();

    const initial_theme = parsed.initial_theme orelse &tui.themes.themes.default;
    try runInteractive(allocator, &framework, initial_theme);
}

fn runInteractive(
    allocator: std.mem.Allocator,
    framework: *abi.App,
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
        utils.output.println("\nAvailable commands (run individually):", .{});
        utils.output.println("  abi llm providers               - Show provider availability", .{});
        utils.output.println("  abi llm run --model llama3 --prompt \"hi\" --backend ollama --strict-backend", .{});
        utils.output.println("  abi bench all                   - Run all benchmarks", .{});
        utils.output.println("  abi system-info                 - Show system information", .{});
        utils.output.println("  abi config show                 - Show current configuration", .{});
        utils.output.println("  abi db stats                    - Show database statistics", .{});
        utils.output.println("  abi gpu backends                - List GPU backends", .{});
        utils.output.println("  abi task list                   - List tasks", .{});
        utils.output.println("  abi --list-features             - Show available features", .{});
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
    utils.output.print(
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
    _ = @import("../tui/types.zig");
    _ = @import("../tui/menu.zig");
    _ = @import("../tui/completion.zig");
    _ = @import("../tui/state.zig");
    _ = @import("../tui/layout.zig");
}

test {
    std.testing.refAllDecls(@This());
}
