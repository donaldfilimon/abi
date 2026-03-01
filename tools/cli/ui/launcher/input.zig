//! TUI input handling.
//!
//! Key event dispatch for normal, search, and preview modes.
//! Also contains action execution (run commands, exit TUI).

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const framework_mod = @import("../../framework/mod.zig");
const tui = @import("../core/mod.zig");
const utils = @import("../../utils/mod.zig");
const commands_mod = @import("../../mod.zig");
const theme_options = @import("theme_options.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const menu_mod = @import("menu.zig");
const render = @import("render.zig");
const tui_layout = @import("layout.zig");

const TuiState = state_mod.TuiState;
const Action = types.Action;
const CommandRef = types.CommandRef;
const colors = types.colors;

// ═══════════════════════════════════════════════════════════════════
// Key Event Dispatch
// ═══════════════════════════════════════════════════════════════════

pub fn handleKeyEvent(state: *TuiState, key: tui.Key) !bool {
    // Clear expired notifications
    state.clearExpiredNotification();

    // Handle preview mode
    if (state.preview_mode) {
        return try handlePreviewKey(state, key);
    }

    if (state.search_mode) {
        return try handleSearchKey(state, key);
    }

    switch (key.code) {
        .ctrl_c => return true,
        .escape => {
            if (state.show_history) {
                state.show_history = false;
            } else if (state.search_len > 0) {
                state.search_len = 0;
                state.completion_state.clear();
                try state.resetFilter();
            }
        },
        .character => {
            if (key.char) |ch| {
                switch (ch) {
                    'q' => return true,
                    'j' => state.moveDown(),
                    'k' => state.moveUp(),
                    '/' => {
                        state.search_mode = true;
                    },
                    'g' => state.goHome(),
                    'G' => state.goEnd(),
                    't' => {
                        // Cycle theme
                        state.theme_manager.nextTheme();
                        state.showNotification(theme_options.themeNotificationMessage(state.theme_manager.current.name), .info);
                    },
                    'T' => {
                        // Cycle theme backwards
                        state.theme_manager.prevTheme();
                        state.showNotification(theme_options.themeNotificationMessage(state.theme_manager.current.name), .info);
                    },
                    '?' => {
                        // Show preview for selected item
                        if (state.selectedItem() != null) {
                            state.preview_mode = true;
                        }
                    },
                    'h' => {
                        // Toggle history view
                        state.show_history = !state.show_history;
                    },
                    '1'...'9' => {
                        // Quick launch
                        const num = ch - '0';
                        if (menu_mod.findByShortcut(state.items, num)) |idx| {
                            const item = state.items[idx];
                            if (try executeActionFromState(state, item.action)) return true;
                        }
                    },
                    else => {},
                }
            }
        },
        .up => state.moveUp(),
        .down => state.moveDown(),
        .page_up => state.pageUp(),
        .page_down => state.pageDown(),
        .home => state.goHome(),
        .end => state.goEnd(),
        .enter => {
            if (state.selectedItem()) |item| {
                if (try executeActionFromState(state, item.action)) return true;
            }
        },
        else => {},
    }
    return false;
}

fn handlePreviewKey(state: *TuiState, key: tui.Key) !bool {
    switch (key.code) {
        .ctrl_c => return true,
        .escape => {
            state.preview_mode = false;
        },
        .enter => {
            // Run the previewed command
            state.preview_mode = false;
            if (state.selectedItem()) |item| {
                if (try executeActionFromState(state, item.action)) return true;
            }
        },
        else => {},
    }
    return false;
}

fn handleSearchKey(state: *TuiState, key: tui.Key) !bool {
    switch (key.code) {
        .ctrl_c => return true,
        .escape => {
            state.search_mode = false;
            state.completion_state.clear();
        },
        .tab => {
            // Tab completion
            if (state.completion_state.active) {
                state.nextCompletion();
            } else {
                try state.updateCompletions();
            }
        },
        .enter => {
            // Accept completion if active, otherwise execute selected
            if (state.completion_state.active) {
                try state.acceptCompletion();
                return false; // Stay in search mode
            }

            state.search_mode = false;
            // Execute selected if any
            if (state.selectedItem()) |item| {
                if (try executeActionFromState(state, item.action)) return true;
            }
        },
        .backspace => {
            if (state.search_len > 0) {
                state.search_len -= 1;
                try state.applyFilter();
                try state.updateCompletions();
            }
        },
        .character => {
            if (key.char) |ch| {
                if (state.search_len < state.search_buffer.len) {
                    state.search_buffer[state.search_len] = ch;
                    state.search_len += 1;
                    try state.applyFilter();
                    try state.updateCompletions();
                }
            }
        },
        .up => {
            // Navigate completions with arrow keys
            if (state.completion_state.active) {
                state.prevCompletion();
            } else {
                state.moveUp();
            }
        },
        .down => {
            // Navigate completions with arrow keys
            if (state.completion_state.active) {
                state.nextCompletion();
            } else {
                state.moveDown();
            }
        },
        else => {},
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════
// Action Execution
// ═══════════════════════════════════════════════════════════════════

pub fn executeActionFromState(state: *TuiState, action: Action) !bool {
    if (action == .quit) return true;

    switch (action) {
        .command => |cmd| try state.addToHistory(cmd.id),
        else => {},
    }

    try state.terminal.exit();
    errdefer state.terminal.enter() catch {};

    runAction(state.allocator, state.framework, action) catch {};

    utils.output.printInfo("\nPress Enter to return to menu...", .{});
    _ = state.terminal.readKey() catch {};
    try state.terminal.enter();

    return false;
}

fn runAction(allocator: std.mem.Allocator, framework: *abi.App, action: Action) !void {
    _ = framework;

    switch (action) {
        .command => |cmd| runCommand(allocator, cmd) catch |err| {
            utils.output.printError("Command '{s}' failed: {t}", .{ commandLabel(cmd), err });
            return err;
        },
        .version => utils.output.printInfo("ABI Framework v{s}", .{abi.version()}),
        .help => printHelp(),
        .quit => {},
    }
}

fn commandLabel(cmd: CommandRef) []const u8 {
    if (std.mem.eql(u8, cmd.command, "train") and cmd.args.len > 0) {
        const first = std.mem.sliceTo(cmd.args[0], 0);
        if (std.mem.eql(u8, first, "monitor")) return "train monitor";
    }
    return cmd.id;
}

fn runCommand(allocator: std.mem.Allocator, cmd: CommandRef) !void {
    var io_backend = utils.io_backend.initIoBackend(allocator);
    defer io_backend.deinit();

    const cmd_ctx = context_mod.CommandContext{
        .allocator = allocator,
        .io = io_backend.io(),
    };

    const command_name = framework_mod.completion.resolveAlias(&commands_mod.descriptors, cmd.command);
    const matched = try framework_mod.router.runCommand(cmd_ctx, &commands_mod.descriptors, command_name, cmd.args);
    if (!matched) {
        return framework_mod.errors.Error.UnknownCommand;
    }
}

pub fn printHelp() void {
    std.debug.print(
        \\{s}Usage:{s} abi ui launch
        \\
        \\Launch an interactive terminal UI to select and run ABI commands.
        \\
        \\{s}Navigation:{s}
        \\  Up/Down, j/k    Navigate the menu
        \\  PgUp/PgDn       Page navigation
        \\  Home/End, g/G   Jump to first/last
        \\  Mouse wheel     Scroll the menu
        \\  Mouse click     Select item
        \\
        \\{s}Actions:{s}
        \\  Enter           Run the selected command
        \\  1-9             Quick launch (numbered items)
        \\  /               Search/filter commands
        \\  Tab             Autocomplete in search mode
        \\  Esc             Clear search / Exit modes
        \\  q, Ctrl+C       Exit the TUI launcher
        \\
        \\{s}Features:{s}
        \\  ?               Preview command details before running
        \\  t / T           Cycle through color themes (forward/backward)
        \\  h               Toggle command history panel
        \\
        \\{s}Themes:{s}
        \\  default, monokai, solarized, nord, gruvbox, high_contrast, minimal
        \\
    , .{
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
        colors.bold,
        colors.reset,
    });
}

test {
    std.testing.refAllDecls(@This());
}
