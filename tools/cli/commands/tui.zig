//! Interactive TUI command launcher.
//!
//! Provides a terminal-based interface for selecting and running ABI CLI commands.
//! Features: categories, search/filter, quick launch (1-9), mouse support, status bar.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const tui = @import("../tui/mod.zig");
const utils = @import("../utils/mod.zig");

const agent = @import("agent.zig");
const bench = @import("bench.zig");
const config = @import("config.zig");
const db = @import("db.zig");
const discord = @import("discord.zig");
const embed = @import("embed.zig");
const explore = @import("explore.zig");
const gpu = @import("gpu.zig");
const llm = @import("llm.zig");
const network = @import("network.zig");
const simd = @import("simd.zig");
const system_info = @import("system_info.zig");
const train = @import("train.zig");
const task = @import("task.zig");

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

const Category = enum {
    ai,
    data,
    system,
    tools,
    meta,

    fn icon(self: Category) []const u8 {
        return switch (self) {
            .ai => "🤖",
            .data => "💾",
            .system => "⚙️",
            .tools => "🔧",
            .meta => "📋",
        };
    }

    fn name(self: Category) []const u8 {
        return switch (self) {
            .ai => "AI & ML",
            .data => "Data",
            .system => "System",
            .tools => "Tools",
            .meta => "Meta",
        };
    }

    fn color(self: Category) []const u8 {
        return switch (self) {
            .ai => "\x1b[35m", // Magenta
            .data => "\x1b[34m", // Blue
            .system => "\x1b[33m", // Yellow
            .tools => "\x1b[32m", // Green
            .meta => "\x1b[36m", // Cyan
        };
    }
};

const Action = union(enum) {
    command: Command,
    version: void,
    help: void,
    quit: void,
};

const Command = enum {
    db,
    agent,
    bench,
    config,
    discord,
    embed,
    explore,
    gpu,
    llm,
    network,
    simd,
    system_info,
    train,
    task,
};

const MenuItem = struct {
    label: []const u8,
    description: []const u8,
    action: Action,
    category: Category,
    shortcut: ?u8 = null, // Quick launch key (1-9)
    usage: []const u8 = "", // Usage string for preview
    examples: []const []const u8 = &[_][]const u8{}, // Example commands
    related: []const []const u8 = &[_][]const u8{}, // Related commands

    fn categoryColor(self: *const MenuItem, theme: *const tui.Theme) []const u8 {
        return switch (self.category) {
            .ai => theme.category_ai,
            .data => theme.category_data,
            .system => theme.category_system,
            .tools => theme.category_tools,
            .meta => theme.category_meta,
        };
    }
};

/// Command history entry
const HistoryEntry = struct {
    command: Command,
    timestamp: i64,
};

const TuiState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    framework: *abi.Framework,
    items: []const MenuItem,
    filtered_indices: std.ArrayListUnmanaged(usize),
    selected: usize,
    scroll_offset: usize,
    search_mode: bool,
    search_buffer: [64]u8,
    search_len: usize,
    visible_rows: usize,
    term_size: tui.TerminalSize,
    // New features
    theme_manager: tui.ThemeManager,
    preview_mode: bool,
    history: std.ArrayListUnmanaged(HistoryEntry),
    show_history: bool,
    notification: ?[]const u8,
    notification_level: tui.Toast.Level,
    notification_time: i64,

    fn init(allocator: std.mem.Allocator, terminal: *tui.Terminal, framework: *abi.Framework) !TuiState {
        var state = TuiState{
            .allocator = allocator,
            .terminal = terminal,
            .framework = framework,
            .items = menuItemsExtended(),
            .filtered_indices = .empty,
            .selected = 0,
            .scroll_offset = 0,
            .search_mode = false,
            .search_buffer = undefined,
            .search_len = 0,
            .visible_rows = 10,
            .term_size = terminal.size(),
            .theme_manager = tui.ThemeManager.init(),
            .preview_mode = false,
            .history = .empty,
            .show_history = false,
            .notification = null,
            .notification_level = .info,
            .notification_time = 0,
        };
        // Initialize with all items
        try state.resetFilter();
        return state;
    }

    fn deinit(self: *TuiState) void {
        self.filtered_indices.deinit(self.allocator);
        self.history.deinit(self.allocator);
    }

    fn theme(self: *const TuiState) *const tui.Theme {
        return self.theme_manager.current;
    }

    fn addToHistory(self: *TuiState, cmd: Command) !void {
        // Remove duplicates
        var i: usize = 0;
        while (i < self.history.items.len) {
            if (self.history.items[i].command == cmd) {
                _ = self.history.orderedRemove(i);
            } else {
                i += 1;
            }
        }
        // Add to front
        try self.history.insert(self.allocator, 0, .{
            .command = cmd,
            .timestamp = std.time.milliTimestamp(),
        });
        // Keep only last 10
        while (self.history.items.len > 10) {
            _ = self.history.pop();
        }
    }

    fn showNotification(self: *TuiState, message: []const u8, level: tui.Toast.Level) void {
        self.notification = message;
        self.notification_level = level;
        self.notification_time = std.time.milliTimestamp();
    }

    fn clearExpiredNotification(self: *TuiState) void {
        if (self.notification != null) {
            const elapsed = std.time.milliTimestamp() - self.notification_time;
            if (elapsed > 3000) { // 3 second display
                self.notification = null;
            }
        }
    }

    fn resetFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        for (self.items, 0..) |_, i| {
            try self.filtered_indices.append(self.allocator, i);
        }
        self.selected = 0;
        self.scroll_offset = 0;
    }

    fn applyFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        const query = self.search_buffer[0..self.search_len];

        for (self.items, 0..) |item, i| {
            if (query.len == 0 or containsIgnoreCase(item.label, query) or containsIgnoreCase(item.description, query)) {
                try self.filtered_indices.append(self.allocator, i);
            }
        }

        if (self.selected >= self.filtered_indices.items.len) {
            self.selected = if (self.filtered_indices.items.len > 0) self.filtered_indices.items.len - 1 else 0;
        }
        self.scroll_offset = 0;
    }

    fn selectedItem(self: *const TuiState) ?*const MenuItem {
        if (self.filtered_indices.items.len == 0) return null;
        const idx = self.filtered_indices.items[self.selected];
        return &self.items[idx];
    }

    fn moveUp(self: *TuiState) void {
        if (self.selected > 0) {
            self.selected -= 1;
            if (self.selected < self.scroll_offset) {
                self.scroll_offset = self.selected;
            }
        }
    }

    fn moveDown(self: *TuiState) void {
        if (self.selected + 1 < self.filtered_indices.items.len) {
            self.selected += 1;
            if (self.selected >= self.scroll_offset + self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    fn pageUp(self: *TuiState) void {
        if (self.selected >= self.visible_rows) {
            self.selected -= self.visible_rows;
        } else {
            self.selected = 0;
        }
        if (self.selected < self.scroll_offset) {
            self.scroll_offset = self.selected;
        }
    }

    fn pageDown(self: *TuiState) void {
        const max = self.filtered_indices.items.len;
        if (max == 0) return;
        if (self.selected + self.visible_rows < max) {
            self.selected += self.visible_rows;
        } else {
            self.selected = max - 1;
        }
        if (self.selected >= self.scroll_offset + self.visible_rows) {
            self.scroll_offset = self.selected - self.visible_rows + 1;
        }
    }

    fn goHome(self: *TuiState) void {
        self.selected = 0;
        self.scroll_offset = 0;
    }

    fn goEnd(self: *TuiState) void {
        if (self.filtered_indices.items.len > 0) {
            self.selected = self.filtered_indices.items.len - 1;
            if (self.selected >= self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    fn handleMouseClick(self: *TuiState, row: u16) bool {
        // Account for header (4 rows: title, border, blank, category header)
        const header_rows: u16 = 4;
        if (row < header_rows) return false;

        const menu_row = row - header_rows;
        const clicked_idx = self.scroll_offset + @as(usize, menu_row);

        if (clicked_idx < self.filtered_indices.items.len) {
            self.selected = clicked_idx;
            return true; // Indicate selection changed, could trigger action
        }
        return false;
    }
};

const empty_args = &[_][:0]const u8{};

// ═══════════════════════════════════════════════════════════════════════════
// Box Drawing Characters
// ═══════════════════════════════════════════════════════════════════════════

const box = struct {
    const tl = "╭"; // Top-left
    const tr = "╮"; // Top-right
    const bl = "╰"; // Bottom-left
    const br = "╯"; // Bottom-right
    const h = "─"; // Horizontal
    const v = "│"; // Vertical
    const lsep = "├"; // Left separator
    const rsep = "┤"; // Right separator
};

// ═══════════════════════════════════════════════════════════════════════════
// Colors
// ═══════════════════════════════════════════════════════════════════════════

const colors = struct {
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const dim = "\x1b[2m";
    const italic = "\x1b[3m";
    const underline = "\x1b[4m";

    const black = "\x1b[30m";
    const red = "\x1b[31m";
    const green = "\x1b[32m";
    const yellow = "\x1b[33m";
    const blue = "\x1b[34m";
    const magenta = "\x1b[35m";
    const cyan = "\x1b[36m";
    const white = "\x1b[37m";

    const bg_black = "\x1b[40m";
    const bg_blue = "\x1b[44m";
    const bg_cyan = "\x1b[46m";

    const bright_black = "\x1b[90m";
    const bright_white = "\x1b[97m";
};

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Entry point for the TUI command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    try runInteractive(allocator, &framework);
}

fn runInteractive(allocator: std.mem.Allocator, framework: *abi.Framework) !void {
    // Check platform support before initializing
    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("TUI is not supported on {s}.", .{caps.platform_name});
        utils.output.printInfo("This platform lacks terminal control capabilities required for the interactive UI.", .{});
        return;
    }

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    terminal.enter() catch |err| {
        switch (err) {
            error.PlatformNotSupported => {
                const caps = tui.Terminal.capabilities();
                utils.output.printError("TUI is not supported on {s}.", .{caps.platform_name});
            },
            error.ConsoleUnavailable, error.ConsoleModeFailed => {
                utils.output.printError("Console is not available or cannot be configured.", .{});
                utils.output.printInfo("On Windows, ensure you're running in a terminal (not through pipes).", .{});
            },
            else => {
                utils.output.printError("Failed to initialize terminal: {t}", .{err});
            },
        }
        utils.output.printInfo("Run directly from a terminal for interactive features.", .{});
        return;
    };
    defer terminal.exit() catch {};

    var state = try TuiState.init(allocator, &terminal, framework);
    defer state.deinit();

    // Calculate visible rows based on terminal size
    state.term_size = terminal.size();
    state.visible_rows = @max(5, state.term_size.rows -| 10);

    while (true) {
        // Refresh terminal size
        state.term_size = terminal.size();
        state.visible_rows = @max(5, state.term_size.rows -| 10);

        // Clear and render
        try terminal.clear();
        try renderFrame(&state);

        // Read input
        const event = try terminal.readEvent();

        switch (event) {
            .key => |key| {
                if (try handleKeyEvent(&state, key)) break;
            },
            .mouse => |mouse| {
                if (mouse.pressed and mouse.button == .left) {
                    if (state.handleMouseClick(mouse.row)) {
                        // Double-click could run the command
                    }
                } else if (mouse.button == .wheel_up) {
                    state.moveUp();
                } else if (mouse.button == .wheel_down) {
                    state.moveDown();
                }
            },
        }
    }
}

fn handleKeyEvent(state: *TuiState, key: tui.Key) !bool {
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
                        state.showNotification("Theme changed", .info);
                    },
                    'T' => {
                        // Cycle theme backwards
                        state.theme_manager.prevTheme();
                        state.showNotification("Theme changed", .info);
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
                        if (findByShortcut(state.items, num)) |idx| {
                            const item = state.items[idx];
                            if (item.action == .command) {
                                try state.addToHistory(item.action.command);
                            }
                            try state.terminal.exit();
                            try runAction(state.allocator, state.framework, item.action);
                            std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                            _ = state.terminal.readKey() catch {};
                            try state.terminal.enter();
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
                if (item.action == .quit) return true;

                // Add to history
                if (item.action == .command) {
                    try state.addToHistory(item.action.command);
                }

                try state.terminal.exit();
                try runAction(state.allocator, state.framework, item.action);

                std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                _ = state.terminal.readKey() catch {};

                try state.terminal.enter();
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
                if (item.action == .quit) return true;

                if (item.action == .command) {
                    try state.addToHistory(item.action.command);
                }

                try state.terminal.exit();
                try runAction(state.allocator, state.framework, item.action);

                std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                _ = state.terminal.readKey() catch {};

                try state.terminal.enter();
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
        },
        .enter => {
            state.search_mode = false;
            // Execute selected if any
            if (state.selectedItem()) |item| {
                if (item.action == .quit) return true;

                try state.terminal.exit();
                try runAction(state.allocator, state.framework, item.action);

                std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                _ = state.terminal.readKey() catch {};

                try state.terminal.enter();
            }
        },
        .backspace => {
            if (state.search_len > 0) {
                state.search_len -= 1;
                try state.applyFilter();
            }
        },
        .character => {
            if (key.char) |ch| {
                if (state.search_len < state.search_buffer.len) {
                    state.search_buffer[state.search_len] = ch;
                    state.search_len += 1;
                    try state.applyFilter();
                }
            }
        },
        .up => state.moveUp(),
        .down => state.moveDown(),
        else => {},
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering
// ═══════════════════════════════════════════════════════════════════════════

fn renderFrame(state: *TuiState) !void {
    const term = state.terminal;
    const width: usize = @min(80, state.term_size.cols);

    // Handle preview mode
    if (state.preview_mode) {
        try renderPreview(term, state, width);
        return;
    }

    // Title bar
    try renderTitleBar(term, state, width);

    // Notification (if any)
    if (state.notification) |msg| {
        try renderNotification(term, state, width, msg);
    }

    // Search bar (if active or has content)
    if (state.search_mode or state.search_len > 0) {
        try renderSearchBar(term, state, width);
    }

    // History panel (if shown)
    if (state.show_history and state.history.items.len > 0) {
        try renderHistory(term, state, width);
    }

    // Menu items
    try renderMenuItems(term, state, width);

    // Status bar
    try renderStatusBar(term, state, width);

    // Help bar
    try renderHelpBar(term, state, width);
}

fn renderTitleBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();

    // Top border
    try term.write(theme.border);
    try term.write(box.tl);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.tr);
    try term.write(theme.reset);
    try term.write("\n");

    // Title
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);

    const title = " ABI Framework ";
    const version_str = abi.version();
    const theme_indicator = state.theme_manager.current.name;
    const title_len = title.len + version_str.len + theme_indicator.len + 6; // " vX.X.X [theme]"
    const left_pad = (width - 2 - title_len) / 2;
    const right_pad = width - 2 - title_len - left_pad;

    try writeRepeat(term, " ", left_pad);
    try term.write(theme.bold);
    try term.write(theme.primary);
    try term.write(title);
    try term.write(theme.text_dim);
    try term.write("v");
    try term.write(version_str);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.text_muted);
    try term.write("[");
    try term.write(theme_indicator);
    try term.write("]");
    try term.write(theme.reset);
    try writeRepeat(term, " ", right_pad);

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");

    // Separator
    try term.write(theme.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderNotification(term: *tui.Terminal, state: *TuiState, width: usize, msg: []const u8) !void {
    const theme = state.theme();
    const level_color = switch (state.notification_level) {
        .success => theme.success,
        .info => theme.info,
        .warning => theme.warning,
        .@"error" => theme.@"error",
    };
    const icon = switch (state.notification_level) {
        .success => "✓",
        .info => "ℹ",
        .warning => "⚠",
        .@"error" => "✗",
    };

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(level_color);
    try term.write(icon);
    try term.write(" ");
    try term.write(msg);
    try term.write(theme.reset);

    const used = 4 + msg.len;
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderHistory(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();

    // Header
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.bold);
    try term.write(theme.accent);
    try term.write("Recent Commands:");
    try term.write(theme.reset);

    try writeRepeat(term, " ", width - 20);
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");

    // Show up to 5 recent commands
    const max_show = @min(state.history.items.len, 5);
    for (0..max_show) |i| {
        const entry = state.history.items[i];
        const cmd_name = commandName(entry.command);

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("   ");
        try term.write(theme.text_dim);
        var num_buf: [2]u8 = undefined;
        num_buf[0] = '1' + @as(u8, @intCast(i));
        num_buf[1] = '.';
        try term.write(&num_buf);
        try term.write(" ");
        try term.write(theme.reset);
        try term.write(theme.secondary);
        try term.write(cmd_name);
        try term.write(theme.reset);

        const used = 7 + cmd_name.len;
        if (used < width - 1) {
            try writeRepeat(term, " ", width - 1 - used);
        }

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("\n");
    }

    // Separator
    try term.write(theme.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderPreview(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();
    const item = state.selectedItem() orelse return;

    // Double-line box for preview
    try term.write("\n");
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dtl);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(tui.widgets.box.dtr);
    try term.write(theme.reset);
    try term.write("\n");

    // Title
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.bold);
    try term.write(item.categoryColor(theme));
    try term.write(item.label);
    try term.write(theme.reset);

    const title_len = item.label.len + 2;
    if (title_len < width - 1) {
        try writeRepeat(term, " ", width - 1 - title_len);
    }
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("\n");

    // Description
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.text_dim);
    try term.write(item.description);
    try term.write(theme.reset);

    const desc_len = item.description.len + 2;
    if (desc_len < width - 1) {
        try writeRepeat(term, " ", width - 1 - desc_len);
    }
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("\n");

    // Separator
    try term.write(theme.primary);
    try term.write(box.lsep);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");

    // Usage section
    if (item.usage.len > 0) {
        try renderPreviewSection(term, theme, "Usage", width);
        try renderPreviewLine(term, theme, item.usage, width);
    }

    // Examples section
    if (item.examples.len > 0) {
        try renderPreviewSection(term, theme, "Examples", width);
        for (item.examples) |example| {
            try term.write(theme.primary);
            try term.write(tui.widgets.box.dv);
            try term.write(theme.reset);
            try term.write("   ");
            try term.write(theme.success);
            try term.write("$ ");
            try term.write(theme.reset);
            try term.write(example);

            const ex_len = example.len + 5;
            if (ex_len < width - 1) {
                try writeRepeat(term, " ", width - 1 - ex_len);
            }
            try term.write(theme.primary);
            try term.write(tui.widgets.box.dv);
            try term.write(theme.reset);
            try term.write("\n");
        }
    }

    // Related commands
    if (item.related.len > 0) {
        try renderPreviewSection(term, theme, "Related", width);
        try term.write(theme.primary);
        try term.write(tui.widgets.box.dv);
        try term.write(theme.reset);
        try term.write("   ");

        var total_len: usize = 3;
        for (item.related, 0..) |rel, i| {
            if (i > 0) {
                try term.write(", ");
                total_len += 2;
            }
            try term.write(theme.accent);
            try term.write(rel);
            try term.write(theme.reset);
            total_len += rel.len;
        }

        if (total_len < width - 1) {
            try writeRepeat(term, " ", width - 1 - total_len);
        }
        try term.write(theme.primary);
        try term.write(tui.widgets.box.dv);
        try term.write(theme.reset);
        try term.write("\n");
    }

    // Footer
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dbl);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(tui.widgets.box.dbr);
    try term.write(theme.reset);
    try term.write("\n\n");

    // Help text
    try term.write(theme.text_dim);
    try term.write(" Press ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("Enter");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" to run, ");
    try term.write(theme.reset);
    try term.write(theme.accent);
    try term.write("Esc");
    try term.write(theme.reset);
    try term.write(theme.text_dim);
    try term.write(" to go back\n");
    try term.write(theme.reset);
}

fn renderPreviewSection(term: *tui.Terminal, theme: *const tui.Theme, title: []const u8, width: usize) !void {
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try writeRepeat(term, " ", width - 2);
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("\n");

    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write(" ");
    try term.write(theme.bold);
    try term.write(theme.primary);
    try term.write(title);
    try term.write(":");
    try term.write(theme.reset);

    const sect_len = title.len + 3;
    if (sect_len < width - 1) {
        try writeRepeat(term, " ", width - 1 - sect_len);
    }
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderPreviewLine(term: *tui.Terminal, theme: *const tui.Theme, text: []const u8, width: usize) !void {
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("   ");
    try term.write(text);

    const line_len = text.len + 3;
    if (line_len < width - 1) {
        try writeRepeat(term, " ", width - 1 - line_len);
    }
    try term.write(theme.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(theme.reset);
    try term.write("\n");
}

fn commandName(cmd: Command) []const u8 {
    return switch (cmd) {
        .db => "db",
        .agent => "agent",
        .bench => "bench",
        .config => "config",
        .discord => "discord",
        .embed => "embed",
        .explore => "explore",
        .gpu => "gpu",
        .llm => "llm",
        .network => "network",
        .simd => "simd",
        .system_info => "system-info",
        .train => "train",
        .task => "task",
    };
}

fn renderSearchBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);

    try term.write(" ");
    if (state.search_mode) {
        try term.write(theme.accent);
    } else {
        try term.write(theme.text_dim);
    }
    try term.write("/");
    try term.write(theme.reset);
    try term.write(" ");

    const query = state.search_buffer[0..state.search_len];
    try term.write(query);

    if (state.search_mode) {
        try term.write(theme.accent);
        try term.write("_");
        try term.write(theme.reset);
    }

    const used = 4 + query.len + @as(usize, if (state.search_mode) 1 else 0);
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");

    // Separator
    try term.write(theme.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderMenuItems(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();
    const items = state.items;
    const indices = state.filtered_indices.items;
    const visible = state.visible_rows;
    const start = state.scroll_offset;
    const end = @min(start + visible, indices.len);

    // Show scroll indicator if needed
    if (start > 0) {
        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write("   ▲ more above");
        try writeRepeat(term, " ", width - 18);
        try term.write(theme.reset);
        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("\n");
    }

    for (start..end) |i| {
        const idx = indices[i];
        const item = items[idx];
        const is_selected = i == state.selected;

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);

        if (is_selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
            try term.write(" ▸ ");
        } else {
            try term.write("   ");
        }

        // Shortcut number
        if (item.shortcut) |num| {
            try term.write(theme.text_dim);
            var buf: [2]u8 = undefined;
            buf[0] = '0' + num;
            buf[1] = 0;
            try term.write(buf[0..1]);
            try term.write(theme.reset);
            if (is_selected) {
                try term.write(theme.selection_bg);
                try term.write(theme.selection_fg);
            }
            try term.write(" ");
        } else {
            try term.write("  ");
        }

        // Category color
        try term.write(item.categoryColor(theme));
        if (is_selected) try term.write(theme.bold);
        try term.write(item.label);
        try term.write(theme.reset);
        if (is_selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
        }

        // Padding
        const label_len = item.label.len;
        const padding = 16 -| @min(label_len, 16);
        try writeRepeat(term, " ", padding);

        // Description
        try term.write(theme.text_dim);
        if (is_selected) try term.write(theme.selection_fg);
        const desc_max = width -| 24;
        const desc_len = @min(item.description.len, desc_max);
        try term.write(item.description[0..desc_len]);
        try term.write(theme.reset);

        // Fill rest
        const used = 6 + label_len + padding + desc_len;
        if (used < width - 1) {
            try writeRepeat(term, " ", width - 1 - used);
        }

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("\n");
    }

    // Show scroll indicator if needed
    if (end < indices.len) {
        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write("   ▼ more below");
        try writeRepeat(term, " ", width - 18);
        try term.write(theme.reset);
        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("\n");
    }

    // Fill empty rows if menu is shorter than visible area
    const rendered = end - start + @as(usize, if (start > 0) 1 else 0) + @as(usize, if (end < indices.len) 1 else 0);
    if (rendered < visible) {
        for (0..(visible - rendered)) |_| {
            try term.write(theme.border);
            try term.write(box.v);
            try term.write(theme.reset);
            try writeRepeat(term, " ", width - 2);
            try term.write(theme.border);
            try term.write(box.v);
            try term.write(theme.reset);
            try term.write("\n");
        }
    }
}

fn renderStatusBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();

    // Separator
    try term.write(theme.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(theme.reset);
    try term.write("\n");

    // Status content
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write(" ");

    // OS info - use platform capabilities for accurate detection
    const os_name = tui.Terminal.platformName();
    try term.write(theme.text_dim);
    try term.write(os_name);

    // CPU count
    var cpu_buf: [16]u8 = undefined;
    const cpu_str = std.fmt.bufPrint(&cpu_buf, " │ {d} CPU", .{cpu_count}) catch "";
    try term.write(cpu_str);

    // TTY indicator
    try term.write(if (is_tty) " │ TTY" else " │ pipe");

    // Item count
    var count_buf: [32]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, " │ {d}/{d} items", .{
        state.filtered_indices.items.len,
        state.items.len,
    }) catch "?";
    try term.write(count_str);
    try term.write(theme.reset);

    const used = 2 + os_name.len + cpu_str.len + 7 + count_str.len; // 7 for TTY/pipe indicator
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");
}

fn renderHelpBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const theme = state.theme();

    // Bottom border with help
    try term.write(theme.border);
    try term.write(box.bl);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.br);
    try term.write(theme.reset);
    try term.write("\n");

    // Help text
    try term.write(" ");
    if (state.search_mode) {
        try term.write(theme.text_dim);
        try term.write("Type to filter │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("Enter");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Run │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("Esc");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Cancel");
        try term.write(theme.reset);
    } else {
        try term.write(theme.accent);
        try term.write("Enter");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Run │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("?");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Preview │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("t");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Theme │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("h");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" History │ ");
        try term.write(theme.reset);
        try term.write(theme.accent);
        try term.write("q");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Quit");
        try term.write(theme.reset);
    }
    try term.write("\n");
}

fn writeRepeat(term: *tui.Terminal, char: []const u8, count: usize) !void {
    for (0..count) |_| {
        try term.write(char);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Actions
// ═══════════════════════════════════════════════════════════════════════════

fn runAction(allocator: std.mem.Allocator, framework: *abi.Framework, action: Action) !void {
    _ = framework;
    std.debug.print("\n", .{});

    switch (action) {
        .command => |cmd| try runCommand(allocator, cmd),
        .version => utils.output.printInfo("ABI Framework v{s}", .{abi.version()}),
        .help => printHelp(),
        .quit => {},
    }
}

fn runCommand(allocator: std.mem.Allocator, cmd: Command) !void {
    switch (cmd) {
        .db => try db.run(allocator, empty_args),
        .agent => try agent.run(allocator, empty_args),
        .bench => try bench.run(allocator, empty_args),
        .config => try config.run(allocator, empty_args),
        .discord => try discord.run(allocator, empty_args),
        .embed => try embed.run(allocator, empty_args),
        .explore => try explore.run(allocator, empty_args),
        .gpu => try gpu.run(allocator, empty_args),
        .llm => try llm.run(allocator, empty_args),
        .network => try network.run(allocator, empty_args),
        .simd => try simd.run(allocator, empty_args),
        .system_info => try system_info.run(allocator, empty_args),
        .train => try train.run(allocator, empty_args),
        .task => try task.run(allocator, empty_args),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Menu Items
// ═══════════════════════════════════════════════════════════════════════════

fn menuItemsExtended() []const MenuItem {
    return &[_]MenuItem{
        // AI & ML (shortcuts 1-3)
        .{
            .label = "AI Agent",
            .description = "Interactive AI assistant",
            .action = .{ .command = .agent },
            .category = .ai,
            .shortcut = 1,
            .usage = "abi agent [--message \"...\"] [--persona <name>]",
            .examples = &[_][]const u8{ "abi agent", "abi agent --message \"Hello\"" },
            .related = &[_][]const u8{ "llm", "train" },
        },
        .{
            .label = "LLM",
            .description = "Local LLM inference",
            .action = .{ .command = .llm },
            .category = .ai,
            .shortcut = 2,
            .usage = "abi llm <subcommand> [options]",
            .examples = &[_][]const u8{ "abi llm chat", "abi llm generate \"prompt\"", "abi llm info" },
            .related = &[_][]const u8{ "agent", "embed" },
        },
        .{
            .label = "Training",
            .description = "Run training pipelines",
            .action = .{ .command = .train },
            .category = .ai,
            .shortcut = 3,
            .usage = "abi train <subcommand> [options]",
            .examples = &[_][]const u8{ "abi train run", "abi train resume", "abi train info" },
            .related = &[_][]const u8{ "agent", "llm" },
        },
        .{
            .label = "Embeddings",
            .description = "Generate embeddings",
            .action = .{ .command = .embed },
            .category = .ai,
            .usage = "abi embed [--provider <name>] <text>",
            .examples = &[_][]const u8{ "abi embed \"hello world\"", "abi embed --provider openai \"text\"" },
            .related = &[_][]const u8{ "db", "llm" },
        },

        // Data (shortcuts 4-5)
        .{
            .label = "Database",
            .description = "Manage vector database",
            .action = .{ .command = .db },
            .category = .data,
            .shortcut = 4,
            .usage = "abi db <subcommand> [options]",
            .examples = &[_][]const u8{ "abi db stats", "abi db add", "abi db query", "abi db backup" },
            .related = &[_][]const u8{ "embed", "explore" },
        },
        .{
            .label = "Explore",
            .description = "Search the codebase",
            .action = .{ .command = .explore },
            .category = .data,
            .shortcut = 5,
            .usage = "abi explore [query]",
            .examples = &[_][]const u8{ "abi explore", "abi explore \"function name\"" },
            .related = &[_][]const u8{ "db", "agent" },
        },

        // System (shortcuts 6-7)
        .{
            .label = "GPU",
            .description = "GPU devices and backends",
            .action = .{ .command = .gpu },
            .category = .system,
            .shortcut = 6,
            .usage = "abi gpu <subcommand>",
            .examples = &[_][]const u8{ "abi gpu backends", "abi gpu devices", "abi gpu summary" },
            .related = &[_][]const u8{ "bench", "system-info" },
        },
        .{
            .label = "Network",
            .description = "Cluster management",
            .action = .{ .command = .network },
            .category = .system,
            .shortcut = 7,
            .usage = "abi network <subcommand>",
            .examples = &[_][]const u8{ "abi network list", "abi network status", "abi network register" },
            .related = &[_][]const u8{ "system-info", "config" },
        },
        .{
            .label = "System Info",
            .description = "System and framework status",
            .action = .{ .command = .system_info },
            .category = .system,
            .usage = "abi system-info",
            .examples = &[_][]const u8{"abi system-info"},
            .related = &[_][]const u8{ "gpu", "network" },
        },

        // Tools (shortcuts 8-9)
        .{
            .label = "Benchmarks",
            .description = "Performance benchmarks",
            .action = .{ .command = .bench },
            .category = .tools,
            .shortcut = 8,
            .usage = "abi bench [suite]",
            .examples = &[_][]const u8{ "abi bench", "abi bench all", "abi bench simd" },
            .related = &[_][]const u8{ "simd", "gpu" },
        },
        .{
            .label = "SIMD",
            .description = "SIMD performance demo",
            .action = .{ .command = .simd },
            .category = .tools,
            .shortcut = 9,
            .usage = "abi simd",
            .examples = &[_][]const u8{"abi simd"},
            .related = &[_][]const u8{ "bench", "gpu" },
        },
        .{
            .label = "Config",
            .description = "Configuration management",
            .action = .{ .command = .config },
            .category = .tools,
            .usage = "abi config <subcommand>",
            .examples = &[_][]const u8{ "abi config show", "abi config init", "abi config validate" },
            .related = &[_][]const u8{ "system-info", "network" },
        },
        .{
            .label = "Tasks",
            .description = "Task management",
            .action = .{ .command = .task },
            .category = .tools,
            .usage = "abi task <subcommand>",
            .examples = &[_][]const u8{ "abi task list", "abi task add", "abi task done" },
            .related = &[_][]const u8{ "agent", "config" },
        },
        .{
            .label = "Discord",
            .description = "Discord bot integration",
            .action = .{ .command = .discord },
            .category = .tools,
            .usage = "abi discord <subcommand>",
            .examples = &[_][]const u8{ "abi discord status", "abi discord guilds" },
            .related = &[_][]const u8{ "agent", "config" },
        },

        // Meta
        .{ .label = "Help", .description = "Show CLI usage", .action = .help, .category = .meta },
        .{ .label = "Version", .description = "Show version", .action = .version, .category = .meta },
        .{ .label = "Quit", .description = "Exit the launcher", .action = .quit, .category = .meta },
    };
}

fn findByShortcut(items: []const MenuItem, num: u8) ?usize {
    for (items, 0..) |item, i| {
        if (item.shortcut) |s| {
            if (s == num) return i;
        }
    }
    return null;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (needle, 0..) |nc, j| {
            const hc = haystack[i + j];
            if (toLower(hc) != toLower(nc)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

fn toLower(c: u8) u8 {
    if (c >= 'A' and c <= 'Z') return c + 32;
    return c;
}

fn printHelp() void {
    std.debug.print(
        \\{s}Usage:{s} abi tui
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
