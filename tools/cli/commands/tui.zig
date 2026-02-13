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
const model = @import("model.zig");
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
    model,
    network,
    simd,
    system_info,
    train,
    train_monitor,
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

/// Match type for completion scoring
const MatchType = enum {
    exact_prefix, // Exact prefix match (highest priority)
    fuzzy, // Fuzzy character match
    history_recent, // Recently used command
    substring, // Substring match (lowest priority)

    fn indicator(self: MatchType) []const u8 {
        return switch (self) {
            .exact_prefix => "≡",
            .fuzzy => "≈",
            .history_recent => "↺",
            .substring => "⊂",
        };
    }
};

/// Completion suggestion with ranking score
const CompletionSuggestion = struct {
    item_index: usize, // Index into MenuItems array
    score: u32, // Ranking score (higher = better)
    match_type: MatchType, // How the match was found
};

/// Completion state for the TUI
const CompletionState = struct {
    suggestions: std.ArrayListUnmanaged(CompletionSuggestion),
    selected_suggestion: usize, // Index into suggestions array
    active: bool, // Whether dropdown is shown
    max_visible: usize, // Max suggestions to show (default: 5)

    fn init() CompletionState {
        return .{
            .suggestions = .empty,
            .selected_suggestion = 0,
            .active = false,
            .max_visible = 5,
        };
    }

    fn deinit(self: *CompletionState, allocator: std.mem.Allocator) void {
        self.suggestions.deinit(allocator);
    }

    fn clear(self: *CompletionState) void {
        self.suggestions.clearRetainingCapacity();
        self.selected_suggestion = 0;
        self.active = false;
    }
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
    // Tab completion state
    completion: CompletionState,

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
            .completion = CompletionState.init(),
        };
        // Initialize with all items
        try state.resetFilter();
        return state;
    }

    fn deinit(self: *TuiState) void {
        self.filtered_indices.deinit(self.allocator);
        self.history.deinit(self.allocator);
        self.completion.deinit(self.allocator);
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
            .timestamp = abi.utils.unixMs(),
        });
        // Keep only last 10
        while (self.history.items.len > 10) {
            _ = self.history.pop();
        }
    }

    fn showNotification(self: *TuiState, message: []const u8, level: tui.Toast.Level) void {
        self.notification = message;
        self.notification_level = level;
        self.notification_time = abi.utils.unixMs();
    }

    fn clearExpiredNotification(self: *TuiState) void {
        if (self.notification != null) {
            const elapsed = abi.utils.unixMs() - self.notification_time;
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

    /// Generate completion suggestions based on current search query
    fn updateCompletions(self: *TuiState) !void {
        self.completion.clear();

        const query = self.search_buffer[0..self.search_len];
        if (query.len == 0) {
            self.completion.active = false;
            return;
        }

        // Score all items
        for (self.items, 0..) |*item, i| {
            if (calculateCompletionScore(item, query, self.history.items)) |suggestion| {
                var s = suggestion;
                s.item_index = i;
                try self.completion.suggestions.append(self.allocator, s);
            }
        }

        // Sort by score (highest first)
        std.mem.sort(
            CompletionSuggestion,
            self.completion.suggestions.items,
            {},
            suggestionCompare,
        );

        // Activate if we have suggestions
        self.completion.active = self.completion.suggestions.items.len > 0;
        self.completion.selected_suggestion = 0;
    }

    /// Cycle to next completion suggestion
    fn nextCompletion(self: *TuiState) void {
        if (!self.completion.active or self.completion.suggestions.items.len == 0) return;
        self.completion.selected_suggestion += 1;
        if (self.completion.selected_suggestion >= self.completion.suggestions.items.len) {
            self.completion.selected_suggestion = 0;
        }
    }

    /// Cycle to previous completion suggestion
    fn prevCompletion(self: *TuiState) void {
        if (!self.completion.active or self.completion.suggestions.items.len == 0) return;
        if (self.completion.selected_suggestion == 0) {
            self.completion.selected_suggestion = self.completion.suggestions.items.len - 1;
        } else {
            self.completion.selected_suggestion -= 1;
        }
    }

    /// Accept current completion suggestion
    fn acceptCompletion(self: *TuiState) !void {
        if (!self.completion.active or self.completion.suggestions.items.len == 0) return;

        const suggestion = self.completion.suggestions.items[self.completion.selected_suggestion];
        const item = &self.items[suggestion.item_index];

        // Copy label to search buffer
        const label = item.label;
        const copy_len = @min(label.len, self.search_buffer.len);
        @memcpy(self.search_buffer[0..copy_len], label[0..copy_len]);
        self.search_len = copy_len;

        // Update filter and select the completed item
        try self.applyFilter();

        // Find the item in filtered results and select it
        for (self.filtered_indices.items, 0..) |idx, i| {
            if (idx == suggestion.item_index) {
                self.selected = i;
                break;
            }
        }

        // Hide completions after accepting
        self.completion.clear();
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
pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    const fw_config = (abi.FrameworkOptions{}).toConfig();
    var framework = try abi.Framework.initWithIo(allocator, fw_config, io);
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

    // NOTE: Removed explicit TTY check using std.os.isatty which is not available on Windows.
    // The `terminal.enter()` call below will gracefully handle non‑interactive environments
    // and provide the same fallback command list when a real terminal cannot be used.

    var terminal = tui.Terminal.init(allocator);
    defer terminal.deinit();

    // Attempt to enter the TUI. Preserve detailed error information for debugging.
    terminal.enter() catch |err| {
        // Log the specific error using the {t} formatter for clarity.
        utils.output.printError("Failed to start interactive TUI: {t}", .{err});
        utils.output.printInfo("Falling back to command list display.", .{});
        std.debug.print("\nAvailable commands (run individually):\n", .{});
        std.debug.print("  abi llm list                    - List supported LLM formats\n", .{});
        std.debug.print("  abi llm demo                    - Demo LLM interface (no model needed)\n", .{});
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
                        // Execute the selected action
                        if (state.selectedItem()) |item| {
                            if (item.action == .quit) break;

                            if (item.action == .command) {
                                try state.addToHistory(item.action.command);
                            }

                            try state.terminal.exit();
                            try runAction(state.allocator, state.framework, item.action);

                            std.debug.print("\n{s}Press Enter to return to menu...{s}", .{ colors.dim, colors.reset });
                            _ = state.terminal.readKey() catch {};

                            try state.terminal.enter();
                        }
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
                state.completion.clear();
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
            state.completion.clear();
        },
        .tab => {
            // Tab completion
            if (state.completion.active) {
                state.nextCompletion();
            } else {
                try state.updateCompletions();
            }
        },
        .enter => {
            // Accept completion if active, otherwise execute selected
            if (state.completion.active) {
                try state.acceptCompletion();
                return false; // Stay in search mode
            }

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
            if (state.completion.active) {
                state.prevCompletion();
            } else {
                state.moveUp();
            }
        },
        .down => {
            // Navigate completions with arrow keys
            if (state.completion.active) {
                state.nextCompletion();
            } else {
                state.moveDown();
            }
        },
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

        // Render completion dropdown if active
        if (state.search_mode and state.completion.active) {
            try renderCompletionDropdown(term, state, width);
        }
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
        .model => "model",
        .network => "network",
        .simd => "simd",
        .system_info => "system-info",
        .train => "train",
        .train_monitor => "train-monitor",
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

fn renderCompletionDropdown(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    if (!state.completion.active) return;

    const theme = state.theme();
    const suggestions = state.completion.suggestions.items;
    const max_show = @min(state.completion.max_visible, suggestions.len);

    if (max_show == 0) return;

    const dropdown_width = @min(50, width - 8);

    // Top border of dropdown
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("   ");
    try term.write(theme.text_dim);
    try term.write("╭");
    try writeRepeat(term, "─", dropdown_width);
    try term.write("╮");
    try term.write(theme.reset);

    const remaining = width -| (4 + dropdown_width + 2);
    try writeRepeat(term, " ", remaining);
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("\n");

    // Suggestions
    for (0..max_show) |i| {
        const suggestion = suggestions[i];
        const item = state.items[suggestion.item_index];
        const is_selected = i == state.completion.selected_suggestion;

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("   ");
        try term.write(theme.text_dim);
        try term.write("│");
        try term.write(theme.reset);

        if (is_selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
            try term.write(" ▸ ");
        } else {
            try term.write("   ");
        }

        // Match type indicator
        try term.write(theme.text_muted);
        try term.write(suggestion.match_type.indicator());
        try term.write(" ");
        try term.write(theme.reset);

        if (is_selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
        }

        // Label with category color
        try term.write(item.categoryColor(theme));
        if (is_selected) try term.write(theme.bold);
        const label_max = @min(item.label.len, 18);
        try term.write(item.label[0..label_max]);
        try term.write(theme.reset);

        if (is_selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
        }

        // Padding after label
        const label_pad = 18 -| label_max;
        try writeRepeat(term, " ", label_pad);

        // Short description
        try term.write(theme.text_dim);
        const desc_max = @min(item.description.len, dropdown_width - 28);
        try term.write(item.description[0..desc_max]);
        try term.write(theme.reset);

        // Fill rest of dropdown line
        const used_in_dropdown = 5 + label_max + label_pad + desc_max;
        if (used_in_dropdown < dropdown_width) {
            try writeRepeat(term, " ", dropdown_width - used_in_dropdown);
        }

        try term.write(theme.text_dim);
        try term.write("│");
        try term.write(theme.reset);

        const final_pad = width -| (4 + dropdown_width + 2);
        try writeRepeat(term, " ", final_pad);

        try term.write(theme.border);
        try term.write(box.v);
        try term.write(theme.reset);
        try term.write("\n");
    }

    // Bottom border with count hint
    try term.write(theme.border);
    try term.write(box.v);
    try term.write(theme.reset);
    try term.write("   ");
    try term.write(theme.text_dim);
    try term.write("╰");

    // Show count if truncated
    if (suggestions.len > max_show) {
        var count_buf: [16]u8 = undefined;
        const count_str = std.fmt.bufPrint(&count_buf, " {d}/{d} ", .{
            max_show,
            suggestions.len,
        }) catch "";
        const hint_len = count_str.len;
        const bar_len = (dropdown_width -| hint_len) / 2;
        try writeRepeat(term, "─", bar_len);
        try term.write(count_str);
        try writeRepeat(term, "─", dropdown_width -| bar_len -| hint_len);
    } else {
        try writeRepeat(term, "─", dropdown_width);
    }

    try term.write("╯");
    try term.write(theme.reset);

    const remaining2 = width -| (4 + dropdown_width + 2);
    try writeRepeat(term, " ", remaining2);
    try term.write(theme.border);
    try term.write(box.v);
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
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const cpu_str = std.fmt.bufPrint(&cpu_buf, " │ {d} CPU", .{cpu_count}) catch "";
    try term.write(cpu_str);

    // TTY indicator (if we're in the TUI, we're in a terminal)
    try term.write(" │ TTY");

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
        try term.write("Tab");
        try term.write(theme.reset);
        try term.write(theme.text_dim);
        try term.write(" Complete │ ");
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
        .model => try model.run(allocator, empty_args),
        .network => try network.run(allocator, empty_args),
        .simd => try simd.run(allocator, empty_args),
        .system_info => try system_info.run(allocator, empty_args),
        .train => try train.run(allocator, empty_args),
        .train_monitor => try train.run(allocator, &[_][:0]const u8{"monitor"}),
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
            .related = &[_][]const u8{ "agent", "llm", "train-monitor" },
        },
        .{
            .label = "Training Monitor",
            .description = "Live training dashboard",
            .action = .{ .command = .train_monitor },
            .category = .ai,
            .usage = "abi train monitor [run-id]",
            .examples = &[_][]const u8{ "abi train monitor", "abi train monitor --log-dir ./logs" },
            .related = &[_][]const u8{ "train", "llm" },
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
        .{
            .label = "Model",
            .description = "Model management (download, cache, switch)",
            .action = .{ .command = .model },
            .category = .ai,
            .usage = "abi model <subcommand> [options]",
            .examples = &[_][]const u8{ "abi model list", "abi model download llama-7b", "abi model info mistral" },
            .related = &[_][]const u8{ "llm", "agent", "embed" },
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

/// Check if haystack starts with needle (case-insensitive)
fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (needle, 0..) |nc, i| {
        if (toLower(haystack[i]) != toLower(nc)) return false;
    }
    return true;
}

/// Fuzzy match: returns score if all query characters appear in order
/// Higher score = better match (consecutive chars, early positions)
fn fuzzyMatch(label: []const u8, query: []const u8) ?u32 {
    if (query.len == 0) return null;
    if (query.len > label.len) return null;

    var score: u32 = 500; // Base fuzzy score
    var query_idx: usize = 0;
    var last_match_pos: usize = 0;
    var consecutive_bonus: u32 = 0;

    for (label, 0..) |lc, label_idx| {
        if (query_idx >= query.len) break;

        if (toLower(lc) == toLower(query[query_idx])) {
            // Bonus for consecutive matches
            if (label_idx == last_match_pos + 1 and label_idx > 0) {
                consecutive_bonus += 20;
            }
            // Bonus for early matches
            if (label_idx < 3) {
                score += 30 - @as(u32, @intCast(label_idx)) * 10;
            }
            last_match_pos = label_idx;
            query_idx += 1;
        }
    }

    // All query characters must be found
    if (query_idx < query.len) return null;

    return score + consecutive_bonus;
}

/// Calculate completion score for a menu item
fn calculateCompletionScore(item: *const MenuItem, query: []const u8, history_items: []const HistoryEntry) ?CompletionSuggestion {
    const label = item.label;

    // Check for exact prefix match (highest priority)
    if (startsWithIgnoreCase(label, query)) {
        // Check if recently used
        const is_recent = isRecentlyUsed(item, history_items);
        return CompletionSuggestion{
            .item_index = 0, // Will be set by caller
            .score = if (is_recent) 1100 else 1000,
            .match_type = if (is_recent) .history_recent else .exact_prefix,
        };
    }

    // Check for fuzzy match
    if (fuzzyMatch(label, query)) |fuzzy_score| {
        return CompletionSuggestion{
            .item_index = 0,
            .score = fuzzy_score,
            .match_type = .fuzzy,
        };
    }

    // Check for substring match in label or description
    if (containsIgnoreCase(label, query) or containsIgnoreCase(item.description, query)) {
        return CompletionSuggestion{
            .item_index = 0,
            .score = 200,
            .match_type = .substring,
        };
    }

    return null;
}

/// Check if an item was recently used
fn isRecentlyUsed(item: *const MenuItem, history_items: []const HistoryEntry) bool {
    switch (item.action) {
        .command => |cmd| {
            // Check recent history (last 5 items)
            const check_count = @min(history_items.len, 5);
            for (history_items[history_items.len - check_count ..]) |entry| {
                if (entry.command == cmd) return true;
            }
        },
        else => {},
    }
    return false;
}

/// Comparison function for sorting suggestions (higher score first)
fn suggestionCompare(_: void, a: CompletionSuggestion, b: CompletionSuggestion) bool {
    return a.score > b.score;
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
