//! TUI state management.
//!
//! Owns the interactive state: selection, scroll, search, history,
//! completion, themes. Methods mutate state in response to user actions.

const std = @import("std");
const abi = @import("abi");
const tui = @import("../core/mod.zig");
const types = @import("types.zig");
const completion = @import("completion.zig");
const menu = @import("menu.zig");
const tui_layout = @import("layout.zig");

const MenuItem = types.MenuItem;
const CompletionState = types.CompletionState;
const CompletionSuggestion = types.CompletionSuggestion;

pub const TuiState = struct {
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    framework: *abi.App,
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
    history: std.ArrayListUnmanaged(types.HistoryEntry),
    show_history: bool,
    notification: ?[]const u8,
    notification_level: tui.Toast.Level,
    notification_time: i64,
    // Tab completion state
    completion_state: CompletionState,

    pub fn init(
        allocator: std.mem.Allocator,
        terminal: *tui.Terminal,
        framework: *abi.App,
        initial_theme: *const tui.Theme,
    ) !TuiState {
        var theme_manager = tui.ThemeManager.init();
        theme_manager.current = initial_theme;

        var state = TuiState{
            .allocator = allocator,
            .terminal = terminal,
            .framework = framework,
            .items = menu.menuItemsExtended(),
            .filtered_indices = .empty,
            .selected = 0,
            .scroll_offset = 0,
            .search_mode = false,
            .search_buffer = undefined,
            .search_len = 0,
            .term_size = terminal.size(),
            .visible_rows = tui_layout.computeVisibleRows(terminal.size().rows),
            .theme_manager = theme_manager,
            .preview_mode = false,
            .history = .empty,
            .show_history = false,
            .notification = null,
            .notification_level = .info,
            .notification_time = 0,
            .completion_state = CompletionState.init(),
        };
        // Initialize with all items
        try state.resetFilter();
        return state;
    }

    pub fn deinit(self: *TuiState) void {
        self.filtered_indices.deinit(self.allocator);
        self.history.deinit(self.allocator);
        self.completion_state.deinit(self.allocator);
    }

    pub fn theme(self: *const TuiState) *const tui.Theme {
        return self.theme_manager.current;
    }

    pub fn addToHistory(self: *TuiState, command_id: []const u8) !void {
        // Remove duplicates
        var i: usize = 0;
        while (i < self.history.items.len) {
            if (std.mem.eql(u8, self.history.items[i].command_id, command_id)) {
                _ = self.history.orderedRemove(i);
            } else {
                i += 1;
            }
        }
        // Add to front
        try self.history.insert(self.allocator, 0, .{
            .command_id = command_id,
            .timestamp = abi.services.shared.utils.unixMs(),
        });
        // Keep only last 10
        while (self.history.items.len > 10) {
            _ = self.history.pop();
        }
    }

    pub fn showNotification(self: *TuiState, message: []const u8, level: tui.Toast.Level) void {
        self.notification = message;
        self.notification_level = level;
        self.notification_time = abi.services.shared.utils.unixMs();
    }

    pub fn clearExpiredNotification(self: *TuiState) void {
        if (self.notification != null) {
            const elapsed = abi.services.shared.utils.unixMs() - self.notification_time;
            if (elapsed > 3000) { // 3 second display
                self.notification = null;
            }
        }
    }

    pub fn resetFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        for (self.items, 0..) |_, i| {
            try self.filtered_indices.append(self.allocator, i);
        }
        self.selected = 0;
        self.scroll_offset = 0;
    }

    pub fn applyFilter(self: *TuiState) !void {
        self.filtered_indices.clearRetainingCapacity();
        const query = self.search_buffer[0..self.search_len];

        for (self.items, 0..) |item, i| {
            if (query.len == 0 or
                completion.containsIgnoreCase(item.label, query) or
                completion.containsIgnoreCase(item.description, query))
            {
                try self.filtered_indices.append(self.allocator, i);
            }
        }

        if (self.selected >= self.filtered_indices.items.len) {
            self.selected = if (self.filtered_indices.items.len > 0)
                self.filtered_indices.items.len - 1
            else
                0;
        }
        self.scroll_offset = 0;
    }

    pub fn selectedItem(self: *const TuiState) ?*const MenuItem {
        if (self.filtered_indices.items.len == 0) return null;
        const idx = self.filtered_indices.items[self.selected];
        return &self.items[idx];
    }

    pub fn moveUp(self: *TuiState) void {
        if (self.selected > 0) {
            self.selected -= 1;
            if (self.selected < self.scroll_offset) {
                self.scroll_offset = self.selected;
            }
        }
    }

    pub fn moveDown(self: *TuiState) void {
        if (self.selected + 1 < self.filtered_indices.items.len) {
            self.selected += 1;
            if (self.selected >= self.scroll_offset + self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    pub fn pageUp(self: *TuiState) void {
        if (self.selected >= self.visible_rows) {
            self.selected -= self.visible_rows;
        } else {
            self.selected = 0;
        }
        if (self.selected < self.scroll_offset) {
            self.scroll_offset = self.selected;
        }
    }

    pub fn pageDown(self: *TuiState) void {
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

    pub fn goHome(self: *TuiState) void {
        self.selected = 0;
        self.scroll_offset = 0;
    }

    pub fn goEnd(self: *TuiState) void {
        if (self.filtered_indices.items.len > 0) {
            self.selected = self.filtered_indices.items.len - 1;
            if (self.selected >= self.visible_rows) {
                self.scroll_offset = self.selected - self.visible_rows + 1;
            }
        }
    }

    /// Generate completion suggestions based on current search query.
    pub fn updateCompletions(self: *TuiState) !void {
        self.completion_state.clear();

        const query = self.search_buffer[0..self.search_len];
        if (query.len == 0) {
            self.completion_state.active = false;
            return;
        }

        // Score all items
        for (self.items, 0..) |*item, i| {
            if (completion.calculateCompletionScore(item, query, self.history.items)) |suggestion| {
                var s = suggestion;
                s.item_index = i;
                try self.completion_state.suggestions.append(self.allocator, s);
            }
        }

        // Sort by score (highest first)
        std.mem.sort(
            CompletionSuggestion,
            self.completion_state.suggestions.items,
            {},
            completion.suggestionCompare,
        );

        // Activate if we have suggestions
        self.completion_state.active = self.completion_state.suggestions.items.len > 0;
        self.completion_state.selected_suggestion = 0;
    }

    /// Cycle to next completion suggestion.
    pub fn nextCompletion(self: *TuiState) void {
        if (!self.completion_state.active or self.completion_state.suggestions.items.len == 0) return;
        self.completion_state.selected_suggestion += 1;
        if (self.completion_state.selected_suggestion >= self.completion_state.suggestions.items.len) {
            self.completion_state.selected_suggestion = 0;
        }
    }

    /// Cycle to previous completion suggestion.
    pub fn prevCompletion(self: *TuiState) void {
        if (!self.completion_state.active or self.completion_state.suggestions.items.len == 0) return;
        if (self.completion_state.selected_suggestion == 0) {
            self.completion_state.selected_suggestion = self.completion_state.suggestions.items.len - 1;
        } else {
            self.completion_state.selected_suggestion -= 1;
        }
    }

    /// Accept current completion suggestion.
    pub fn acceptCompletion(self: *TuiState) !void {
        if (!self.completion_state.active or self.completion_state.suggestions.items.len == 0) return;

        const suggestion = self.completion_state.suggestions.items[self.completion_state.selected_suggestion];
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
        self.completion_state.clear();
    }

    pub fn handleMouseClick(self: *TuiState, row: u16, menu_start_row: u16) bool {
        const row0 = if (row == 0) @as(u16, 0) else row - 1;
        const clicked_idx = tui_layout.clickedIndexFromRow(
            row0,
            menu_start_row,
            self.scroll_offset > 0,
            self.scroll_offset,
            self.visible_rows,
            self.filtered_indices.items.len,
        ) orelse return false;

        self.selected = clicked_idx;
        return true;
    }
};

test {
    std.testing.refAllDecls(@This());
}
