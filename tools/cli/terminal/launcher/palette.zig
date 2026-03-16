//! Reusable command palette for launcher-style UIs.

const std = @import("std");
const abi = @import("abi");
const tui = @import("../mod.zig");
const render_utils = @import("../render_utils.zig");
const completion = @import("completion.zig");
const launcher_catalog = @import("launcher_catalog.zig");
const types = @import("types.zig");

const MenuItem = types.MenuItem;
const Action = types.Action;
const CompletionState = types.CompletionState;
const CompletionSuggestion = types.CompletionSuggestion;
const HistoryEntry = types.HistoryEntry;

pub const Outcome = union(enum) {
    none,
    close,
    submit: Action,
    quit,
};

pub const CommandPalette = struct {
    allocator: std.mem.Allocator,
    items: []const MenuItem,
    filtered_indices: std.ArrayListUnmanaged(usize),
    history: std.ArrayListUnmanaged(HistoryEntry),
    completion_state: CompletionState,
    search_buffer: [64]u8,
    search_len: usize,
    selected: usize,
    scroll_offset: usize,
    visible_rows: usize,
    active: bool,

    pub fn init(allocator: std.mem.Allocator) !CommandPalette {
        var palette = CommandPalette{
            .allocator = allocator,
            .items = launcher_catalog.menuItems(),
            .filtered_indices = .empty,
            .history = .empty,
            .completion_state = CompletionState.init(),
            .search_buffer = undefined,
            .search_len = 0,
            .selected = 0,
            .scroll_offset = 0,
            .visible_rows = 6,
            .active = false,
        };
        try palette.resetFilter();
        return palette;
    }

    pub fn deinit(self: *CommandPalette) void {
        self.filtered_indices.deinit(self.allocator);
        self.history.deinit(self.allocator);
        self.completion_state.deinit(self.allocator);
    }

    pub fn open(self: *CommandPalette) !void {
        self.active = true;
        self.search_len = 0;
        self.selected = 0;
        self.scroll_offset = 0;
        self.completion_state.clear();
        try self.resetFilter();
    }

    pub fn close(self: *CommandPalette) void {
        self.active = false;
        self.completion_state.clear();
    }

    pub fn toggle(self: *CommandPalette) !void {
        if (self.active) {
            self.close();
        } else {
            try self.open();
        }
    }

    pub fn render(
        self: *CommandPalette,
        term: *tui.Terminal,
        theme: *const tui.Theme,
        term_size: tui.TerminalSize,
    ) !void {
        if (!self.active) return;

        const width = clampU16(term_size.cols, 56, 88) -| 4;
        const height = clampU16(term_size.rows, 14, 20) -| 2;
        const rect = tui.Rect{
            .x = (term_size.cols -| width) / 2,
            .y = (term_size.rows -| height) / 2,
            .width = width,
            .height = height,
        };
        if (rect.width < 8 or rect.height < 8) return;

        self.visible_rows = @max(@as(usize, 4), @min(@as(usize, 9), @as(usize, rect.height) -| 9));
        self.ensureSelectionVisible();

        try render_utils.drawBox(term, rect, .double, theme);
        try renderTitleRow(term, theme, rect);
        try renderSearchRow(self, term, theme, rect);
        try render_utils.drawSeparator(term, rect, 3, .double, theme);
        try renderItems(self, term, theme, rect);
        const preview_separator_row: u16 = 4 + @as(u16, @intCast(self.visible_rows));
        try render_utils.drawSeparator(term, rect, preview_separator_row, .double, theme);
        try renderPreview(self, term, theme, rect, preview_separator_row + 1);
    }

    pub fn handleKey(self: *CommandPalette, key: tui.Key) !Outcome {
        if (!self.active) return .none;

        switch (key.code) {
            .ctrl_c => return .quit,
            .escape => {
                self.close();
                return .close;
            },
            .enter => {
                if (self.selectedItem()) |item| {
                    switch (item.action) {
                        .command => |cmd| try self.addToHistory(cmd.id),
                        else => {},
                    }
                    self.close();
                    return .{ .submit = item.action };
                }
            },
            .backspace => {
                if (self.search_len > 0) {
                    self.search_len -= 1;
                    try self.applyFilter();
                    try self.updateCompletions();
                }
            },
            .tab => {
                if (self.completion_state.active) {
                    try self.acceptCompletion();
                } else {
                    try self.updateCompletions();
                }
            },
            .up => self.moveUp(),
            .down => self.moveDown(),
            .page_up => self.pageUp(),
            .page_down => self.pageDown(),
            .home => self.goHome(),
            .end => self.goEnd(),
            .character => {
                if (key.char) |ch| {
                    if (ch == '?') {
                        return .none;
                    }
                    if (self.search_len < self.search_buffer.len) {
                        self.search_buffer[self.search_len] = ch;
                        self.search_len += 1;
                        try self.applyFilter();
                        try self.updateCompletions();
                    }
                }
            },
            else => {},
        }

        return .none;
    }

    fn addToHistory(self: *CommandPalette, command_id: []const u8) !void {
        var i: usize = 0;
        while (i < self.history.items.len) {
            if (std.mem.eql(u8, self.history.items[i].command_id, command_id)) {
                _ = self.history.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        try self.history.insert(self.allocator, 0, .{
            .command_id = command_id,
            .timestamp = abi.foundation.utils.unixMs(),
        });
        while (self.history.items.len > 10) {
            _ = self.history.pop();
        }
    }

    fn resetFilter(self: *CommandPalette) !void {
        self.filtered_indices.clearRetainingCapacity();
        for (self.items, 0..) |_, i| {
            try self.filtered_indices.append(self.allocator, i);
        }
        self.selected = 0;
        self.scroll_offset = 0;
    }

    fn applyFilter(self: *CommandPalette) !void {
        self.filtered_indices.clearRetainingCapacity();
        const query = self.search_buffer[0..self.search_len];

        for (self.items, 0..) |item, i| {
            if (query.len == 0 or
                completion.containsIgnoreCase(item.label, query) or
                completion.containsIgnoreCase(item.description, query) or
                containsRelated(item.related, query))
            {
                try self.filtered_indices.append(self.allocator, i);
            }
        }

        if (self.filtered_indices.items.len == 0) {
            self.selected = 0;
            self.scroll_offset = 0;
            return;
        }

        if (self.selected >= self.filtered_indices.items.len) {
            self.selected = self.filtered_indices.items.len - 1;
        }
        self.ensureSelectionVisible();
    }

    fn updateCompletions(self: *CommandPalette) !void {
        self.completion_state.clear();

        const query = self.search_buffer[0..self.search_len];
        if (query.len == 0) return;

        for (self.items, 0..) |*item, i| {
            if (completion.calculateCompletionScore(item, query, self.history.items)) |suggestion| {
                var ranked = suggestion;
                ranked.item_index = i;
                try self.completion_state.suggestions.append(self.allocator, ranked);
            }
        }

        std.mem.sort(
            CompletionSuggestion,
            self.completion_state.suggestions.items,
            {},
            completion.suggestionCompare,
        );

        self.completion_state.active = self.completion_state.suggestions.items.len > 0;
        self.completion_state.selected_suggestion = 0;
    }

    fn acceptCompletion(self: *CommandPalette) !void {
        if (!self.completion_state.active or self.completion_state.suggestions.items.len == 0) return;

        const suggestion = self.completion_state.suggestions.items[self.completion_state.selected_suggestion];
        const item = &self.items[suggestion.item_index];
        const copy_len = @min(item.label.len, self.search_buffer.len);
        @memcpy(self.search_buffer[0..copy_len], item.label[0..copy_len]);
        self.search_len = copy_len;
        try self.applyFilter();

        for (self.filtered_indices.items, 0..) |idx, i| {
            if (idx == suggestion.item_index) {
                self.selected = i;
                break;
            }
        }

        self.completion_state.clear();
    }

    fn selectedItem(self: *const CommandPalette) ?*const MenuItem {
        if (self.filtered_indices.items.len == 0) return null;
        return &self.items[self.filtered_indices.items[self.selected]];
    }

    fn moveUp(self: *CommandPalette) void {
        if (self.selected > 0) {
            self.selected -= 1;
        }
        self.ensureSelectionVisible();
    }

    fn moveDown(self: *CommandPalette) void {
        if (self.selected + 1 < self.filtered_indices.items.len) {
            self.selected += 1;
        }
        self.ensureSelectionVisible();
    }

    fn pageUp(self: *CommandPalette) void {
        if (self.selected >= self.visible_rows) {
            self.selected -= self.visible_rows;
        } else {
            self.selected = 0;
        }
        self.ensureSelectionVisible();
    }

    fn pageDown(self: *CommandPalette) void {
        if (self.filtered_indices.items.len == 0) return;
        if (self.selected + self.visible_rows < self.filtered_indices.items.len) {
            self.selected += self.visible_rows;
        } else {
            self.selected = self.filtered_indices.items.len - 1;
        }
        self.ensureSelectionVisible();
    }

    fn goHome(self: *CommandPalette) void {
        self.selected = 0;
        self.ensureSelectionVisible();
    }

    fn goEnd(self: *CommandPalette) void {
        if (self.filtered_indices.items.len == 0) return;
        self.selected = self.filtered_indices.items.len - 1;
        self.ensureSelectionVisible();
    }

    fn ensureSelectionVisible(self: *CommandPalette) void {
        if (self.selected < self.scroll_offset) {
            self.scroll_offset = self.selected;
        } else if (self.selected >= self.scroll_offset + self.visible_rows) {
            self.scroll_offset = self.selected - self.visible_rows + 1;
        }
    }
};

fn renderTitleRow(term: *tui.Terminal, theme: *const tui.Theme, rect: tui.Rect) !void {
    try render_utils.moveTo(term, rect.x + 2, rect.y + 1);
    try term.write(theme.header);
    try term.write("Hybrid Command Palette");
    try term.write(theme.reset);
}

fn renderSearchRow(self: *CommandPalette, term: *tui.Terminal, theme: *const tui.Theme, rect: tui.Rect) !void {
    const inner_width = @as(usize, rect.width) -| 4;
    try render_utils.moveTo(term, rect.x + 2, rect.y + 2);
    try term.write(theme.accent);
    try term.write("> ");
    try term.write(theme.text);
    const query = self.search_buffer[0..self.search_len];
    const placeholder = "Type to filter commands";
    if (query.len > 0) {
        _ = try render_utils.writeClipped(term, query, inner_width -| 2);
    } else {
        try term.write(theme.text_muted);
        _ = try render_utils.writeClipped(term, placeholder, inner_width -| 2);
        try term.write(theme.text);
    }
    try term.write(theme.reset);
}

fn renderItems(self: *CommandPalette, term: *tui.Terminal, theme: *const tui.Theme, rect: tui.Rect) !void {
    const start_row = rect.y + 4;
    const visible = self.visible_rows;
    const inner_width = @as(usize, rect.width) -| 4;

    for (0..visible) |offset| {
        const row_y = start_row + @as(u16, @intCast(offset));
        try render_utils.moveTo(term, rect.x + 1, row_y);

        const absolute = self.scroll_offset + offset;
        if (absolute >= self.filtered_indices.items.len) {
            try term.write(theme.text_muted);
            try term.write(" ");
            try render_utils.writeRepeat(term, " ", @as(usize, rect.width) -| 2);
            try term.write(theme.reset);
            continue;
        }

        const item = &self.items[self.filtered_indices.items[absolute]];
        const selected = absolute == self.selected;

        try term.write(theme.border);
        try term.write(render_utils.boxChars(.double).v);
        if (selected) {
            try term.write(theme.selection_bg);
            try term.write(theme.selection_fg);
        } else {
            try term.write(theme.text);
        }
        try term.write(" ");
        const shortcut = if (item.shortcut) |value|
            std.fmt.allocPrint(self.allocator, "{d}", .{value}) catch ""
        else
            "";
        defer if (shortcut.len > 0) self.allocator.free(shortcut);

        var line_buf: [256]u8 = undefined;
        const prefix = if (selected) ">" else " ";
        const line = std.fmt.bufPrint(
            &line_buf,
            "{s} {s}{s}{s}  {s}",
            .{
                prefix,
                if (shortcut.len > 0) shortcut else "",
                if (shortcut.len > 0) "." else "",
                item.label,
                item.description,
            },
        ) catch item.label;
        try render_utils.writePadded(term, line, inner_width);
        try term.write(theme.reset);
        try term.write(theme.border);
        try term.write(render_utils.boxChars(.double).v);
        try term.write(theme.reset);
    }

    if (self.filtered_indices.items.len == 0) {
        try render_utils.moveTo(term, rect.x + 3, start_row);
        try term.write(theme.warning);
        try term.write("No matching commands.");
        try term.write(theme.reset);
    }
}

fn renderPreview(
    self: *const CommandPalette,
    term: *tui.Terminal,
    theme: *const tui.Theme,
    rect: tui.Rect,
    start_row: u16,
) !void {
    const inner_width = @as(usize, rect.width) -| 4;
    const item = self.selectedItem();
    const description = if (item) |it| it.description else "No command selected.";
    const usage = if (item) |it| it.usage else "Enter to run the selected command.";
    const footer = "Tab complete  Enter run  Esc close";

    try render_utils.moveTo(term, rect.x + 2, rect.y + start_row);
    try term.write(theme.text_muted);
    try render_utils.writePadded(term, description, inner_width);
    try term.write(theme.reset);

    try render_utils.moveTo(term, rect.x + 2, rect.y + start_row + 1);
    try term.write(theme.info);
    try render_utils.writePadded(term, usage, inner_width);
    try term.write(theme.reset);

    try render_utils.moveTo(term, rect.x + 2, rect.y + start_row + 2);
    try term.write(theme.text_dim);
    try render_utils.writePadded(term, footer, inner_width);
    try term.write(theme.reset);
}

fn containsRelated(related: []const []const u8, query: []const u8) bool {
    for (related) |entry| {
        if (completion.containsIgnoreCase(entry, query)) return true;
    }
    return false;
}

fn clampU16(value: u16, min_value: u16, max_value: u16) u16 {
    return @max(min_value, @min(value, max_value));
}

test "palette open and close reset query state" {
    var palette = try CommandPalette.init(std.testing.allocator);
    defer palette.deinit();

    palette.search_buffer[0] = 'x';
    palette.search_len = 1;
    try palette.open();
    try std.testing.expect(palette.active);
    try std.testing.expectEqual(@as(usize, 0), palette.search_len);

    palette.close();
    try std.testing.expect(!palette.active);
}

test "palette filters to ui editor" {
    var palette = try CommandPalette.init(std.testing.allocator);
    defer palette.deinit();
    try palette.open();

    for ("editor") |ch| {
        _ = try palette.handleKey(.{ .code = .character, .char = ch });
    }

    const item = palette.selectedItem() orelse return error.TestExpectedItem;
    try std.testing.expectEqualStrings("Editor", item.label);
}

test "palette enter submits selected command action" {
    var palette = try CommandPalette.init(std.testing.allocator);
    defer palette.deinit();
    try palette.open();

    for ("editor") |ch| {
        _ = try palette.handleKey(.{ .code = .character, .char = ch });
    }

    const outcome = try palette.handleKey(.{ .code = .enter });
    switch (outcome) {
        .submit => |action| switch (action) {
            .command => |cmd| try std.testing.expectEqualStrings("editor", cmd.id),
            else => return error.TestExpectedCommandAction,
        },
        else => return error.TestExpectedSubmit,
    }
}

test {
    std.testing.refAllDecls(@This());
}
