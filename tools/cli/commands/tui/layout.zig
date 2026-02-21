//! TUI layout helpers.

const std = @import("std");

pub const CompactLayout = struct {
    min_width: usize = 40,
    max_width: usize = 140,
};

pub fn computeVisibleRows(rows: u16) usize {
    const content_rows = rows -| 10;
    return @max(@as(usize, 5), @as(usize, @intCast(content_rows)));
}

pub fn clampedFrameWidth(cols: u16) usize {
    const value = @as(usize, @intCast(cols));
    return std.math.clamp(value, @as(usize, 40), @as(usize, 140));
}

pub fn completionDropdownRowCount(state: anytype) u16 {
    if (!state.search_mode or !state.completion.active) return 0;
    const suggestions_len = state.completion.suggestions.items.len;
    const shown = @min(suggestions_len, state.completion.max_visible);
    return @as(u16, @intCast(shown + 2));
}

pub fn menuStartRow(state: anytype) u16 {
    var row: u16 = 3;
    if (state.notification != null) row += 1;

    if (state.search_mode or state.search_len > 0) {
        row += 2;
        row += completionDropdownRowCount(state);
    }

    if (state.show_history and state.history.items.len > 0) {
        const shown = @min(state.history.items.len, 5);
        row += @as(u16, @intCast(shown + 2));
    }

    return row;
}

pub fn clickedIndexFromRow(
    row0: u16,
    menu_start_row: u16,
    show_top_indicator: bool,
    scroll_offset: usize,
    visible_rows: usize,
    filtered_len: usize,
) ?usize {
    if (filtered_len == 0 or row0 < menu_start_row) return null;

    var first_item_row = menu_start_row;
    if (show_top_indicator) {
        if (row0 == menu_start_row) return null;
        first_item_row += 1;
    }
    if (row0 < first_item_row) return null;

    const menu_row: usize = @intCast(row0 - first_item_row);
    if (menu_row >= visible_rows) return null;

    const clicked_idx = scroll_offset + menu_row;
    if (clicked_idx >= filtered_len) return null;
    return clicked_idx;
}
