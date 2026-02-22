//! TUI layout helpers.

const std = @import("std");

pub const CompactLayout = struct {
    min_width: usize = 40,
};

pub fn computeVisibleRows(rows: u16) usize {
    const content_rows = rows -| 11;
    // On very small terminals, don't promise more rows than physically exist
    const min_rows: usize = if (rows >= 11) 5 else @max(1, @as(usize, rows) -| 6);
    return @max(min_rows, @as(usize, @intCast(content_rows)));
}

pub fn clampedFrameWidth(cols: u16) usize {
    return @max(@as(usize, 40), @as(usize, @intCast(cols)));
}

pub fn completionDropdownRowCount(state: anytype) u16 {
    if (!state.search_mode or !state.completion_state.active) return 0;
    const suggestions_len = state.completion_state.suggestions.items.len;
    const shown = @min(suggestions_len, state.completion_state.max_visible);
    return @as(u16, @intCast(shown + 2));
}

pub fn menuStartRow(state: anytype) u16 {
    var row: u16 = 4;
    if (state.notification != null) row += 1;

    if (state.search_mode or state.search_len > 0) {
        row += 2;
        row += completionDropdownRowCount(state);
    }

    if (state.show_history and state.history.items.len > 0) {
        const shown = @min(state.history.items.len, 5);
        row += @as(u16, @intCast(shown + 2));
    }

    // Don't exceed terminal height minus footer space
    const max_row = state.term_size.rows -| 3;
    return @min(row, max_row);
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

test "clicked row mapping handles dynamic header and scroll indicator" {
    // No top indicator: first menu item starts at menu_start_row.
    try std.testing.expectEqual(@as(?usize, 0), clickedIndexFromRow(10, 10, false, 0, 5, 20));
    try std.testing.expectEqual(@as(?usize, 4), clickedIndexFromRow(14, 10, false, 0, 5, 20));

    // With top indicator: row at menu_start_row is indicator, first item is +1.
    try std.testing.expectEqual(@as(?usize, null), clickedIndexFromRow(10, 10, true, 8, 5, 20));
    try std.testing.expectEqual(@as(?usize, 8), clickedIndexFromRow(11, 10, true, 8, 5, 20));

    // Out of bounds.
    try std.testing.expectEqual(@as(?usize, null), clickedIndexFromRow(20, 10, false, 0, 5, 3));
}
