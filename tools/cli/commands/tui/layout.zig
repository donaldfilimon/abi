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

test "clickedIndexFromRow scroll offset and boundary edge cases" {
    // scroll_offset > 0: first visible item is at index 3
    try std.testing.expectEqual(@as(?usize, 3), clickedIndexFromRow(10, 10, false, 3, 5, 20));
    try std.testing.expectEqual(@as(?usize, 7), clickedIndexFromRow(14, 10, false, 3, 5, 20));
    // visible_rows > filtered_len: only filtered_len items exist
    try std.testing.expectEqual(@as(?usize, 0), clickedIndexFromRow(10, 10, false, 0, 10, 1));
    try std.testing.expectEqual(@as(?usize, null), clickedIndexFromRow(11, 10, false, 0, 10, 1));
    // first_item_row boundary: click exactly at first item row
    try std.testing.expectEqual(@as(?usize, 0), clickedIndexFromRow(10, 10, false, 0, 1, 5));
}

test "computeVisibleRows rows >= 11" {
    try std.testing.expectEqual(@as(usize, 13), computeVisibleRows(24));
    try std.testing.expectEqual(@as(usize, 5), computeVisibleRows(16));
}

test "computeVisibleRows rows < 11 and very small terminals" {
    try std.testing.expectEqual(@as(usize, 5), computeVisibleRows(11));
    try std.testing.expectEqual(@as(usize, 4), computeVisibleRows(10));
    try std.testing.expectEqual(@as(usize, 1), computeVisibleRows(7));
    try std.testing.expectEqual(@as(usize, 1), computeVisibleRows(6));
}

test "clampedFrameWidth" {
    try std.testing.expectEqual(@as(usize, 40), clampedFrameWidth(30));
    try std.testing.expectEqual(@as(usize, 40), clampedFrameWidth(40));
    try std.testing.expectEqual(@as(usize, 80), clampedFrameWidth(80));
}

test "completionDropdownRowCount inactive" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    const state = struct {
        search_mode: bool = false,
        completion_state: struct {
            active: bool = false,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .completion_state = .{
            .suggestions = suggestions,
            .max_visible = 5,
        },
    };
    try std.testing.expectEqual(@as(u16, 0), completionDropdownRowCount(state));
}

test "completionDropdownRowCount active with 0 suggestions" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    const state = struct {
        search_mode: bool = true,
        completion_state: struct {
            active: bool = true,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .completion_state = .{
            .suggestions = suggestions,
            .max_visible = 5,
        },
    };
    try std.testing.expectEqual(@as(u16, 2), completionDropdownRowCount(state));
}

test "completionDropdownRowCount active suggestions less than max_visible" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    try suggestions.append(allocator, 1);
    try suggestions.append(allocator, 2);
    const state = struct {
        search_mode: bool = true,
        completion_state: struct {
            active: bool = true,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .completion_state = .{
            .suggestions = suggestions,
            .max_visible = 5,
        },
    };
    try std.testing.expectEqual(@as(u16, 4), completionDropdownRowCount(state));
}

test "completionDropdownRowCount active suggestions >= max_visible" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    for (0..10) |_| try suggestions.append(allocator, 0);
    const state = struct {
        search_mode: bool = true,
        completion_state: struct {
            active: bool = true,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .completion_state = .{
            .suggestions = suggestions,
            .max_visible = 5,
        },
    };
    try std.testing.expectEqual(@as(u16, 7), completionDropdownRowCount(state));
}

test "menuStartRow no notification no search no history" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);
    const state = struct {
        notification: ?[]const u8 = null,
        search_mode: bool = false,
        search_len: usize = 0,
        show_history: bool = false,
        history: std.ArrayListUnmanaged(u8),
        term_size: struct { rows: u16 = 24 },
        completion_state: struct {
            active: bool = false,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .history = history,
        .term_size = .{ .rows = 24 },
        .completion_state = .{ .suggestions = suggestions },
    };
    try std.testing.expectEqual(@as(u16, 4), menuStartRow(state));
}

test "menuStartRow with notification" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);
    const state = struct {
        notification: ?[]const u8 = "hi",
        search_mode: bool = false,
        search_len: usize = 0,
        show_history: bool = false,
        history: std.ArrayListUnmanaged(u8),
        term_size: struct { rows: u16 = 24 },
        completion_state: struct {
            active: bool = false,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .history = history,
        .term_size = .{ .rows = 24 },
        .completion_state = .{ .suggestions = suggestions },
    };
    try std.testing.expectEqual(@as(u16, 5), menuStartRow(state));
}

test "menuStartRow with search and completion" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    try suggestions.append(allocator, 0);
    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);
    const state = struct {
        notification: ?[]const u8 = null,
        search_mode: bool = true,
        search_len: usize = 1,
        show_history: bool = false,
        history: std.ArrayListUnmanaged(u8),
        term_size: struct { rows: u16 = 24 },
        completion_state: struct {
            active: bool = true,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .history = history,
        .term_size = .{ .rows = 24 },
        .completion_state = .{ .suggestions = suggestions, .max_visible = 5 },
    };
    // 4 + 2 (search) + (1 + 2) completion = 9
    try std.testing.expectEqual(@as(u16, 9), menuStartRow(state));
}

test "menuStartRow with history" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);
    try history.append(allocator, 0);
    try history.append(allocator, 0);
    const state = struct {
        notification: ?[]const u8 = null,
        search_mode: bool = false,
        search_len: usize = 0,
        show_history: bool = true,
        history: std.ArrayListUnmanaged(u8),
        term_size: struct { rows: u16 = 24 },
        completion_state: struct {
            active: bool = false,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .history = history,
        .term_size = .{ .rows = 24 },
        .completion_state = .{ .suggestions = suggestions },
    };
    // 4 + (2 + 2) history = 8
    try std.testing.expectEqual(@as(u16, 8), menuStartRow(state));
}

test "menuStartRow clamped to term_size.rows - 3" {
    const allocator = std.testing.allocator;
    var suggestions = std.ArrayListUnmanaged(u8).empty;
    defer suggestions.deinit(allocator);
    for (0..10) |_| try suggestions.append(allocator, 0);
    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);
    for (0..10) |_| try history.append(allocator, 0);
    const state = struct {
        notification: ?[]const u8 = null,
        search_mode: bool = true,
        search_len: usize = 1,
        show_history: bool = true,
        history: std.ArrayListUnmanaged(u8),
        term_size: struct { rows: u16 },
        completion_state: struct {
            active: bool = true,
            suggestions: std.ArrayListUnmanaged(u8),
            max_visible: usize = 5,
        },
    }{
        .history = history,
        .term_size = .{ .rows = 12 },
        .completion_state = .{ .suggestions = suggestions, .max_visible = 5 },
    };
    try std.testing.expectEqual(@as(u16, 9), menuStartRow(state));
}
