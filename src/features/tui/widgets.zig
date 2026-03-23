//! TUI widget primitives.
//!
//! Widgets render themselves into a Screen buffer within a given Rect.

const std = @import("std");
const types = @import("types.zig");
const ansi_mod = @import("ansi.zig");
const render_mod = @import("render.zig");
const Screen = render_mod.Screen;
const Style = types.Style;
const Rect = types.Rect;
const Color = types.Color;

/// Render styled text within a rect (single line, truncated).
pub fn renderText(screen: *Screen, area: Rect, text: []const u8, style: Style) void {
    if (area.width == 0 or area.height == 0) return;
    const max_len = @min(text.len, @as(usize, area.width));
    screen.print(area.x, area.y, text[0..max_len], style);
}

/// Render a bordered panel with a title.
pub fn renderPanel(screen: *Screen, area: Rect, title: []const u8, border_style: Style) void {
    if (area.width < 2 or area.height < 2) return;

    // Top border with title
    screen.setCell(area.x, area.y, .{ .char = 0x250C, .style = border_style }); // ┌
    const inner_w = area.width -| 2;
    const title_len: u16 = @intCast(@min(title.len, inner_w));
    const pad_left = (inner_w -| title_len) / 2;

    var col: u16 = 1;
    while (col <= inner_w) : (col += 1) {
        const title_start = pad_left + 1;
        const title_end = title_start + title_len;
        if (col >= title_start and col < title_end) {
            const idx = col - title_start;
            screen.setCell(area.x + col, area.y, .{ .char = title[idx], .style = border_style });
        } else {
            screen.setCell(area.x + col, area.y, .{ .char = 0x2500, .style = border_style }); // ─
        }
    }
    screen.setCell(area.x + area.width -| 1, area.y, .{ .char = 0x2510, .style = border_style }); // ┐

    // Side borders
    var row: u16 = 1;
    while (row < area.height -| 1) : (row += 1) {
        screen.setCell(area.x, area.y + row, .{ .char = 0x2502, .style = border_style }); // │
        screen.setCell(area.x + area.width -| 1, area.y + row, .{ .char = 0x2502, .style = border_style }); // │
    }

    // Bottom border
    screen.setCell(area.x, area.y + area.height -| 1, .{ .char = 0x2514, .style = border_style }); // └
    col = 1;
    while (col <= inner_w) : (col += 1) {
        screen.setCell(area.x + col, area.y + area.height -| 1, .{ .char = 0x2500, .style = border_style }); // ─
    }
    screen.setCell(area.x + area.width -| 1, area.y + area.height -| 1, .{ .char = 0x2518, .style = border_style }); // ┘
}

/// Render a list of items with a highlighted selection.
pub fn renderList(screen: *Screen, area: Rect, items: []const []const u8, selected: usize, normal_style: Style, selected_style: Style) void {
    if (area.width < 2 or area.height == 0) return;

    for (0..@min(items.len, area.height)) |i| {
        const style = if (i == selected) selected_style else normal_style;
        const item = items[i];
        const max_len = @min(item.len, @as(usize, area.width));
        screen.print(area.x, area.y + @as(u16, @intCast(i)), item[0..max_len], style);

        // Fill remaining width for selected item highlight
        if (i == selected) {
            var col: u16 = @intCast(max_len);
            while (col < area.width) : (col += 1) {
                screen.setCell(area.x + col, area.y + @as(u16, @intCast(i)), .{ .char = ' ', .style = selected_style });
            }
        }
    }
}

/// Render a status bar at the bottom of a rect.
pub fn renderStatusBar(screen: *Screen, area: Rect, left: []const u8, right: []const u8, style: Style) void {
    if (area.width == 0 or area.height == 0) return;
    const y = area.y + area.height -| 1;

    // Fill entire row with style
    var col: u16 = 0;
    while (col < area.width) : (col += 1) {
        screen.setCell(area.x + col, y, .{ .char = ' ', .style = style });
    }

    // Left text
    const left_len = @min(left.len, @as(usize, area.width));
    screen.print(area.x, y, left[0..left_len], style);

    // Right text
    const right_len = @min(right.len, @as(usize, area.width));
    if (area.width >= right_len) {
        screen.print(area.x + area.width - @as(u16, @intCast(right_len)), y, right[0..right_len], style);
    }
}

/// Render a gauge/progress bar.
pub fn renderGauge(screen: *Screen, area: Rect, percent: u8, label: []const u8, filled_style: Style, empty_style: Style) void {
    if (area.width == 0 or area.height == 0) return;
    const clamped: u16 = @min(percent, 100);
    const filled_width: u16 = @intCast((@as(u32, area.width) * @as(u32, clamped)) / 100);

    var col: u16 = 0;
    while (col < area.width) : (col += 1) {
        const style = if (col < filled_width) filled_style else empty_style;
        const char: u21 = if (col < filled_width) 0x2588 else 0x2591; // █ vs ░
        screen.setCell(area.x + col, area.y, .{ .char = char, .style = style });
    }

    // Overlay label centered
    if (label.len > 0) {
        const label_len: u16 = @intCast(@min(label.len, area.width));
        const label_x = area.x + (area.width -| label_len) / 2;
        screen.print(label_x, area.y, label[0..label_len], filled_style);
    }
}

/// Render a simple table with headers and rows.
pub fn renderTable(screen: *Screen, area: Rect, headers: []const []const u8, rows: []const []const []const u8, header_style: Style, row_style: Style) void {
    if (area.width == 0 or area.height == 0 or headers.len == 0) return;

    const col_width: u16 = area.width / @as(u16, @intCast(headers.len));

    // Render headers
    for (headers, 0..) |header, i| {
        const x = area.x + @as(u16, @intCast(i)) * col_width;
        const max_len = @min(header.len, @as(usize, col_width));
        screen.print(x, area.y, header[0..max_len], header_style);
    }

    // Render rows
    for (rows, 0..) |row, row_idx| {
        const y = area.y + @as(u16, @intCast(row_idx)) + 1;
        if (y >= area.y + area.height) break;

        for (row, 0..) |cell, col_idx| {
            if (col_idx >= headers.len) break;
            const x = area.x + @as(u16, @intCast(col_idx)) * col_width;
            const max_len = @min(cell.len, @as(usize, col_width));
            screen.print(x, y, cell[0..max_len], row_style);
        }
    }
}

test "renderText basic" {
    var screen = try Screen.init(std.testing.allocator, 20, 5);
    defer screen.deinit();
    renderText(&screen, .{ .x = 0, .y = 0, .width = 20, .height = 1 }, "Hello", .{});
    try std.testing.expectEqual(@as(u21, 'H'), screen.back[0].char);
    try std.testing.expectEqual(@as(u21, 'o'), screen.back[4].char);
}

test "renderText truncates" {
    var screen = try Screen.init(std.testing.allocator, 5, 1);
    defer screen.deinit();
    renderText(&screen, .{ .x = 0, .y = 0, .width = 3, .height = 1 }, "Hello", .{});
    try std.testing.expectEqual(@as(u21, 'H'), screen.back[0].char);
    try std.testing.expectEqual(@as(u21, 'l'), screen.back[2].char);
    try std.testing.expectEqual(@as(u21, ' '), screen.back[3].char); // not written
}

test "renderPanel draws borders" {
    var screen = try Screen.init(std.testing.allocator, 10, 5);
    defer screen.deinit();
    renderPanel(&screen, .{ .x = 0, .y = 0, .width = 10, .height = 5 }, "Test", .{});
    try std.testing.expectEqual(@as(u21, 0x250C), screen.back[0].char); // ┌
    try std.testing.expectEqual(@as(u21, 0x2510), screen.back[9].char); // ┐
}

test "renderStatusBar fills row" {
    var screen = try Screen.init(std.testing.allocator, 20, 3);
    defer screen.deinit();
    const style = Style{ .bg = .blue };
    renderStatusBar(&screen, .{ .x = 0, .y = 0, .width = 20, .height = 3 }, "Left", "Right", style);
    // Last row should have the left text
    try std.testing.expectEqual(@as(u21, 'L'), screen.back[40].char);
}

test "renderGauge draws filled and empty" {
    var screen = try Screen.init(std.testing.allocator, 10, 1);
    defer screen.deinit();
    renderGauge(&screen, .{ .x = 0, .y = 0, .width = 10, .height = 1 }, 50, "", .{ .fg = .green }, .{ .fg = .white });
    try std.testing.expectEqual(@as(u21, 0x2588), screen.back[0].char); // █
    try std.testing.expectEqual(@as(u21, 0x2591), screen.back[5].char); // ░
}

test {
    std.testing.refAllDecls(@This());
}
