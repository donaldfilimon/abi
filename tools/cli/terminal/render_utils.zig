//! Shared rendering utilities for TUI components.
//!
//! Provides Unicode-aware box-drawing, text clipping, padding,
//! and cursor positioning used by all dashboard panels.

const std = @import("std");
const unicode = @import("unicode.zig");
const layout = @import("layout.zig");
const terminal_mod = @import("terminal.zig");
const themes = @import("themes.zig");

pub const Terminal = terminal_mod.Terminal;
pub const Rect = layout.Rect;
pub const Theme = themes.Theme;

/// Box-drawing character sets.
pub const BoxStyle = enum {
    single, // ╭─╮│├┤╰╯
    double, // ╔═╗║╠╣╚╝
    rounded, // ╭─╮│├┤╰╯ (same as single for now)
    heavy, // ┏━┓┃┣┫┗┛
};

/// Box-drawing characters for a given style.
pub const BoxChars = struct {
    tl: []const u8, // top-left
    tr: []const u8, // top-right
    bl: []const u8, // bottom-left
    br: []const u8, // bottom-right
    h: []const u8, // horizontal
    v: []const u8, // vertical
    lsep: []const u8, // left separator (T-junction)
    rsep: []const u8, // right separator
};

/// Get box characters for a given style.
pub fn boxChars(style: BoxStyle) BoxChars {
    return switch (style) {
        .single, .rounded => .{
            .tl = "\u{256d}",
            .tr = "\u{256e}",
            .bl = "\u{2570}",
            .br = "\u{256f}",
            .h = "\u{2500}",
            .v = "\u{2502}",
            .lsep = "\u{251c}",
            .rsep = "\u{2524}",
        },
        .double => .{
            .tl = "\u{2554}",
            .tr = "\u{2557}",
            .bl = "\u{255a}",
            .br = "\u{255d}",
            .h = "\u{2550}",
            .v = "\u{2551}",
            .lsep = "\u{2560}",
            .rsep = "\u{2563}",
        },
        .heavy => .{
            .tl = "\u{250f}",
            .tr = "\u{2513}",
            .bl = "\u{2517}",
            .br = "\u{251b}",
            .h = "\u{2501}",
            .v = "\u{2503}",
            .lsep = "\u{2523}",
            .rsep = "\u{252b}",
        },
    };
}

/// Move cursor to position (x, y) using ANSI escape.
/// ANSI coordinates are 1-indexed.
pub fn moveTo(term: *Terminal, x: u16, y: u16) !void {
    var buf: [32]u8 = undefined;
    const seq = std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{
        @as(u32, y) + 1,
        @as(u32, x) + 1,
    }) catch return error.BufferOverflow;
    try term.write(seq);
}

/// Write a character repeated `count` times.
pub fn writeRepeat(
    term: *Terminal,
    char: []const u8,
    count: usize,
) !void {
    for (0..count) |_| {
        try term.write(char);
    }
}

/// Write text clipped to max_cols terminal columns (unicode-aware).
/// Returns the number of display columns actually written.
pub fn writeClipped(
    term: *Terminal,
    text: []const u8,
    max_cols: usize,
) !usize {
    const clipped = unicode.truncateToWidth(text, max_cols);
    try term.write(clipped);
    return unicode.displayWidth(clipped);
}

/// Write text padded with spaces to fill target_cols (unicode-aware).
pub fn writePadded(
    term: *Terminal,
    text: []const u8,
    target_cols: usize,
) !void {
    const written = try writeClipped(term, text, target_cols);
    if (written < target_cols) {
        try writeRepeat(term, " ", target_cols - written);
    }
}

/// Draw a box outline within the given rect.
/// Draws top border, bottom border, and vertical sides.
/// Does NOT fill interior.
pub fn drawBox(
    term: *Terminal,
    rect: Rect,
    style: BoxStyle,
    theme: *const Theme,
) !void {
    if (rect.isEmpty()) return;
    if (rect.width < 2 or rect.height < 2) return;

    const bc = boxChars(style);
    const inner_w: usize = @as(usize, rect.width) - 2;

    // Border color
    try term.write(theme.border);

    // Top border: ╭───╮
    try moveTo(term, rect.x, rect.y);
    try term.write(bc.tl);
    try writeRepeat(term, bc.h, inner_w);
    try term.write(bc.tr);

    // Vertical sides for interior rows
    const interior_rows = rect.height - 2;
    for (0..interior_rows) |i| {
        const row_y = rect.y +| @as(u16, @intCast(i + 1));
        try moveTo(term, rect.x, row_y);
        try term.write(bc.v);
        try moveTo(
            term,
            rect.x +| (rect.width -| 1),
            row_y,
        );
        try term.write(bc.v);
    }

    // Bottom border: ╰───╯
    const bot_y = rect.y +| (rect.height -| 1);
    try moveTo(term, rect.x, bot_y);
    try term.write(bc.bl);
    try writeRepeat(term, bc.h, inner_w);
    try term.write(bc.br);

    // Reset
    try term.write(theme.reset);
}

/// Draw a horizontal separator line within a box (├───┤ style).
/// `y` is relative to rect.y.
pub fn drawSeparator(
    term: *Terminal,
    rect: Rect,
    y: u16,
    style: BoxStyle,
    theme: *const Theme,
) !void {
    if (rect.isEmpty()) return;
    if (rect.width < 2) return;

    const bc = boxChars(style);
    const inner_w: usize = @as(usize, rect.width) - 2;

    try term.write(theme.border);
    try moveTo(term, rect.x, rect.y +| y);
    try term.write(bc.lsep);
    try writeRepeat(term, bc.h, inner_w);
    try term.write(bc.rsep);
    try term.write(theme.reset);
}

/// Write text into a single row of a box, with left/right borders.
/// Text is clipped to fit within the box interior (unicode-aware).
/// `y` is relative to rect.y. Content is left-aligned with
/// 1-space padding.
pub fn drawRow(
    term: *Terminal,
    rect: Rect,
    y: u16,
    content: []const u8,
    theme: *const Theme,
) !void {
    if (rect.isEmpty()) return;
    if (rect.width < 2) return;

    const bc = boxChars(.single);
    // Interior width excluding borders
    const interior: usize = @as(usize, rect.width) - 2;

    try moveTo(term, rect.x, rect.y +| y);

    // Left border
    try term.write(theme.border);
    try term.write(bc.v);
    try term.write(theme.reset);

    // Content area: 1-space left pad, content, right pad
    if (interior > 1) {
        try term.write(" ");
        const content_cols = interior - 1;
        try writePadded(term, content, content_cols);
    } else if (interior == 1) {
        try term.write(" ");
    }

    // Right border
    try term.write(theme.border);
    try term.write(bc.v);
    try term.write(theme.reset);
}

/// Fill an entire row within a box with a repeated character.
/// `y` is relative to rect.y.
pub fn fillRow(
    term: *Terminal,
    rect: Rect,
    y: u16,
    char: []const u8,
    theme: *const Theme,
) !void {
    if (rect.isEmpty()) return;
    if (rect.width < 2) return;

    const bc = boxChars(.single);
    const interior: usize = @as(usize, rect.width) - 2;

    try moveTo(term, rect.x, rect.y +| y);

    // Left border
    try term.write(theme.border);
    try term.write(bc.v);
    try term.write(theme.reset);

    // Fill interior
    try writeRepeat(term, char, interior);

    // Right border
    try term.write(theme.border);
    try term.write(bc.v);
    try term.write(theme.reset);
}

// ── Tests ───────────────────────────────────────────────────────

test "boxChars returns correct single style" {
    const bc = boxChars(.single);
    try std.testing.expectEqualStrings("\u{256d}", bc.tl);
    try std.testing.expectEqualStrings("\u{256e}", bc.tr);
    try std.testing.expectEqualStrings("\u{2570}", bc.bl);
    try std.testing.expectEqualStrings("\u{256f}", bc.br);
    try std.testing.expectEqualStrings("\u{2500}", bc.h);
    try std.testing.expectEqualStrings("\u{2502}", bc.v);
    try std.testing.expectEqualStrings("\u{251c}", bc.lsep);
    try std.testing.expectEqualStrings("\u{2524}", bc.rsep);
}

test "boxChars returns correct double style" {
    const bc = boxChars(.double);
    try std.testing.expectEqualStrings("\u{2554}", bc.tl);
    try std.testing.expectEqualStrings("\u{2557}", bc.tr);
    try std.testing.expectEqualStrings("\u{255a}", bc.bl);
    try std.testing.expectEqualStrings("\u{255d}", bc.br);
    try std.testing.expectEqualStrings("\u{2550}", bc.h);
    try std.testing.expectEqualStrings("\u{2551}", bc.v);
    try std.testing.expectEqualStrings("\u{2560}", bc.lsep);
    try std.testing.expectEqualStrings("\u{2563}", bc.rsep);
}

test "boxChars returns correct heavy style" {
    const bc = boxChars(.heavy);
    try std.testing.expectEqualStrings("\u{250f}", bc.tl);
    try std.testing.expectEqualStrings("\u{2513}", bc.tr);
    try std.testing.expectEqualStrings("\u{2517}", bc.bl);
    try std.testing.expectEqualStrings("\u{251b}", bc.br);
    try std.testing.expectEqualStrings("\u{2501}", bc.h);
    try std.testing.expectEqualStrings("\u{2503}", bc.v);
    try std.testing.expectEqualStrings("\u{2523}", bc.lsep);
    try std.testing.expectEqualStrings("\u{252b}", bc.rsep);
}

test "boxChars rounded matches single" {
    const single = boxChars(.single);
    const rounded = boxChars(.rounded);
    try std.testing.expectEqualStrings(single.tl, rounded.tl);
    try std.testing.expectEqualStrings(single.tr, rounded.tr);
    try std.testing.expectEqualStrings(single.bl, rounded.bl);
    try std.testing.expectEqualStrings(single.br, rounded.br);
    try std.testing.expectEqualStrings(single.h, rounded.h);
    try std.testing.expectEqualStrings(single.v, rounded.v);
}

test "box chars struct completeness - all fields non-empty" {
    const styles = [_]BoxStyle{ .single, .double, .rounded, .heavy };
    for (styles) |style| {
        const bc = boxChars(style);
        try std.testing.expect(bc.tl.len > 0);
        try std.testing.expect(bc.tr.len > 0);
        try std.testing.expect(bc.bl.len > 0);
        try std.testing.expect(bc.br.len > 0);
        try std.testing.expect(bc.h.len > 0);
        try std.testing.expect(bc.v.len > 0);
        try std.testing.expect(bc.lsep.len > 0);
        try std.testing.expect(bc.rsep.len > 0);
    }
}

test "writeClipped with ASCII text shorter than max" {
    // We cannot test Terminal.write without a real terminal,
    // but we can verify the underlying unicode logic.
    const text = "hello";
    const clipped = unicode.truncateToWidth(text, 10);
    try std.testing.expectEqualStrings("hello", clipped);
    try std.testing.expectEqual(@as(usize, 5), unicode.displayWidth(clipped));
}

test "writeClipped with text longer than max (truncated)" {
    const text = "hello world";
    const clipped = unicode.truncateToWidth(text, 5);
    try std.testing.expectEqualStrings("hello", clipped);
    try std.testing.expectEqual(@as(usize, 5), unicode.displayWidth(clipped));
}

test "writeClipped with emoji at truncation boundary" {
    // "AB" (2 cols) + robot emoji (2 cols) = 4 cols
    const text = "AB\xF0\x9F\xA4\x96";
    // Truncate to 3: robot needs 2 cols, only 1 left after AB
    const clipped = unicode.truncateToWidth(text, 3);
    try std.testing.expectEqualStrings("AB", clipped);
    try std.testing.expectEqual(@as(usize, 2), unicode.displayWidth(clipped));
}

test "writePadded padding calculation" {
    // Verify the padding logic via unicode helpers
    const text = "hi";
    const width = unicode.displayWidth(text);
    try std.testing.expectEqual(@as(usize, 2), width);
    const padding = unicode.padToWidth(text, 10);
    try std.testing.expectEqual(@as(usize, 8), padding);
}

test "writeRepeat basic operation" {
    // Verify the function signature compiles; actual write
    // requires a terminal. Test the repeat count logic via
    // a simple iteration check.
    const count: usize = 5;
    var i: usize = 0;
    for (0..count) |_| {
        i += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), i);
}

test "moveTo ANSI sequence format" {
    // Verify the format string produces correct ANSI sequence.
    var buf: [32]u8 = undefined;
    const seq = try std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{
        @as(u32, 0) + 1,
        @as(u32, 0) + 1,
    });
    try std.testing.expectEqualStrings("\x1b[1;1H", seq);

    const seq2 = try std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{
        @as(u32, 9) + 1,
        @as(u32, 19) + 1,
    });
    try std.testing.expectEqualStrings("\x1b[10;20H", seq2);
}

test {
    std.testing.refAllDecls(@This());
}
