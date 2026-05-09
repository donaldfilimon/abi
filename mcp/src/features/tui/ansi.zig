//! ANSI escape code generation.
//!
//! Pure functions that write ANSI escape sequences to a writer.
//! No I/O or terminal state — just string generation.

const std = @import("std");
const types = @import("types.zig");
const Color = types.Color;
const Style = types.Style;
const Rect = types.Rect;

const ESC = "\x1b[";

/// Move cursor to position (1-indexed for ANSI).
pub fn moveCursor(writer: *std.Io.Writer, x: u16, y: u16) !void {
    try writer.print(ESC ++ "{d};{d}H", .{ y + 1, x + 1 });
}

/// Clear the entire screen.
pub fn clearScreen(writer: *std.Io.Writer) !void {
    try writer.writeAll(ESC ++ "2J");
}

/// Clear the current line.
pub fn clearLine(writer: *std.Io.Writer) !void {
    try writer.writeAll(ESC ++ "2K");
}

/// Hide the cursor.
pub fn hideCursor(writer: *std.Io.Writer) !void {
    try writer.writeAll(ESC ++ "?25l");
}

/// Show the cursor.
pub fn showCursor(writer: *std.Io.Writer) !void {
    try writer.writeAll(ESC ++ "?25h");
}

/// Reset all styles to default.
pub fn resetStyle(writer: *std.Io.Writer) !void {
    try writer.writeAll(ESC ++ "0m");
}

/// Apply a Style via SGR escape sequences.
pub fn setStyle(writer: *std.Io.Writer, style: Style) !void {
    try writer.writeAll(ESC ++ "0"); // reset first

    if (style.bold) try writer.writeAll(";1");
    if (style.dim) try writer.writeAll(";2");
    if (style.italic) try writer.writeAll(";3");
    if (style.underline) try writer.writeAll(";4");
    if (style.reverse) try writer.writeAll(";7");
    if (style.strikethrough) try writer.writeAll(";9");

    // Foreground color
    try writeColor(writer, style.fg, false);
    // Background color
    try writeColor(writer, style.bg, true);

    try writer.writeAll("m");
}

fn writeColor(writer: *std.Io.Writer, color: Color, is_bg: bool) !void {
    switch (color) {
        .default => {},
        .ansi256 => |code| {
            if (is_bg) {
                try writer.print(";48;5;{d}", .{code});
            } else {
                try writer.print(";38;5;{d}", .{code});
            }
        },
        .rgb => |c| {
            if (is_bg) {
                try writer.print(";48;2;{d};{d};{d}", .{ c.r, c.g, c.b });
            } else {
                try writer.print(";38;2;{d};{d};{d}", .{ c.r, c.g, c.b });
            }
        },
        else => {
            if (is_bg) {
                if (color.bgCode()) |code| {
                    try writer.print(";{d}", .{code});
                }
            } else {
                if (color.fgCode()) |code| {
                    try writer.print(";{d}", .{code});
                }
            }
        },
    }
}

/// Draw a box with Unicode box-drawing characters.
pub fn drawBox(writer: *std.Io.Writer, rect: Rect, style: Style) !void {
    if (rect.width < 2 or rect.height < 2) return;

    try setStyle(writer, style);

    // Top border
    try moveCursor(writer, rect.x, rect.y);
    try writer.writeAll("┌");
    var i: u16 = 0;
    while (i < rect.width -| 2) : (i += 1) {
        try writer.writeAll("─");
    }
    try writer.writeAll("┐");

    // Side borders
    var row: u16 = 1;
    while (row < rect.height -| 1) : (row += 1) {
        try moveCursor(writer, rect.x, rect.y + row);
        try writer.writeAll("│");
        try moveCursor(writer, rect.x + rect.width -| 1, rect.y + row);
        try writer.writeAll("│");
    }

    // Bottom border
    try moveCursor(writer, rect.x, rect.y + rect.height -| 1);
    try writer.writeAll("└");
    i = 0;
    while (i < rect.width -| 2) : (i += 1) {
        try writer.writeAll("─");
    }
    try writer.writeAll("┘");

    try resetStyle(writer);
}

/// Draw a box with a centered title in the top border.
pub fn drawBoxWithTitle(writer: *std.Io.Writer, rect: Rect, title: []const u8, style: Style) !void {
    if (rect.width < 2 or rect.height < 2) return;

    try setStyle(writer, style);

    // Top border with title
    try moveCursor(writer, rect.x, rect.y);
    try writer.writeAll("┌");

    const inner_width = rect.width -| 2;
    const title_len: u16 = @intCast(@min(title.len, inner_width));
    const pad_left = (inner_width -| title_len) / 2;
    const pad_right = inner_width -| title_len -| pad_left;

    var j: u16 = 0;
    while (j < pad_left) : (j += 1) try writer.writeAll("─");
    if (title_len > 0) try writer.writeAll(title[0..title_len]);
    j = 0;
    while (j < pad_right) : (j += 1) try writer.writeAll("─");
    try writer.writeAll("┐");

    // Side borders
    var row: u16 = 1;
    while (row < rect.height -| 1) : (row += 1) {
        try moveCursor(writer, rect.x, rect.y + row);
        try writer.writeAll("│");
        try moveCursor(writer, rect.x + rect.width -| 1, rect.y + row);
        try writer.writeAll("│");
    }

    // Bottom border
    try moveCursor(writer, rect.x, rect.y + rect.height -| 1);
    try writer.writeAll("└");
    j = 0;
    while (j < inner_width) : (j += 1) try writer.writeAll("─");
    try writer.writeAll("┘");

    try resetStyle(writer);
}

// ── Tests ───────────────────────────────────────────────────────────────

test "moveCursor generates correct sequence" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try moveCursor(&writer, 5, 10);
    try std.testing.expectEqualStrings("\x1b[11;6H", buf[0..writer.end]);
}

test "clearScreen generates correct sequence" {
    var buf: [16]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try clearScreen(&writer);
    try std.testing.expectEqualStrings("\x1b[2J", buf[0..writer.end]);
}

test "hideCursor and showCursor" {
    var buf: [32]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try hideCursor(&writer);
    try std.testing.expectEqualStrings("\x1b[?25l", buf[0..writer.end]);

    writer = std.Io.Writer.fixed(&buf);
    try showCursor(&writer);
    try std.testing.expectEqualStrings("\x1b[?25h", buf[0..writer.end]);
}

test "resetStyle generates reset sequence" {
    var buf: [16]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try resetStyle(&writer);
    try std.testing.expectEqualStrings("\x1b[0m", buf[0..writer.end]);
}

test "setStyle with bold" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try setStyle(&writer, .{ .bold = true });
    const written = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, ";1") != null);
    try std.testing.expect(written[written.len - 1] == 'm');
}

test "setStyle with color" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try setStyle(&writer, .{ .fg = .red });
    const written = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, ";31") != null);
}

test "setStyle with rgb color" {
    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try setStyle(&writer, .{ .fg = .{ .rgb = .{ .r = 255, .g = 128, .b = 0 } } });
    const written = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, ";38;2;255;128;0") != null);
}

test "setStyle with ansi256 color" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try setStyle(&writer, .{ .bg = .{ .ansi256 = 42 } });
    const written = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, ";48;5;42") != null);
}

test "drawBox generates box characters" {
    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try drawBox(&writer, .{ .x = 0, .y = 0, .width = 5, .height = 3 }, .{});
    const written = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, written, "┌") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "┐") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "└") != null);
    try std.testing.expect(std.mem.indexOf(u8, written, "┘") != null);
}

test "drawBox skips tiny rects" {
    var buf: [64]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try drawBox(&writer, .{ .x = 0, .y = 0, .width = 1, .height = 1 }, .{});
    try std.testing.expectEqual(@as(usize, 0), writer.end);
}

test {
    std.testing.refAllDecls(@This());
}
