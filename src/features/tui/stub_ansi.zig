const std = @import("std");
const types = @import("types.zig");

pub fn moveCursor(writer: *std.Io.Writer, x: u16, y: u16) !void {
    _ = writer;
    _ = x;
    _ = y;
}
pub fn clearScreen(writer: *std.Io.Writer) !void {
    _ = writer;
}
pub fn clearLine(writer: *std.Io.Writer) !void {
    _ = writer;
}
pub fn hideCursor(writer: *std.Io.Writer) !void {
    _ = writer;
}
pub fn showCursor(writer: *std.Io.Writer) !void {
    _ = writer;
}
pub fn resetStyle(writer: *std.Io.Writer) !void {
    _ = writer;
}
pub fn setStyle(writer: *std.Io.Writer, style: types.Style) !void {
    _ = writer;
    _ = style;
}
pub fn drawBox(writer: *std.Io.Writer, rect: types.Rect, style: types.Style) !void {
    _ = writer;
    _ = rect;
    _ = style;
}
pub fn drawBoxWithTitle(writer: *std.Io.Writer, rect: types.Rect, title: []const u8, style: types.Style) !void {
    _ = writer;
    _ = rect;
    _ = title;
    _ = style;
}
