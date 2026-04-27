//! Double-buffered screen renderer.
//!
//! Maintains front and back buffers of terminal cells. On flush,
//! only emits ANSI sequences for cells that changed between frames.

const std = @import("std");
const types = @import("types.zig");
const ansi = @import("ansi.zig");
const Cell = types.Cell;
const Style = types.Style;
const Rect = types.Rect;

/// Double-buffered screen for efficient terminal rendering.
pub const Screen = struct {
    front: []Cell,
    back: []Cell,
    width: u16,
    height: u16,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: u16, height: u16) !Screen {
        const size = @as(usize, width) * @as(usize, height);
        const front = try allocator.alloc(Cell, size);
        const back = try allocator.alloc(Cell, size);
        @memset(front, Cell.blank);
        @memset(back, Cell.blank);
        return .{
            .front = front,
            .back = back,
            .width = width,
            .height = height,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Screen) void {
        self.allocator.free(self.front);
        self.allocator.free(self.back);
    }

    /// Clear the back buffer.
    pub fn clear(self: *Screen) void {
        @memset(self.back, Cell.blank);
    }

    /// Set a cell in the back buffer.
    pub fn setCell(self: *Screen, x: u16, y: u16, cell: Cell) void {
        if (x >= self.width or y >= self.height) return;
        self.back[@as(usize, y) * @as(usize, self.width) + @as(usize, x)] = cell;
    }

    /// Write a string to the back buffer with a given style.
    pub fn print(self: *Screen, x: u16, y: u16, text: []const u8, style: Style) void {
        var col = x;
        for (text) |byte| {
            if (col >= self.width) break;
            self.setCell(col, y, .{ .char = byte, .style = style });
            col += 1;
        }
    }

    /// Flush changes: emit ANSI only for cells that differ between front and back.
    pub fn flush(self: *Screen, writer: *std.Io.Writer) !void {
        var last_style: ?Style = null;

        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const idx = y * @as(usize, self.width) + x;
                const back_cell = self.back[idx];
                const front_cell = self.front[idx];

                if (std.meta.eql(back_cell, front_cell)) continue;

                try ansi.moveCursor(writer, @intCast(x), @intCast(y));

                if (last_style == null or !std.meta.eql(back_cell.style, last_style.?)) {
                    try ansi.setStyle(writer, back_cell.style);
                    last_style = back_cell.style;
                }

                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(back_cell.char, &buf) catch 1;
                try writer.writeAll(buf[0..len]);
            }
        }

        if (last_style != null) {
            try ansi.resetStyle(writer);
        }

        // Swap: copy back → front
        @memcpy(self.front, self.back);
    }

    /// Resize the screen buffers.
    pub fn resize(self: *Screen, width: u16, height: u16) !void {
        const new_size = @as(usize, width) * @as(usize, height);
        self.allocator.free(self.front);
        self.allocator.free(self.back);
        self.front = try self.allocator.alloc(Cell, new_size);
        self.back = try self.allocator.alloc(Cell, new_size);
        @memset(self.front, Cell.blank);
        @memset(self.back, Cell.blank);
        self.width = width;
        self.height = height;
    }

    /// Get the full screen rect.
    pub fn rect(self: *const Screen) Rect {
        return .{ .x = 0, .y = 0, .width = self.width, .height = self.height };
    }
};

test "Screen init and deinit" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();
    try std.testing.expectEqual(@as(u16, 80), screen.width);
    try std.testing.expectEqual(@as(u16, 24), screen.height);
}

test "Screen setCell and print" {
    var screen = try Screen.init(std.testing.allocator, 10, 5);
    defer screen.deinit();

    screen.setCell(0, 0, .{ .char = 'A', .style = .{} });
    try std.testing.expectEqual(@as(u21, 'A'), screen.back[0].char);

    screen.print(0, 1, "Hello", .{});
    try std.testing.expectEqual(@as(u21, 'H'), screen.back[10].char);
    try std.testing.expectEqual(@as(u21, 'o'), screen.back[14].char);
}

test "Screen setCell bounds check" {
    var screen = try Screen.init(std.testing.allocator, 10, 5);
    defer screen.deinit();

    // Out of bounds should not crash
    screen.setCell(100, 100, .{ .char = 'X', .style = .{} });
}

test "Screen clear resets back buffer" {
    var screen = try Screen.init(std.testing.allocator, 10, 5);
    defer screen.deinit();

    screen.print(0, 0, "Test", .{});
    screen.clear();
    try std.testing.expectEqual(@as(u21, ' '), screen.back[0].char);
}

test "Screen resize" {
    var screen = try Screen.init(std.testing.allocator, 10, 5);
    defer screen.deinit();

    try screen.resize(20, 10);
    try std.testing.expectEqual(@as(u16, 20), screen.width);
    try std.testing.expectEqual(@as(u16, 10), screen.height);
}

test "Screen flush writes changed cells" {
    var screen = try Screen.init(std.testing.allocator, 5, 2);
    defer screen.deinit();

    screen.setCell(0, 0, .{ .char = 'X', .style = .{} });

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try screen.flush(&writer);
    const written = buf[0..writer.end];
    // Should have emitted something for the changed cell
    try std.testing.expect(written.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, written, "X") != null);
}

test "Screen rect returns full dimensions" {
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();
    const r = screen.rect();
    try std.testing.expectEqual(@as(u16, 80), r.width);
    try std.testing.expectEqual(@as(u16, 24), r.height);
}

test {
    std.testing.refAllDecls(@This());
}
