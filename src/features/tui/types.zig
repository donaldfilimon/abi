//! Shared TUI types used by both mod.zig and stub.zig.

const std = @import("std");

pub const TuiError = error{
    TerminalInitFailed,
    UnsupportedTerminal,
    RenderFailed,
    InputError,
    FeatureDisabled,
};

pub const Error = TuiError || std.mem.Allocator.Error;

/// Terminal color specification.
pub const Color = union(enum) {
    default,
    black,
    red,
    green,
    yellow,
    blue,
    magenta,
    cyan,
    white,
    bright_black,
    bright_red,
    bright_green,
    bright_yellow,
    bright_blue,
    bright_magenta,
    bright_cyan,
    bright_white,
    ansi256: u8,
    rgb: struct { r: u8, g: u8, b: u8 },

    /// Return the ANSI SGR foreground code for named colors.
    pub fn fgCode(self: Color) ?u8 {
        return switch (self) {
            .default => null,
            .black => 30,
            .red => 31,
            .green => 32,
            .yellow => 33,
            .blue => 34,
            .magenta => 35,
            .cyan => 36,
            .white => 37,
            .bright_black => 90,
            .bright_red => 91,
            .bright_green => 92,
            .bright_yellow => 93,
            .bright_blue => 94,
            .bright_magenta => 95,
            .bright_cyan => 96,
            .bright_white => 97,
            .ansi256, .rgb => null,
        };
    }

    /// Return the ANSI SGR background code for named colors.
    pub fn bgCode(self: Color) ?u8 {
        const fg = self.fgCode() orelse return null;
        return fg + 10;
    }
};

/// Text style attributes.
pub const Style = struct {
    fg: Color = .default,
    bg: Color = .default,
    bold: bool = false,
    italic: bool = false,
    underline: bool = false,
    dim: bool = false,
    strikethrough: bool = false,
    reverse: bool = false,

    pub const default = Style{};

    pub fn withFg(color: Color) Style {
        return .{ .fg = color };
    }

    pub fn withBold() Style {
        return .{ .bold = true };
    }
};

/// Keyboard input events.
pub const Key = union(enum) {
    char: u21,
    enter,
    escape,
    backspace,
    tab,
    delete,
    up,
    down,
    left,
    right,
    home,
    end,
    page_up,
    page_down,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    ctrl: u8,
    alt: u8,
};

/// Input events from the terminal.
pub const Event = union(enum) {
    key: Key,
    resize: struct { width: u16, height: u16 },
};

/// A rectangular region of the screen.
pub const Rect = struct {
    x: u16 = 0,
    y: u16 = 0,
    width: u16 = 0,
    height: u16 = 0,

    /// Split horizontally at a given row offset from the top.
    pub fn splitHorizontal(self: Rect, at: u16) struct { top: Rect, bottom: Rect } {
        const split_at = @min(at, self.height);
        return .{
            .top = .{ .x = self.x, .y = self.y, .width = self.width, .height = split_at },
            .bottom = .{ .x = self.x, .y = self.y + split_at, .width = self.width, .height = self.height -| split_at },
        };
    }

    /// Split vertically at a given column offset from the left.
    pub fn splitVertical(self: Rect, at: u16) struct { left: Rect, right: Rect } {
        const split_at = @min(at, self.width);
        return .{
            .left = .{ .x = self.x, .y = self.y, .width = split_at, .height = self.height },
            .right = .{ .x = self.x + split_at, .y = self.y, .width = self.width -| split_at, .height = self.height },
        };
    }

    /// Return the area in cells.
    pub fn area(self: Rect) u32 {
        return @as(u32, self.width) * @as(u32, self.height);
    }

    /// Check if a point is inside this rect.
    pub fn contains(self: Rect, px: u16, py: u16) bool {
        return px >= self.x and px < self.x + self.width and
            py >= self.y and py < self.y + self.height;
    }
};

/// A single terminal cell.
pub const Cell = struct {
    char: u21 = ' ',
    style: Style = .{},

    pub const blank = Cell{};
};

/// Layout direction for splitting regions.
pub const Direction = enum {
    horizontal,
    vertical,
};

/// Constraint for layout splitting.
pub const Constraint = union(enum) {
    fixed: u16,
    percentage: u8,
    min: u16,
};

test "Rect splitHorizontal" {
    const r = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const split = r.splitHorizontal(3);
    try std.testing.expectEqual(@as(u16, 3), split.top.height);
    try std.testing.expectEqual(@as(u16, 21), split.bottom.height);
    try std.testing.expectEqual(@as(u16, 3), split.bottom.y);
}

test "Rect splitVertical" {
    const r = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    const split = r.splitVertical(30);
    try std.testing.expectEqual(@as(u16, 30), split.left.width);
    try std.testing.expectEqual(@as(u16, 50), split.right.width);
    try std.testing.expectEqual(@as(u16, 30), split.right.x);
}

test "Rect hit testing contains" {
    const r = Rect{ .x = 10, .y = 5, .width = 20, .height = 10 };
    try std.testing.expect(r.contains(10, 5));
    try std.testing.expect(r.contains(15, 10));
    try std.testing.expect(r.contains(29, 14));

    // Out of bounds
    try std.testing.expect(!r.contains(9, 5));
    try std.testing.expect(!r.contains(10, 4));
    try std.testing.expect(!r.contains(30, 14));
    try std.testing.expect(!r.contains(29, 15));
}

test "Rect contains" {
    const r = Rect{ .x = 5, .y = 5, .width = 10, .height = 10 };
    try std.testing.expect(r.contains(5, 5));
    try std.testing.expect(r.contains(14, 14));
    try std.testing.expect(!r.contains(15, 15));
    try std.testing.expect(!r.contains(4, 5));
}

test "Rect area" {
    const r = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    try std.testing.expectEqual(@as(u32, 1920), r.area());
}

test "Color fgCode" {
    const red: Color = .red;
    const bright_red: Color = .bright_red;
    const default: Color = .default;
    try std.testing.expectEqual(@as(?u8, 31), red.fgCode());
    try std.testing.expectEqual(@as(?u8, 91), bright_red.fgCode());
    try std.testing.expect(default.fgCode() == null);
}

test "Color bgCode" {
    const red: Color = .red;
    try std.testing.expectEqual(@as(?u8, 41), red.bgCode());
}

test "Cell default is blank space" {
    const cell = Cell{};
    try std.testing.expectEqual(@as(u21, ' '), cell.char);
}

test "Style default has no attributes" {
    const s = Style{};
    try std.testing.expect(!s.bold);
    try std.testing.expect(!s.italic);
    try std.testing.expect(!s.underline);
}

test {
    std.testing.refAllDecls(@This());
}
