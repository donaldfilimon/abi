//! Terminal User Interface (TUI) Framework
//!
//! Interactive terminal UI with widgets, colors, and keyboard handling.

const std = @import("std");
const ArrayList = std.array_list.Managed;
const builtin = @import("builtin");

/// ANSI escape codes for terminal control
pub const ansi = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const underline = "\x1b[4m";

    // Colors
    pub const black = "\x1b[30m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";

    // Bright colors
    pub const bright_black = "\x1b[90m";
    pub const bright_red = "\x1b[91m";
    pub const bright_green = "\x1b[92m";
    pub const bright_yellow = "\x1b[93m";
    pub const bright_blue = "\x1b[94m";
    pub const bright_magenta = "\x1b[95m";
    pub const bright_cyan = "\x1b[96m";
    pub const bright_white = "\x1b[97m";

    // Background colors
    pub const bg_black = "\x1b[40m";
    pub const bg_red = "\x1b[41m";
    pub const bg_green = "\x1b[42m";
    pub const bg_yellow = "\x1b[43m";
    pub const bg_blue = "\x1b[44m";
    pub const bg_magenta = "\x1b[45m";
    pub const bg_cyan = "\x1b[46m";
    pub const bg_white = "\x1b[47m";

    // Cursor control
    pub const clear_screen = "\x1b[2J";
    pub const clear_line = "\x1b[2K";
    pub const cursor_home = "\x1b[H";
    pub const cursor_hide = "\x1b[?25l";
    pub const cursor_show = "\x1b[?25h";

    pub fn moveTo(x: u16, y: u16) [16]u8 {
        var buf: [16]u8 = undefined;
        const len = std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{ y + 1, x + 1 }) catch return buf;
        _ = len;
        return buf;
    }

    pub fn rgb(r: u8, g: u8, b: u8) [20]u8 {
        var buf: [20]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "\x1b[38;2;{d};{d};{d}m", .{ r, g, b }) catch {};
        return buf;
    }
};

/// Box drawing characters
pub const box = struct {
    pub const top_left = "┌";
    pub const top_right = "┐";
    pub const bottom_left = "└";
    pub const bottom_right = "┘";
    pub const horizontal = "─";
    pub const vertical = "│";
    pub const cross = "┼";
    pub const t_down = "┬";
    pub const t_up = "┴";
    pub const t_right = "├";
    pub const t_left = "┤";

    // Double line
    pub const double_top_left = "╔";
    pub const double_top_right = "╗";
    pub const double_bottom_left = "╚";
    pub const double_bottom_right = "╝";
    pub const double_horizontal = "═";
    pub const double_vertical = "║";
};

/// UI Theme
pub const Theme = struct {
    primary: []const u8 = ansi.bright_cyan,
    secondary: []const u8 = ansi.bright_magenta,
    success: []const u8 = ansi.bright_green,
    warning: []const u8 = ansi.bright_yellow,
    error_color: []const u8 = ansi.bright_red,
    text: []const u8 = ansi.white,
    muted: []const u8 = ansi.bright_black,

    pub const default = Theme{};
    pub const dark = Theme{
        .primary = ansi.cyan,
        .secondary = ansi.magenta,
        .text = ansi.bright_white,
    };
};

/// Widget base for TUI components
pub const Widget = struct {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    visible: bool = true,
    focused: bool = false,

    pub fn contains(self: Widget, px: u16, py: u16) bool {
        return px >= self.x and px < self.x + self.width and
            py >= self.y and py < self.y + self.height;
    }
};

/// Progress bar widget
pub const ProgressBar = struct {
    widget: Widget,
    progress: f32, // 0.0 to 1.0
    label: []const u8,
    show_percentage: bool = true,
    filled_char: []const u8 = "█",
    empty_char: []const u8 = "░",
    theme: Theme = Theme.default,

    pub fn init(x: u16, y: u16, width: u16, label: []const u8) ProgressBar {
        return .{
            .widget = .{ .x = x, .y = y, .width = width, .height = 1 },
            .progress = 0,
            .label = label,
        };
    }

    pub fn setProgress(self: *ProgressBar, value: f32) void {
        self.progress = std.math.clamp(value, 0.0, 1.0);
    }

    pub fn render(self: *ProgressBar) void {
        const bar_width = self.widget.width - @as(u16, @intCast(self.label.len)) - 7;
        const filled = @as(u16, @intFromFloat(@as(f32, @floatFromInt(bar_width)) * self.progress));

        std.debug.print("{s}{s} [", .{ self.theme.text, self.label });
        std.debug.print("{s}", .{self.theme.primary});
        for (0..filled) |_| std.debug.print("{s}", .{self.filled_char});
        std.debug.print("{s}", .{self.theme.muted});
        for (0..bar_width - filled) |_| std.debug.print("{s}", .{self.empty_char});
        std.debug.print("{s}] ", .{ansi.reset});

        if (self.show_percentage) {
            std.debug.print("{d:5.1}%", .{self.progress * 100});
        }
        std.debug.print("\n", .{});
    }
};

/// Spinner widget for loading states
pub const Spinner = struct {
    widget: Widget,
    frame: usize = 0,
    label: []const u8,
    frames: []const []const u8 = &.{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" },
    theme: Theme = Theme.default,

    pub fn init(x: u16, y: u16, label: []const u8) Spinner {
        return .{
            .widget = .{ .x = x, .y = y, .width = @intCast(label.len + 3), .height = 1 },
            .label = label,
        };
    }

    pub fn tick(self: *Spinner) void {
        self.frame = (self.frame + 1) % self.frames.len;
    }

    pub fn render(self: *Spinner) void {
        std.debug.print("{s}{s}{s} {s}\n", .{
            self.theme.primary,
            self.frames[self.frame],
            self.theme.text,
            self.label,
        });
    }
};

/// Table widget for data display
pub const Table = struct {
    headers: []const []const u8,
    rows: ArrayList([]const []const u8),
    col_widths: []u16,
    theme: Theme = Theme.default,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, headers: []const []const u8) !Table {
        var widths = try allocator.alloc(u16, headers.len);
        for (headers, 0..) |h, i| {
            widths[i] = @intCast(h.len);
        }
        return .{
            .headers = headers,
            .rows = ArrayList([]const []const u8).init(allocator),
            .col_widths = widths,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Table) void {
        self.rows.deinit();
        self.allocator.free(self.col_widths);
    }

    pub fn addRow(self: *Table, row: []const []const u8) !void {
        try self.rows.append(row);
        for (row, 0..) |cell, i| {
            if (i < self.col_widths.len) {
                self.col_widths[i] = @max(self.col_widths[i], @as(u16, @intCast(cell.len)));
            }
        }
    }

    pub fn render(self: *Table) void {
        // Header
        std.debug.print("{s}{s}", .{ self.theme.primary, ansi.bold });
        for (self.headers, 0..) |h, i| {
            std.debug.print(" {s}", .{h});
            const pad = self.col_widths[i] - @as(u16, @intCast(h.len));
            for (0..pad + 1) |_| std.debug.print(" ", .{});
        }
        std.debug.print("{s}\n", .{ansi.reset});

        // Separator
        std.debug.print("{s}", .{self.theme.muted});
        for (self.col_widths) |w| {
            std.debug.print(" ", .{});
            for (0..w + 1) |_| std.debug.print("─", .{});
        }
        std.debug.print("{s}\n", .{ansi.reset});

        // Rows
        for (self.rows.items) |row| {
            std.debug.print("{s}", .{self.theme.text});
            for (row, 0..) |cell, i| {
                if (i < self.col_widths.len) {
                    std.debug.print(" {s}", .{cell});
                    const pad = self.col_widths[i] - @as(u16, @intCast(cell.len));
                    for (0..pad + 1) |_| std.debug.print(" ", .{});
                }
            }
            std.debug.print("{s}\n", .{ansi.reset});
        }
    }
};

/// Menu widget for selection
pub const Menu = struct {
    items: []const []const u8,
    selected: usize = 0,
    title: []const u8,
    theme: Theme = Theme.default,

    pub fn init(title: []const u8, items: []const []const u8) Menu {
        return .{ .title = title, .items = items };
    }

    pub fn up(self: *Menu) void {
        if (self.selected > 0) self.selected -= 1;
    }

    pub fn down(self: *Menu) void {
        if (self.selected < self.items.len - 1) self.selected += 1;
    }

    pub fn render(self: *Menu) void {
        std.debug.print("{s}{s}{s}{s}\n", .{ self.theme.primary, ansi.bold, self.title, ansi.reset });
        std.debug.print("{s}─────────────────────{s}\n", .{ self.theme.muted, ansi.reset });

        for (self.items, 0..) |item, i| {
            if (i == self.selected) {
                std.debug.print("{s}{s} ▶ {s}{s}\n", .{ self.theme.primary, ansi.bold, item, ansi.reset });
            } else {
                std.debug.print("{s}   {s}{s}\n", .{ self.theme.text, item, ansi.reset });
            }
        }
    }
};

/// Application TUI container
pub const App = struct {
    allocator: std.mem.Allocator,
    title: []const u8,
    theme: Theme,
    running: bool = false,

    pub fn init(allocator: std.mem.Allocator, title: []const u8) App {
        return .{
            .allocator = allocator,
            .title = title,
            .theme = Theme.default,
        };
    }

    pub fn clear(self: *App) void {
        _ = self;
        std.debug.print("{s}{s}", .{ ansi.clear_screen, ansi.cursor_home });
    }

    pub fn header(self: *App) void {
        std.debug.print("{s}{s}", .{ self.theme.primary, box.double_top_left });
        for (0..self.title.len + 2) |_| std.debug.print("{s}", .{box.double_horizontal});
        std.debug.print("{s}\n", .{box.double_top_right});

        std.debug.print("{s} {s}{s}{s}{s} {s}\n", .{
            box.double_vertical,
            ansi.bold,
            self.title,
            ansi.reset,
            self.theme.primary,
            box.double_vertical,
        });

        std.debug.print("{s}", .{box.double_bottom_left});
        for (0..self.title.len + 2) |_| std.debug.print("{s}", .{box.double_horizontal});
        std.debug.print("{s}{s}\n\n", .{ box.double_bottom_right, ansi.reset });
    }

    pub fn footer(self: *App, message: []const u8) void {
        std.debug.print("\n{s}{s}{s}\n", .{ self.theme.muted, message, ansi.reset });
    }
};

/// Print styled text helpers
pub fn success(msg: []const u8) void {
    std.debug.print("{s}✓ {s}{s}\n", .{ ansi.bright_green, msg, ansi.reset });
}

pub fn warning(msg: []const u8) void {
    std.debug.print("{s}⚠ {s}{s}\n", .{ ansi.bright_yellow, msg, ansi.reset });
}

pub fn errorMsg(msg: []const u8) void {
    std.debug.print("{s}✗ {s}{s}\n", .{ ansi.bright_red, msg, ansi.reset });
}

pub fn info(msg: []const u8) void {
    std.debug.print("{s}ℹ {s}{s}\n", .{ ansi.bright_cyan, msg, ansi.reset });
}

test "progress bar" {
    var bar = ProgressBar.init(0, 0, 40, "Loading");
    bar.setProgress(0.5);
    bar.render();
}

test "menu navigation" {
    const testing = std.testing;
    var menu = Menu.init("Options", &.{ "Option 1", "Option 2", "Option 3" });

    try testing.expectEqual(@as(usize, 0), menu.selected);
    menu.down();
    try testing.expectEqual(@as(usize, 1), menu.selected);
    menu.up();
    try testing.expectEqual(@as(usize, 0), menu.selected);
}
