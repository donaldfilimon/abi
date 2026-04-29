const std = @import("std");
const types = @import("types.zig");
const render_mod = @import("stub_render.zig");

pub fn renderText(screen: *render_mod.Screen, area: types.Rect, text: []const u8, style: types.Style) void {
    _ = screen;
    _ = area;
    _ = text;
    _ = style;
}
pub fn renderPanel(screen: *render_mod.Screen, area: types.Rect, title: []const u8, border_style: types.Style) void {
    _ = screen;
    _ = area;
    _ = title;
    _ = border_style;
}
pub fn renderList(screen: *render_mod.Screen, area: types.Rect, items: []const []const u8, selected: usize, normal_style: types.Style, selected_style: types.Style) void {
    _ = screen;
    _ = area;
    _ = items;
    _ = selected;
    _ = normal_style;
    _ = selected_style;
}
pub fn renderStatusBar(screen: *render_mod.Screen, area: types.Rect, left: []const u8, right: []const u8, style: types.Style) void {
    _ = screen;
    _ = area;
    _ = left;
    _ = right;
    _ = style;
}
pub fn renderGauge(screen: *render_mod.Screen, area: types.Rect, percent: u8, label: []const u8, filled_style: types.Style, empty_style: types.Style) void {
    _ = screen;
    _ = area;
    _ = percent;
    _ = label;
    _ = filled_style;
    _ = empty_style;
}
pub fn renderTable(screen: *render_mod.Screen, area: types.Rect, headers: []const []const u8, rows: []const []const []const u8, header_style: types.Style, row_style: types.Style) void {
    _ = screen;
    _ = area;
    _ = headers;
    _ = rows;
    _ = header_style;
    _ = row_style;
}

pub const List = struct {
    items: []const []const u8 = &[_][]const u8{},
    selected: usize = 0,
    normal_style: types.Style = .{},
    selected_style: types.Style = .{ .reverse = true },

    pub fn render(self: *const List, screen: *render_mod.Screen, area: types.Rect) void {
        _ = self;
        _ = screen;
        _ = area;
    }
};

pub const Spinner = struct {
    frames: []const []const u8 = &[_][]const u8{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" },
    index: usize = 0,
    style: types.Style = .{},

    pub fn tick(self: *Spinner) void {
        _ = self;
    }
    pub fn render(self: *const Spinner, screen: *render_mod.Screen, area: types.Rect) void {
        _ = self;
        _ = screen;
        _ = area;
    }
};

pub const TextInput = struct {
    buffer: std.ArrayList(u8),
    cursor: usize = 0,
    style: types.Style = .{},
    cursor_style: types.Style = .{ .reverse = true },

    pub fn init(allocator: std.mem.Allocator) TextInput {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
        };
    }

    pub fn deinit(self: *TextInput) void {
        _ = self;
    }
    pub fn insert(self: *TextInput, char: u8) !void {
        _ = self;
        _ = char;
    }
    pub fn backspace(self: *TextInput) void {
        _ = self;
    }
    pub fn render(self: *const TextInput, screen: *render_mod.Screen, area: types.Rect) void {
        _ = self;
        _ = screen;
        _ = area;
    }
};
