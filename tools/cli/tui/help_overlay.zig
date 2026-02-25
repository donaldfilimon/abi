//! Reusable help overlay for TUI dashboards.
//!
//! Renders a centered box with keybinding help text.
//! Used by the unified dashboard shell and individual panels.

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const render_utils = @import("render_utils.zig");

pub const HelpOverlay = struct {
    title: []const u8,
    lines: []const []const u8,
    visible: bool,

    pub fn init(title: []const u8, lines: []const []const u8) HelpOverlay {
        return .{ .title = title, .lines = lines, .visible = false };
    }

    pub fn toggle(self: *HelpOverlay) void {
        self.visible = !self.visible;
    }

    pub fn show(self: *HelpOverlay) void {
        self.visible = true;
    }

    pub fn hide(self: *HelpOverlay) void {
        self.visible = false;
    }

    /// Render the overlay centered in the given area.
    pub fn render(self: *const HelpOverlay, term: *terminal.Terminal, theme: *const themes.Theme, width: u16, height: u16) !void {
        if (!self.visible) return;

        const content_lines = self.lines.len + 4; // title + blank + lines + blank + footer
        const box_width: u16 = @min(60, width -| 4);
        const box_height: u16 = @min(@as(u16, @intCast(content_lines + 2)), height -| 2);
        const start_col = (width -| box_width) / 2;
        const start_row = (height -| box_height) / 2;

        // Top border
        try term.moveTo(start_row, start_col);
        try term.write(theme.border);
        try term.write("\u{2554}"); // top-left double
        try render_utils.writeRepeat(term, "\u{2550}", @as(usize, box_width) -| 2); // horizontal double
        try term.write("\u{2557}"); // top-right double
        try term.write(theme.reset);

        var row: u16 = start_row + 1;

        // Title line
        try renderBoxLine(term, theme, start_col, box_width, row, "");
        row += 1;
        try renderBoxLine(term, theme, start_col, box_width, row, self.title);
        row += 1;
        try renderBoxLine(term, theme, start_col, box_width, row, "");
        row += 1;

        // Content lines
        const max_lines = @min(self.lines.len, @as(usize, box_height) -| 5);
        for (self.lines[0..max_lines]) |line| {
            try renderBoxLine(term, theme, start_col, box_width, row, line);
            row += 1;
        }

        // Empty line before footer
        try renderBoxLine(term, theme, start_col, box_width, row, "");
        row += 1;

        // Bottom border
        try term.moveTo(row, start_col);
        try term.write(theme.border);
        try term.write("\u{255A}"); // bottom-left double
        try render_utils.writeRepeat(term, "\u{2550}", @as(usize, box_width) -| 2); // horizontal double
        try term.write("\u{255D}"); // bottom-right double
        try term.write(theme.reset);

        // Footer hint (clamped to terminal bounds)
        const footer_row = @min(row + 1, height -| 1);
        try term.moveTo(footer_row, start_col + 4);
        try term.write(theme.text_dim);
        try term.write("Press q, ?, Esc, or Enter to close");
        try term.write(theme.reset);
    }

    fn renderBoxLine(term: *terminal.Terminal, theme: *const themes.Theme, start_col: u16, box_width: u16, row: u16, text: []const u8) !void {
        try term.moveTo(row, start_col);
        try term.write(theme.border);
        try term.write("\u{2551}"); // vertical double
        try term.write(theme.reset);

        const inner: usize = @as(usize, box_width) -| 2;
        const display = if (text.len > inner) text[0..inner] else text;
        try term.write(theme.text);
        try term.write(display);
        try term.write(theme.reset);

        if (display.len < inner) {
            try render_utils.writeRepeat(term, " ", inner - display.len);
        }

        try term.write(theme.border);
        try term.write("\u{2551}"); // vertical double
        try term.write(theme.reset);
    }
};

test "help_overlay toggle" {
    const lines = [_][]const u8{ "  [q] quit", "  [p] pause" };
    var overlay = HelpOverlay.init("Test Help", &lines);
    try std.testing.expect(!overlay.visible);
    overlay.toggle();
    try std.testing.expect(overlay.visible);
    overlay.toggle();
    try std.testing.expect(!overlay.visible);
    overlay.show();
    try std.testing.expect(overlay.visible);
    overlay.hide();
    try std.testing.expect(!overlay.visible);
}

test {
    std.testing.refAllDecls(@This());
}
