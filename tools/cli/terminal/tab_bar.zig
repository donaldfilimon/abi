//! Tab bar renderer for the unified TUI dashboard.
//!
//! Renders a horizontal tab strip: ` GPU | Agent | Train | ... `
//! Active tab is highlighted using theme.tab_active; others use theme.tab_inactive.

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const render_utils = @import("render_utils.zig");

pub const TabBar = struct {
    labels: []const []const u8,
    active: usize,

    pub fn init(labels: []const []const u8) TabBar {
        return .{ .labels = labels, .active = 0 };
    }

    pub fn setActive(self: *TabBar, idx: usize) void {
        if (idx < self.labels.len) self.active = idx;
    }

    pub fn next(self: *TabBar) void {
        if (self.labels.len == 0) return;
        self.active = (self.active + 1) % self.labels.len;
    }

    pub fn prev(self: *TabBar) void {
        if (self.labels.len == 0) return;
        self.active = if (self.active == 0) self.labels.len - 1 else self.active - 1;
    }

    /// Category group boundaries. Tabs at these indices start a new group,
    /// rendered with a double-pipe separator instead of the regular pipe.
    /// Groups: AI Tools (0-4), Data (5-7), Infra (8-10), System (11-14).
    const group_starts = [_]usize{ 0, 5, 8, 11 };

    /// Returns true if the tab at `idx` is the first tab in a category group.
    fn isGroupStart(idx: usize) bool {
        for (group_starts) |g| {
            if (g == idx) return true;
        }
        return false;
    }

    /// Render the tab bar at the given terminal row.
    /// Uses the full width, drawing tabs left-aligned with separators.
    /// Groups are separated by double-pipe ║ and each tab has a status dot.
    pub fn render(self: *const TabBar, term: *terminal.Terminal, theme: *const themes.Theme, row: u16, width: u16) !void {
        try term.moveTo(row, 0);

        // Leading space
        try term.write(" ");
        var used: usize = 1;

        for (self.labels, 0..) |label, i| {
            const is_active = (i == self.active);

            // Separator between tabs: ║ between groups, │ within
            if (i > 0) {
                const sep = if (isGroupStart(i)) " \u{2551} " else " \u{2502} ";
                try term.write(theme.tab_separator);
                try term.write(sep);
                try term.write(theme.reset);
                used += 3;
            }

            // Status dot (green = active, gray = idle)
            const dot_color = if (is_active) theme.status_active else theme.status_idle;
            try term.write(dot_color);
            try term.write("\u{25cf}");
            try term.write(theme.reset);
            used += 1;

            // Tab label
            const tab_style = if (is_active) theme.tab_active else theme.tab_inactive;
            try term.write(tab_style);
            try term.write(" ");
            try term.write(label);
            try term.write(" ");
            try term.write(theme.reset);
            used += label.len + 2;
        }

        // Fill remaining width
        if (used < @as(usize, width)) {
            try render_utils.writeRepeat(term, " ", @as(usize, width) - used);
        }
    }

    /// Returns the number of tabs.
    pub fn count(self: *const TabBar) usize {
        return self.labels.len;
    }
};

// Tests
test "tab_bar init and navigation" {
    const labels = [_][]const u8{ "GPU", "Agent", "Train" };
    var bar = TabBar.init(&labels);
    try std.testing.expectEqual(@as(usize, 0), bar.active);

    bar.next();
    try std.testing.expectEqual(@as(usize, 1), bar.active);

    bar.next();
    try std.testing.expectEqual(@as(usize, 2), bar.active);

    bar.next(); // wraps
    try std.testing.expectEqual(@as(usize, 0), bar.active);

    bar.prev(); // wraps back
    try std.testing.expectEqual(@as(usize, 2), bar.active);
}

test "tab_bar setActive bounds" {
    const labels = [_][]const u8{ "A", "B" };
    var bar = TabBar.init(&labels);
    bar.setActive(1);
    try std.testing.expectEqual(@as(usize, 1), bar.active);
    bar.setActive(99); // out of bounds, no change
    try std.testing.expectEqual(@as(usize, 1), bar.active);
}

test {
    std.testing.refAllDecls(@This());
}
