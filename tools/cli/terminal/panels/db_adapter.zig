//! Database panel adapter for the unified dashboard.
//!
//! Wraps `db_panel.DatabasePanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const db_panel = @import("../db_panel.zig");
const ru = @import("../render_utils.zig");

pub const DbAdapter = struct {
    inner: db_panel.DatabasePanel,

    /// Height consumed by the summary card row (3 card rows + 1 gap).
    const card_rows: u16 = 4;

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) DbAdapter {
        return .{ .inner = db_panel.DatabasePanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *DbAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        // ── Summary cards ──────────────────────────────────────
        const card_width: u16 = @min(rect.width / 3, 20);
        var card_x = rect.x;

        var buf1: [32]u8 = undefined;
        const vec_val = std.fmt.bufPrint(&buf1, "{d}", .{@as(u32, 48200)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Vectors", vec_val, theme.info, theme);
        card_x += card_width + 1;

        var buf2: [32]u8 = undefined;
        const dim_val = std.fmt.bufPrint(&buf2, "{d}", .{@as(u16, 1536)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Dimension", dim_val, theme.accent, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Index", "Ready", theme.success, theme);

        // Delegate to inner panel with shifted rect
        self.inner.theme = theme;
        try self.inner.render(rect.y +| card_rows, rect.x, rect.width, rect.height -| card_rows);
    }

    pub fn tick(self: *DbAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *DbAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *DbAdapter) []const u8 {
        return "DB";
    }

    pub fn shortcutHint(_: *DbAdapter) []const u8 {
        return "6";
    }

    pub fn deinit(self: *DbAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *DbAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(DbAdapter, self);
    }
};

test "db_adapter name and hint" {
    const adapter: *DbAdapter = undefined;
    try std.testing.expectEqualStrings("DB", adapter.name());
    try std.testing.expectEqualStrings("6", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
