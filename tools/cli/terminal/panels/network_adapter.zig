//! Network panel adapter for the unified dashboard.
//!
//! Wraps `network_panel.NetworkPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const network_panel = @import("../network_panel.zig");
const ru = @import("../render_utils.zig");

pub const NetworkAdapter = struct {
    inner: network_panel.NetworkPanel,

    /// Height consumed by the summary card row (3 card rows + 1 gap).
    const card_rows: u16 = 4;

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) NetworkAdapter {
        return .{ .inner = network_panel.NetworkPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *NetworkAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        // ── Summary cards ──────────────────────────────────────
        const card_width: u16 = @min(rect.width / 3, 20);
        var card_x = rect.x;

        var buf1: [32]u8 = undefined;
        const peers_val = std.fmt.bufPrint(&buf1, "{d}", .{@as(u16, 5)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Peers", peers_val, theme.info, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Raft State", "Leader", theme.success, theme);
        card_x += card_width + 1;

        var buf3: [32]u8 = undefined;
        const lat_val = std.fmt.bufPrint(&buf3, "{d}ms", .{@as(u16, 12)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Latency", lat_val, theme.accent, theme);

        // Delegate to inner panel with shifted rect
        self.inner.theme = theme;
        try self.inner.render(rect.y +| card_rows, rect.x, rect.width, rect.height -| card_rows);
    }

    pub fn tick(self: *NetworkAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *NetworkAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *NetworkAdapter) []const u8 {
        return "Net";
    }

    pub fn shortcutHint(_: *NetworkAdapter) []const u8 {
        return "7";
    }

    pub fn deinit(self: *NetworkAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *NetworkAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(NetworkAdapter, self);
    }
};

test "network_adapter name and hint" {
    const adapter: *NetworkAdapter = undefined;
    try std.testing.expectEqualStrings("Net", adapter.name());
    try std.testing.expectEqualStrings("7", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
