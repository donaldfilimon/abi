//! Agent panel adapter for the unified dashboard.
//!
//! Wraps `agent_panel.AgentPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const agent_panel = @import("../agent_panel.zig");
const ru = @import("../render_utils.zig");

pub const AgentAdapter = struct {
    inner: agent_panel.AgentPanel,

    /// Height consumed by the summary card row (3 card rows + 1 gap).
    const card_rows = ru.summary_card_rows;

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) AgentAdapter {
        return .{ .inner = agent_panel.AgentPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *AgentAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        // ── Summary cards ──────────────────────────────────────
        const card_width: u16 = @min(rect.width / 4, 20);
        var card_x = rect.x;

        var buf1: [32]u8 = undefined;
        const agents_val = std.fmt.bufPrint(&buf1, "{d}", .{@as(u16, 3)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Active Agents", agents_val, theme.success, theme);
        card_x += card_width + 1;

        var buf2: [32]u8 = undefined;
        const tokens_val = std.fmt.bufPrint(&buf2, "{d}k", .{@as(u16, 128)}) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Total Tokens", tokens_val, theme.info, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Model", "claude-4", theme.accent, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Status", "Running", theme.success, theme);

        // Delegate to inner panel with shifted rect
        self.inner.theme = theme;
        try self.inner.render(rect.y +| card_rows, rect.x, rect.width, rect.height -| card_rows);
    }

    pub fn tick(self: *AgentAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *AgentAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *AgentAdapter) []const u8 {
        return "Agent";
    }

    pub fn shortcutHint(_: *AgentAdapter) []const u8 {
        return "2";
    }

    pub fn deinit(self: *AgentAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *AgentAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(AgentAdapter, self);
    }
};

test "agent_adapter name and hint" {
    const adapter: *AgentAdapter = undefined;
    try std.testing.expectEqualStrings("Agent", adapter.name());
    try std.testing.expectEqualStrings("2", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
