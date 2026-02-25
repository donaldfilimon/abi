//! Brain dashboard panel adapter for the unified dashboard.
//!
//! Wraps `brain_panel.BrainDashboardPanel` to conform to the Panel vtable interface.
//! Note: BrainDashboardPanel has no allocator, no update, and no deinit,
//! so tick and deinit are no-ops. Render creates default DashboardData.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const brain_panel = @import("../brain_panel.zig");

pub const BrainAdapter = struct {
    inner: brain_panel.BrainDashboardPanel,
    dashboard_data: brain_panel.DashboardData,

    pub fn init(term: *terminal.Terminal, theme: *const themes.Theme) BrainAdapter {
        return .{
            .inner = brain_panel.BrainDashboardPanel.init(term, theme),
            .dashboard_data = brain_panel.DashboardData.init(),
        };
    }

    // -- Panel vtable methods --

    pub fn render(self: *BrainAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        self.inner.term = term;
        try self.inner.render(&self.dashboard_data, rect.y, rect.x, rect.width, rect.height);
    }

    pub fn tick(_: *BrainAdapter) anyerror!void {
        // BrainDashboardPanel has no update method; data is externally supplied.
    }

    pub fn handleEvent(_: *BrainAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *BrainAdapter) []const u8 {
        return "Brain";
    }

    pub fn shortcutHint(_: *BrainAdapter) []const u8 {
        return "9";
    }

    pub fn deinit(_: *BrainAdapter) void {
        // BrainDashboardPanel has no deinit; no resources to free.
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *BrainAdapter) panel_mod {
        return panel_mod.from(BrainAdapter, self);
    }
};

test "brain_adapter name and hint" {
    var adapter: BrainAdapter = std.mem.zeroes(BrainAdapter);
    try std.testing.expectEqualStrings("Brain", adapter.name());
    try std.testing.expectEqualStrings("9", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
