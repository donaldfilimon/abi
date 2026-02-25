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

pub const AgentAdapter = struct {
    inner: agent_panel.AgentPanel,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) AgentAdapter {
        return .{ .inner = agent_panel.AgentPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *AgentAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term; // inner holds its own terminal ref
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
    pub fn panel(self: *AgentAdapter) panel_mod {
        return panel_mod.from(AgentAdapter, self);
    }
};

test "agent_adapter name and hint" {
    var adapter: AgentAdapter = std.mem.zeroes(AgentAdapter);
    try std.testing.expectEqualStrings("Agent", adapter.name());
    try std.testing.expectEqualStrings("2", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
