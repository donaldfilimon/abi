//! Network panel adapter for the unified dashboard.
//!
//! Wraps `network_panel.NetworkPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../core/panel.zig");
const terminal = @import("../core/terminal.zig");
const layout = @import("../core/layout.zig");
const themes = @import("../core/themes.zig");
const events = @import("../core/events.zig");
const network_panel = @import("../network_panel.zig");

pub const NetworkAdapter = struct {
    inner: network_panel.NetworkPanel,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) NetworkAdapter {
        return .{ .inner = network_panel.NetworkPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *NetworkAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term; // inner holds its own terminal ref
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
    var adapter: NetworkAdapter = std.mem.zeroes(NetworkAdapter);
    try std.testing.expectEqualStrings("Net", adapter.name());
    try std.testing.expectEqualStrings("7", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
