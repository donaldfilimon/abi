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

pub const DbAdapter = struct {
    inner: db_panel.DatabasePanel,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) DbAdapter {
        return .{ .inner = db_panel.DatabasePanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *DbAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term; // inner holds its own terminal ref
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
    var adapter: DbAdapter = std.mem.zeroes(DbAdapter);
    try std.testing.expectEqualStrings("DB", adapter.name());
    try std.testing.expectEqualStrings("6", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
