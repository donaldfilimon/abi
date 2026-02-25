//! Model management panel adapter for the unified dashboard.
//!
//! Wraps `model_panel.ModelManagementPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const model_panel = @import("../model_panel.zig");

pub const ModelAdapter = struct {
    inner: model_panel.ModelManagementPanel,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) ModelAdapter {
        return .{ .inner = model_panel.ModelManagementPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *ModelAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(@intCast(rect.y), @intCast(rect.x), @intCast(rect.width), @intCast(rect.height));
        _ = term; // inner holds its own terminal ref
    }

    pub fn tick(self: *ModelAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *ModelAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *ModelAdapter) []const u8 {
        return "Model";
    }

    pub fn shortcutHint(_: *ModelAdapter) []const u8 {
        return "4";
    }

    pub fn deinit(self: *ModelAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *ModelAdapter) panel_mod {
        return panel_mod.from(ModelAdapter, self);
    }
};

test "model_adapter name and hint" {
    var adapter: ModelAdapter = std.mem.zeroes(ModelAdapter);
    try std.testing.expectEqualStrings("Model", adapter.name());
    try std.testing.expectEqualStrings("4", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
