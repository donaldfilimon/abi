//! Streaming dashboard adapter for the unified dashboard.
//!
//! Wraps `streaming_dashboard.StreamingDashboard` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const streaming_dashboard = @import("../streaming_dashboard.zig");

pub const StreamingAdapter = struct {
    inner: streaming_dashboard.StreamingDashboard,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) !StreamingAdapter {
        return .{ .inner = try streaming_dashboard.StreamingDashboard.init(allocator, term, theme, "localhost:8080") };
    }

    // -- Panel vtable methods --

    pub fn render(self: *StreamingAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(@intCast(rect.y), @intCast(rect.x), @intCast(rect.width), @intCast(rect.height));
        _ = term; // inner holds its own terminal ref
    }

    pub fn tick(self: *StreamingAdapter) anyerror!void {
        try self.inner.pollMetrics();
    }

    pub fn handleEvent(_: *StreamingAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *StreamingAdapter) []const u8 {
        return "Stream";
    }

    pub fn shortcutHint(_: *StreamingAdapter) []const u8 {
        return "5";
    }

    pub fn deinit(self: *StreamingAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *StreamingAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(StreamingAdapter, self);
    }
};

test "streaming_adapter name and hint" {
    var adapter: StreamingAdapter = std.mem.zeroes(StreamingAdapter);
    try std.testing.expectEqualStrings("Stream", adapter.name());
    try std.testing.expectEqualStrings("5", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
