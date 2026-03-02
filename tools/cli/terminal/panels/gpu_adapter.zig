//! GPU panel adapter for the unified dashboard.
//!
//! Wraps `gpu_monitor.GpuMonitor` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const gpu_monitor = @import("../gpu_monitor.zig");

pub const GpuAdapter = struct {
    inner: gpu_monitor.GpuMonitor,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) GpuAdapter {
        return .{ .inner = gpu_monitor.GpuMonitor.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *GpuAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term; // inner holds its own terminal ref
    }

    pub fn tick(self: *GpuAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *GpuAdapter, _: events.Event) anyerror!bool {
        return false; // GPU monitor has no interactive controls
    }

    pub fn name(_: *GpuAdapter) []const u8 {
        return "GPU";
    }

    pub fn shortcutHint(_: *GpuAdapter) []const u8 {
        return "1";
    }

    pub fn deinit(self: *GpuAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *GpuAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(GpuAdapter, self);
    }
};

test "gpu_adapter name and hint" {
    var adapter: GpuAdapter = std.mem.zeroes(GpuAdapter);
    try std.testing.expectEqualStrings("GPU", adapter.name());
    try std.testing.expectEqualStrings("1", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
