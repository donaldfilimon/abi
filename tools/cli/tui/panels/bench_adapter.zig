//! Benchmark panel adapter for the unified dashboard.
//!
//! Wraps `bench_panel.BenchmarkPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const bench_panel = @import("../bench_panel.zig");

pub const BenchAdapter = struct {
    inner: bench_panel.BenchmarkPanel,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) BenchAdapter {
        return .{ .inner = bench_panel.BenchmarkPanel.init(allocator, term, theme) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *BenchAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        self.inner.theme = theme;
        try self.inner.render(rect.y, rect.x, rect.width, rect.height);
        _ = term; // inner holds its own terminal ref
    }

    pub fn tick(self: *BenchAdapter) anyerror!void {
        try self.inner.update();
    }

    pub fn handleEvent(_: *BenchAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *BenchAdapter) []const u8 {
        return "Bench";
    }

    pub fn shortcutHint(_: *BenchAdapter) []const u8 {
        return "8";
    }

    pub fn deinit(self: *BenchAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *BenchAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(BenchAdapter, self);
    }
};

test "bench_adapter name and hint" {
    var adapter: BenchAdapter = std.mem.zeroes(BenchAdapter);
    try std.testing.expectEqualStrings("Bench", adapter.name());
    try std.testing.expectEqualStrings("8", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
