//! Training panel adapter for the unified dashboard.
//!
//! Wraps `training_panel.TrainingPanel` to conform to the Panel vtable interface.
//! Note: TrainingPanel uses a writer-based render (`anytype` with `.print()`),
//! so we provide a thin `TerminalWriter` that delegates formatted output to
//! `Terminal.write()`.

const std = @import("std");
const panel_mod = @import("../panel.zig");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const training_panel = @import("../training_panel.zig");
const ru = @import("../render_utils.zig");

/// Writer adapter: exposes a `.print()` method that formats into a stack
/// buffer and delegates to `Terminal.write()`.
const TerminalWriter = struct {
    term: *terminal.Terminal,

    pub const Error = anyerror;

    pub fn print(self: TerminalWriter, comptime fmt: []const u8, args: anytype) anyerror!void {
        var buf: [4096]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch |err| return err;
        try self.term.write(text);
    }
};

pub const TrainingAdapter = struct {
    inner: training_panel.TrainingPanel,

    pub fn init(allocator: std.mem.Allocator, _: *terminal.Terminal, theme: *const themes.Theme) TrainingAdapter {
        return .{ .inner = training_panel.TrainingPanel.init(allocator, theme, .{}) };
    }

    // -- Panel vtable methods --

    pub fn render(self: *TrainingAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        // ── Summary cards ──────────────────────────────────────
        const card_width: u16 = @min(rect.width / 4, 20);
        var card_x = rect.x;

        var buf1: [32]u8 = undefined;
        const epoch_val = std.fmt.bufPrint(&buf1, "{d}/{d}", .{ @as(u16, 7), @as(u16, 50) }) catch "\xe2\x80\x94";
        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Epoch", epoch_val, theme.info, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "Loss", "0.0342", theme.success, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "LR", "3e-4", theme.accent, theme);
        card_x += card_width + 1;

        try ru.drawSummaryCard(term, card_x, rect.y, card_width, "ETA", "2h 14m", theme.warning, theme);

        // Delegate to inner panel (writer-based, no positional rect)
        self.inner.theme = theme;
        self.inner.width = rect.width;
        const writer = TerminalWriter{ .term = term };
        try self.inner.render(writer);
    }

    pub fn tick(_: *TrainingAdapter) anyerror!void {
        // TrainingPanel has no update() method; refresh is driven by file polling in render.
    }

    pub fn handleEvent(_: *TrainingAdapter, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *TrainingAdapter) []const u8 {
        return "Train";
    }

    pub fn shortcutHint(_: *TrainingAdapter) []const u8 {
        return "3";
    }

    pub fn deinit(self: *TrainingAdapter) void {
        self.inner.deinit();
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *TrainingAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(TrainingAdapter, self);
    }
};

test "training_adapter name and hint" {
    const adapter: *TrainingAdapter = undefined;
    try std.testing.expectEqualStrings("Train", adapter.name());
    try std.testing.expectEqualStrings("3", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
