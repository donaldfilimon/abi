//! Chat panel adapter for the unified dashboard.
//!
//! Wraps `chat_panel.ChatPanel` to conform to the Panel vtable interface.

const std = @import("std");
const panel_mod = @import("../panel");
const terminal = @import("../terminal");
const layout = @import("../layout");
const themes = @import("../themes");
const events = @import("../events");
const chat_panel = @import("chat_panel");

pub const ChatAdapter = struct {
    inner: ?chat_panel.ChatPanel,
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) ChatAdapter {
        return .{
            .inner = chat_panel.ChatPanel.init(allocator, term, theme) catch null,
            .allocator = allocator,
            .term = term,
            .theme = theme,
        };
    }

    // -- Panel vtable methods --

    pub fn render(self: *ChatAdapter, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        if (self.inner) |*inner| {
            inner.theme = theme;
            inner.term = term;
            try inner.render(rect.y, rect.x, rect.width, rect.height);
        }
    }

    pub fn tick(self: *ChatAdapter) anyerror!void {
        if (self.inner) |*inner| {
            try inner.update();
        }
    }

    pub fn handleEvent(self: *ChatAdapter, event: events.Event) anyerror!bool {
        if (self.inner) |*inner| {
            return try inner.handleEvent(event);
        }
        return false;
    }

    pub fn name(_: *ChatAdapter) []const u8 {
        return "Chat";
    }

    pub fn shortcutHint(_: *ChatAdapter) []const u8 {
        return "0";
    }

    pub fn deinit(self: *ChatAdapter) void {
        if (self.inner) |*inner| {
            inner.deinit();
        }
    }

    /// Convert to a type-erased Panel.
    pub fn panel(self: *ChatAdapter) panel_mod.Panel {
        return panel_mod.Panel.from(ChatAdapter, self);
    }
};

test "chat_adapter name and hint" {
    const adapter: *ChatAdapter = undefined;
    try std.testing.expectEqualStrings("Chat", adapter.name());
    try std.testing.expectEqualStrings("0", adapter.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
