//! Integrated Multi-Persona Chat TUI
//!
//! Provides the primary interactive chat interface for ABI.

const std = @import("std");
const command_mod = @import("../../command");
const context_mod = @import("../../framework/context");
const utils = @import("../../utils/mod.zig");
const tui = @import("../../terminal/mod.zig");
const chat_panel = @import("../../terminal/panels/chat_panel");

pub const meta: command_mod.Meta = .{
    .name = "chat-tui",
    .description = "Launch the interactive multi-persona TUI chat interface",
    .aliases = &.{"ctui"},
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = args;

    if (!tui.Terminal.isSupported()) {
        const caps = tui.Terminal.capabilities();
        utils.output.printError("TUI chat is not supported on {s}", .{caps.platform_name});
        return;
    }

    var term = tui.Terminal.init(ctx.allocator);
    defer term.deinit();

    var tm = tui.themes.ThemeManager.init();
    // note: ThemeManager doesn't need deinit as it uses static themes

    var panel = try chat_panel.ChatPanel.init(ctx.allocator, &term, tm.current);
    defer panel.deinit();

    var db = tui.dashboard.Dashboard(chat_panel.ChatPanel).init(
        ctx.allocator,
        &term,
        tm.current,
        panel,
        .{
            .title = "ABI Multi-Persona Chat",
            .refresh_rate_ms = 100,
        },
    );
    defer db.deinit();

    try term.enter();
    defer term.exit() catch {};

    try db.run();
}
