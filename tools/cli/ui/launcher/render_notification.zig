//! Notification renderer for command launcher TUI.

const tui = @import("../core/mod.zig");
const layout = @import("layout.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");
const unicode = tui.unicode;

const TuiState = state_mod.TuiState;
const box = types.box;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize, msg: []const u8) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const level_color = switch (state.notification_level) {
        .success => chrome.success,
        .info => chrome.info,
        .warning => chrome.warning,
        .@"error" => chrome.@"error",
    };
    const icon = switch (state.notification_level) {
        .success => " OK ",
        .info => " INFO ",
        .warning => " WARN ",
        .@"error" => " ERR ",
    };
    const inner = layout.frameInnerWidth(width);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(level_color);
    try term.write(icon);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.text);

    const icon_width = unicode.displayWidth(icon);
    const msg_budget = layout.safeSub(inner, icon_width + 3);
    const message = if (msg.len > msg_budget) unicode.truncateToWidth(msg, msg_budget) else msg;
    const msg_w = unicode.displayWidth(message);
    try term.write(message);
    try term.write(th.reset);

    const used = 2 + icon_width + msg_w;
    const pad = layout.safeSub(inner, used);
    if (pad > 0) {
        try writeRepeat(term, " ", pad);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
