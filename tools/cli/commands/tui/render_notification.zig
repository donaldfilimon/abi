//! Notification renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize, msg: []const u8) !void {
    const th = state.theme();
    const level_color = switch (state.notification_level) {
        .success => th.success,
        .info => th.info,
        .warning => th.warning,
        .@"error" => th.@"error",
    };
    const icon = switch (state.notification_level) {
        .success => "\xE2\x9C\x93",
        .info => "\xE2\x84\xB9",
        .warning => "\xE2\x9A\xA0",
        .@"error" => "\xE2\x9C\x97",
    };

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(level_color);
    try term.write(icon);
    try term.write(" ");
    try term.write(msg);
    try term.write(th.reset);

    const icon_w = unicode.displayWidth(icon);
    const msg_w = unicode.displayWidth(msg);
    const used = 2 + icon_w + 1 + msg_w;
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}
