//! Notification renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
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
    const inner = width -| 2;

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
    try term.write(msg);
    try term.write(th.reset);

    const used = 2 + unicode.displayWidth(icon) + unicode.displayWidth(msg);
    if (used < inner) {
        try writeRepeat(term, " ", inner - used);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}
