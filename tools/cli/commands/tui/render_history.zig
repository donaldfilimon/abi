//! History panel renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const menu_mod = @import("menu.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = width -| 2;

    // Header
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" HISTORY ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.text_dim);
    try term.write("recent command launches");
    try term.write(th.reset);

    const used_header: usize = 31;
    if (used_header < inner) try writeRepeat(term, " ", inner - used_header);
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    const max_show = @min(state.history.items.len, 5);
    for (0..max_show) |i| {
        const entry = state.history.items[i];
        const cmd_name = menu_mod.commandName(entry.command_id);

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write(" ");

        try term.write(chrome.keycap_bg);
        try term.write(chrome.keycap_fg);
        try term.write(" ");
        var num_buf: [1]u8 = undefined;
        num_buf[0] = '1' + @as(u8, @intCast(i));
        try term.write(&num_buf);
        try term.write(" ");
        try term.write(th.reset);
        try term.write(" ");

        try term.write(chrome.title);
        try term.write(cmd_name);
        try term.write(th.reset);

        const cmd_w = unicode.displayWidth(cmd_name);
        const used = 8 + cmd_w;
        if (used < inner) try writeRepeat(term, " ", inner - used);

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
