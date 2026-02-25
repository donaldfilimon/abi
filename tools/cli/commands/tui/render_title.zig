//! Title bar renderer for command launcher TUI.

const abi = @import("abi");
const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = width -| 2;

    // Top border
    try term.write(chrome.frame);
    try term.write(box.tl);
    try writeRepeat(term, box.h, inner);
    try term.write(box.tr);
    try term.write(th.reset);
    try term.write("\n");

    // Main title row
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);

    const left_full = " ABI COMMAND MATRIX ";
    const mode = if (state.search_mode) "SEARCH" else if (state.show_history) "HISTORY" else "BROWSE";
    const theme_name = state.theme_manager.current.name;
    const theme_display = if (theme_name.len > 14) theme_name[0..14] else theme_name;

    const right_width = mode.len + theme_display.len + 8; // [mode] [theme]
    const left_max = inner -| right_width -| 1;
    const left = if (left_full.len > left_max) left_full[0..left_max] else left_full;
    const gap = inner -| left.len -| right_width;

    try term.write(th.bold);
    try term.write(chrome.title);
    try term.write(left);
    try term.write(th.reset);
    try writeRepeat(term, " ", gap);

    try writeChip(term, chrome, th, mode);
    try term.write(" ");
    try writeChip(term, chrome, th, theme_display);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Sub-title row
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);

    const subtitle = " local-first orchestration ";
    const max_subtitle = inner -| (abi.version().len + 1);
    const subtitle_display = if (subtitle.len > max_subtitle) subtitle[0..max_subtitle] else subtitle;

    try term.write(chrome.subtitle);
    try term.write(subtitle_display);
    try term.write(chrome.title);
    try term.write("v");
    try term.write(abi.version());
    try term.write(th.reset);

    const subtitle_len = subtitle_display.len + 1 + abi.version().len;
    if (subtitle_len < inner) {
        try writeRepeat(term, " ", inner - subtitle_len);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Separator
    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");
}

fn writeChip(
    term: *tui.Terminal,
    chrome: style_adapter.ChromeStyle,
    th: *const tui.Theme,
    label: []const u8,
) !void {
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write("[");
    try term.write(label);
    try term.write("]");
    try term.write(th.reset);
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
