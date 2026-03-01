//! Title bar renderer for command launcher TUI.

const std = @import("std");
const abi = @import("abi");
const tui = @import("../core/mod.zig");
const layout = @import("layout.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const writeRepeat = tui.render_utils.writeRepeat;
const writeClipped = tui.render_utils.writeClipped;
const unicode = tui.unicode;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = layout.frameInnerWidth(width);

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

    const show_mode = inner >= (mode.len + 4);
    const show_theme = if (show_mode)
        inner >= (mode.len + theme_display.len + 8)
    else
        inner >= (theme_display.len + 4);

    const right_width = (if (show_mode) mode.len + 2 else 0) +
        (if (show_theme) theme_display.len + 2 + (if (show_mode) 1 else 0) else 0);
    const left_max = layout.safeSub(inner, right_width + 1);
    const left = unicode.truncateToWidth(left_full, left_max);
    const left_width = unicode.displayWidth(left);
    const gap = layout.safeSub(inner, left_width + right_width);

    try term.write(th.bold);
    try term.write(chrome.title);
    try term.write(left);
    try term.write(th.reset);
    if (gap > 0) {
        try writeRepeat(term, " ", gap);
    }

    if (show_mode) {
        try writeChip(term, chrome, th, mode);
        if (show_theme) try term.write(" ");
    }
    if (show_theme) {
        try writeChip(term, chrome, th, theme_display);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Sub-title row
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);

    const subtitle = " local-first orchestration ";
    var version_buf: [128]u8 = undefined;
    const version_text = std.fmt.bufPrint(&version_buf, "v{s}", .{abi.version()}) catch "v?";
    const max_subtitle = layout.safeSub(
        inner,
        unicode.displayWidth(version_text) + 1,
    );
    const subtitle_display = unicode.truncateToWidth(subtitle, max_subtitle);
    const subtitle_width = unicode.displayWidth(subtitle_display);

    try term.write(chrome.subtitle);
    try term.write(subtitle_display);
    try term.write(chrome.title);
    try writeClipped(term, version_text, layout.safeSub(inner, subtitle_width));
    try term.write(th.reset);

    const subtitle_len = subtitle_width + 1 + unicode.displayWidth(version_text);
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

test {
    std.testing.refAllDecls(@This());
}
