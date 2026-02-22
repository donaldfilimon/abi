//! Status and help bar renderer for command launcher TUI.

const std = @import("std");
const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const tui_layout = @import("layout.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn renderStatus(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = width -| 2;

    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");

    var buf: [64]u8 = undefined;
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const cpu_str = std.fmt.bufPrint(&buf, "{d}", .{cpu_count}) catch "?";
    const shown_str = std.fmt.bufPrint(&buf, "{d}/{d}", .{
        state.filtered_indices.items.len,
        state.items.len,
    }) catch "?/?";

    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" platform ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(tui.Terminal.platformName());

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" cpu ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(cpu_str);

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" items ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(shown_str);

    try term.write("  ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" theme ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(state.theme_manager.current.name);

    const approx_len = 47 +
        tui.Terminal.platformName().len +
        cpu_str.len +
        shown_str.len +
        state.theme_manager.current.name.len;
    if (approx_len < inner) {
        try writeRepeat(term, " ", inner - approx_len);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}

pub fn renderHelp(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = width -| 2;

    try term.write(chrome.frame);
    try term.write(box.bl);
    try writeRepeat(term, box.h, tui_layout.clampedFrameWidth(state.term_size.cols) - 2);
    try term.write(box.br);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(" ");
    if (state.search_mode) {
        if (inner >= 42) {
            try writeKeyHint(term, chrome, th, "Tab", "complete");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "Enter", "run");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "Esc", "cancel");
        } else {
            try writeKeyHint(term, chrome, th, "Esc", "cancel");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "Enter", "run");
        }
    } else {
        if (inner >= 64) {
            try writeKeyHint(term, chrome, th, "Enter", "run");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "?", "preview");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "t/T", "theme");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "h", "history");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "q", "quit");
        } else if (inner >= 40) {
            try writeKeyHint(term, chrome, th, "Enter", "run");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "t", "theme");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "q", "quit");
        } else {
            try writeKeyHint(term, chrome, th, "q", "quit");
            try term.write(" ");
            try writeKeyHint(term, chrome, th, "?", "help");
        }
    }
    try term.write("\n");
}

fn writeKeyHint(
    term: *tui.Terminal,
    chrome: style_adapter.ChromeStyle,
    th: *const tui.Theme,
    key: []const u8,
    label: []const u8,
) !void {
    try term.write(chrome.keycap_bg);
    try term.write(chrome.keycap_fg);
    try term.write(" ");
    try term.write(key);
    try term.write(" ");
    try term.write(th.reset);
    try term.write(th.text_dim);
    try term.write(" ");
    try term.write(label);
    try term.write(th.reset);
}
