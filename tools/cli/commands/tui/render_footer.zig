//! Status and help bar renderer for command launcher TUI.

const std = @import("std");
const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const tui_layout = @import("layout.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn renderStatus(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();

    try term.write(th.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");

    const os_name = tui.Terminal.platformName();
    try term.write(th.text_dim);
    try term.write(os_name);

    var cpu_buf: [16]u8 = undefined;
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const cpu_str = std.fmt.bufPrint(&cpu_buf, " \u{2502} {d} CPU", .{cpu_count}) catch "";
    try term.write(cpu_str);

    try term.write(" \u{2502} TTY");

    var count_buf: [32]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, " \u{2502} {d}/{d} items", .{
        state.filtered_indices.items.len,
        state.items.len,
    }) catch "?";
    try term.write(count_str);
    try term.write(th.reset);

    const used = 2 +
        unicode.displayWidth(os_name) +
        unicode.displayWidth(cpu_str) +
        7 +
        unicode.displayWidth(count_str);
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}

pub fn renderHelp(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    _ = width;

    try term.write(th.border);
    try term.write(box.bl);
    try writeRepeat(term, box.h, tui_layout.clampedFrameWidth(state.term_size.cols) - 2);
    try term.write(box.br);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(" ");
    if (state.search_mode) {
        try term.write(th.text_dim);
        try term.write("Type to filter \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("Tab");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Complete \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("Enter");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Run \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("Esc");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Cancel");
        try term.write(th.reset);
    } else {
        try term.write(th.accent);
        try term.write("Enter");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Run \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("?");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Preview \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("t");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Theme \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("h");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" History \u{2502} ");
        try term.write(th.reset);
        try term.write(th.accent);
        try term.write("q");
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write(" Quit");
        try term.write(th.reset);
    }
    try term.write("\n");
}
