//! Status and help bar renderer for command launcher TUI.

const std = @import("std");
const tui = @import("../core/mod.zig");
const layout = @import("layout.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");
const unicode = tui.unicode;

const TuiState = state_mod.TuiState;
const box = types.box;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn renderStatus(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = layout.frameInnerWidth(width);
    var budget = inner;

    var cpu_buf: [16]u8 = undefined;
    var shown_buf: [32]u8 = undefined;
    const cpu_count_text = std.fmt.bufPrint(&cpu_buf, "{d}", .{std.Thread.getCpuCount() catch 1}) catch "?";
    const shown_text = std.fmt.bufPrint(&shown_buf, "{d}/{d}", .{
        state.filtered_indices.items.len,
        state.items.len,
    }) catch "?/?";

    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);

    try writeStatusField(
        term,
        chrome,
        th,
        &budget,
        "platform",
        tui.Terminal.platformName(),
        1,
    );
    try writeStatusField(
        term,
        chrome,
        th,
        &budget,
        "cpu",
        cpu_count_text,
        2,
    );
    try writeStatusField(
        term,
        chrome,
        th,
        &budget,
        "items",
        shown_text,
        2,
    );
    try writeStatusField(
        term,
        chrome,
        th,
        &budget,
        "theme",
        state.theme_manager.current.name,
        2,
    );

    if (budget > 0) {
        try writeRepeat(term, " ", budget);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
}

fn writeStatusField(
    term: *tui.Terminal,
    chrome: style_adapter.ChromeStyle,
    th: *const tui.Theme,
    budget: *usize,
    label: []const u8,
    value: []const u8,
    leading: usize,
) !void {
    const chip_width = unicode.displayWidth(label) + 2;
    const minimum_needed = leading + chip_width;
    if (budget.* < minimum_needed) return;

    try writeRepeat(term, " ", leading);
    budget.* -|= leading;

    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" ");
    try term.write(label);
    try term.write(" ");
    try term.write(th.reset);
    budget.* -|= chip_width;

    if (budget.* == 0) return;

    try term.write(" ");
    budget.* -|= 1;
    const clipped = unicode.truncateToWidth(value, budget.*);
    const used = unicode.displayWidth(clipped);
    try term.write(clipped);
    budget.* -|= used;
}

pub fn renderHelp(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = layout.frameInnerWidth(width);

    try term.write(chrome.frame);
    try term.write(box.bl);
    try writeRepeat(term, box.h, inner);
    try term.write(box.br);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(" ");
    const compact_mode = inner <= 26;
    if (state.search_mode) {
        if (compact_mode) {
            if (inner > 10) {
                try writeKeyHint(term, chrome, th, "Esc", "cancel");
                if (inner > 18) {
                    try term.write(" ");
                    try writeKeyHint(term, chrome, th, "Enter", "run");
                }
            } else {
                try writeKeyHint(term, chrome, th, "Esc", "cancel");
            }
        } else if (inner >= 42) {
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
        if (compact_mode) {
            if (inner >= 14) {
                try writeKeyHint(term, chrome, th, "q", "quit");
                try term.write(" ");
                try writeKeyHint(term, chrome, th, "?", "help");
            } else {
                try writeKeyHint(term, chrome, th, "q", "quit");
            }
        } else if (inner >= 64) {
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
