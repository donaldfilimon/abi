//! Search bar and completion dropdown renderer.

const std = @import("std");
const tui = @import("../core/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn renderBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = width -| 2;

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");

    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" SEARCH ");
    try term.write(th.reset);
    try term.write(" ");

    const query = state.search_buffer[0..state.search_len];
    const query_max = inner -| 13;
    const query_display = unicode.truncateToWidth(query, query_max);
    try term.write(th.text);
    try term.write(query_display);
    try term.write(th.reset);

    if (state.search_mode and unicode.displayWidth(query_display) < query_max) {
        try term.write(chrome.selection_rail);
        try term.write("_");
        try term.write(th.reset);
    }

    const query_w = unicode.displayWidth(query_display);
    const used = 12 + query_w + @as(usize, if (state.search_mode and query_w < query_max) 1 else 0);
    if (used < inner) try writeRepeat(term, " ", inner - used);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.frame);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, inner);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");
}

pub fn renderDropdown(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    if (!state.completion_state.active) return;

    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const suggestions = state.completion_state.suggestions.items;
    const max_show = @min(state.completion_state.max_visible, suggestions.len);
    if (max_show == 0) return;

    const inner = width -| 2;
    const dropdown_width = @min(@as(usize, 54), inner -| 4);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("  ");
    try term.write(chrome.frame);
    try term.write("╭");
    try writeRepeat(term, "─", dropdown_width);
    try term.write("╮");
    try term.write(th.reset);
    const pad_top = inner -| (dropdown_width + 2);
    if (pad_top > 0) try writeRepeat(term, " ", pad_top);
    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    for (0..max_show) |i| {
        const suggestion = suggestions[i];
        const item = state.items[suggestion.item_index];
        const is_selected = i == state.completion_state.selected_suggestion;

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("  ");
        try term.write(chrome.frame);
        try term.write("│");
        try term.write(th.reset);

        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
            try term.write(" ▸ ");
        } else {
            try term.write("   ");
        }

        try term.write(th.text_muted);
        try term.write(suggestion.match_type.indicator());
        try term.write(" ");
        try term.write(th.reset);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }

        const label_max = @min(@as(usize, 18), dropdown_width -| 18);
        const label = unicode.truncateToWidth(item.label, label_max);
        const label_w = unicode.displayWidth(label);
        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        try term.write(label);
        try term.write(th.reset);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (label_w < label_max) try writeRepeat(term, " ", label_max - label_w);
        try term.write(" ");

        const desc_max = dropdown_width -| 8 -| label_max;
        const desc = unicode.truncateToWidth(item.description, desc_max);
        const desc_w = unicode.displayWidth(desc);
        try term.write(if (is_selected) chrome.selection_fg else th.text_dim);
        try term.write(desc);
        try term.write(th.reset);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (desc_w < desc_max) try writeRepeat(term, " ", desc_max - desc_w);

        if (is_selected) try term.write(th.reset);
        try term.write(chrome.frame);
        try term.write("│");
        try term.write(th.reset);

        const pad = inner -| (dropdown_width + 2);
        if (pad > 0) try writeRepeat(term, " ", pad);
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("  ");
    try term.write(chrome.frame);
    try term.write("╰");
    try writeRepeat(term, "─", dropdown_width);
    try term.write("╯");
    try term.write(th.reset);

    if (suggestions.len > max_show) {
        var buf: [32]u8 = undefined;
        const summary = std.fmt.bufPrint(&buf, " {d}/{d}", .{ max_show, suggestions.len }) catch "";
        const summary_w = unicode.displayWidth(summary);
        if (summary_w < inner -| (dropdown_width + 2)) {
            try term.write(th.text_dim);
            try term.write(summary);
            try term.write(th.reset);
            const pad = inner -| (dropdown_width + 2 + summary_w);
            if (pad > 0) try writeRepeat(term, " ", pad);
        } else {
            const pad = inner -| (dropdown_width + 2);
            if (pad > 0) try writeRepeat(term, " ", pad);
        }
    } else {
        const pad = inner -| (dropdown_width + 2);
        if (pad > 0) try writeRepeat(term, " ", pad);
    }

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}

test {
    std.testing.refAllDecls(@This());
}
