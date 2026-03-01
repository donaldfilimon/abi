//! Search bar and completion dropdown renderer.

const std = @import("std");
const tui = @import("../core/mod.zig");
const layout = @import("layout.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;
const writeClipped = tui.render_utils.writeClipped;

pub fn renderBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const inner = layout.frameInnerWidth(width);

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");

    const show_search_chip = inner >= 12;
    if (show_search_chip) {
        try term.write(chrome.chip_bg);
        try term.write(chrome.chip_fg);
        try term.write(" SEARCH ");
        try term.write(th.reset);
        try term.write(" ");
    }

    const query = state.search_buffer[0..state.search_len];
    const fixed_prefix = if (show_search_chip) 12 else 1;
    const query_max = layout.safeSub(inner, fixed_prefix);
    const query_display = unicode.truncateToWidth(query, query_max);
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try writeClipped(term, query_display, query_max);
    try term.write(th.reset);

    const cursor_active = state.search_mode and unicode.displayWidth(query_display) < query_max;
    if (cursor_active) {
        try term.write(chrome.selection_rail);
        try term.write("_");
        try term.write(th.reset);
    }

    const query_w = unicode.displayWidth(query_display);
    const used = fixed_prefix -| 1 + query_w + @as(usize, if (cursor_active) 1 else 0);
    const pad = layout.safeSub(inner, used);
    if (pad > 0) try writeRepeat(term, " ", pad);

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

    const inner = layout.frameInnerWidth(width);
    const dropdown_width = @min(@as(usize, 54), layout.safeSub(inner, 4));
    if (dropdown_width < 4) return;

    try term.write(chrome.frame);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("  ");
    try term.write(chrome.frame);
    try term.write("╭");
    try writeRepeat(term, "─", dropdown_width);
    try term.write("╮");
    try term.write(th.reset);
    const pad_top = layout.safeSub(inner, dropdown_width + 2);
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

        const label_budget = @max(@as(usize, 4), @min(@as(usize, 18), dropdown_width -| 18));
        const label = unicode.truncateToWidth(item.label, label_budget);
        const label_w = unicode.displayWidth(label);
        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        try writeClipped(term, label, label_budget);
        try term.write(th.reset);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (label_w < label_budget) {
            try writeRepeat(term, " ", label_budget - label_w);
        }
        try term.write(" ");

        const desc_max = layout.safeSub(dropdown_width, label_budget + 8);
        const desc = unicode.truncateToWidth(item.description, desc_max);
        const desc_w = unicode.displayWidth(desc);
        try term.write(if (is_selected) chrome.selection_fg else th.text_dim);
        try writeClipped(term, desc, desc_max);
        try term.write(th.reset);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (desc_w < desc_max) {
            try writeRepeat(term, " ", desc_max - desc_w);
        }

        if (is_selected) try term.write(th.reset);
        try term.write(chrome.frame);
        try term.write("│");
        try term.write(th.reset);

        const pad = layout.safeSub(inner, dropdown_width + 2);
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
        const summary_budget = layout.safeSub(inner, dropdown_width + 2);
        if (summary_w < summary_budget) {
            const pad = layout.safeSub(summary_budget, summary_w);
            try term.write(chrome.frame);
            try term.write(" ");
            try term.write(th.text_dim);
            try writeClipped(term, summary, summary_budget);
            try term.write(th.reset);
            if (pad > 0) try writeRepeat(term, " ", pad);
        } else {
            const pad = summary_budget;
            if (pad > 0) try writeRepeat(term, " ", pad);
        }
    } else {
        const pad = layout.safeSub(inner, dropdown_width + 2);
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
