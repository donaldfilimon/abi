//! Search bar and completion dropdown renderer.

const std = @import("std");
const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn renderBar(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);

    try term.write(" ");
    if (state.search_mode) {
        try term.write(th.accent);
    } else {
        try term.write(th.text_dim);
    }
    try term.write("/");
    try term.write(th.reset);
    try term.write(" ");

    const query = state.search_buffer[0..state.search_len];
    try term.write(query);

    if (state.search_mode) {
        try term.write(th.accent);
        try term.write("_");
        try term.write(th.reset);
    }

    const query_w = unicode.displayWidth(query);
    const used = 4 + query_w + @as(usize, if (state.search_mode) 1 else 0);
    if (used < width - 1) {
        try writeRepeat(term, " ", width - 1 - used);
    }

    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Separator
    try term.write(th.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");
}

pub fn renderDropdown(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    if (!state.completion_state.active) return;

    const th = state.theme();
    const suggestions = state.completion_state.suggestions.items;
    const max_show = @min(state.completion_state.max_visible, suggestions.len);

    if (max_show == 0) return;

    const dropdown_width = @min(50, width -| 8);

    // Top border of dropdown
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("   ");
    try term.write(th.text_dim);
    try term.write("\u{256d}");
    try writeRepeat(term, "\u{2500}", dropdown_width);
    try term.write("\u{256e}");
    try term.write(th.reset);

    const remaining = width -| (4 + dropdown_width + 2);
    try writeRepeat(term, " ", remaining);
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Suggestions
    for (0..max_show) |i| {
        const suggestion = suggestions[i];
        const item = state.items[suggestion.item_index];
        const is_selected = i == state.completion_state.selected_suggestion;

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("   ");
        try term.write(th.text_dim);
        try term.write("\u{2502}");
        try term.write(th.reset);

        if (is_selected) {
            try term.write(th.selection_bg);
            try term.write(th.selection_fg);
            try term.write(" \xE2\x96\xB8 ");
        } else {
            try term.write("   ");
        }

        // Match type indicator
        try term.write(th.text_muted);
        try term.write(suggestion.match_type.indicator());
        try term.write(" ");
        try term.write(th.reset);

        if (is_selected) {
            try term.write(th.selection_bg);
            try term.write(th.selection_fg);
        }

        // Label with category color — truncate to 18 columns
        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        const label_trunc = unicode.truncateToWidth(item.label, 18);
        const label_w = unicode.displayWidth(label_trunc);
        try term.write(label_trunc);
        try term.write(th.reset);

        if (is_selected) {
            try term.write(th.selection_bg);
            try term.write(th.selection_fg);
        }

        // Padding after label
        const label_pad = 18 -| label_w;
        try writeRepeat(term, " ", label_pad);

        // Short description — truncate to fit
        try term.write(th.text_dim);
        const desc_max_cols = dropdown_width -| 28;
        const desc_trunc = unicode.truncateToWidth(item.description, desc_max_cols);
        const desc_w = unicode.displayWidth(desc_trunc);
        try term.write(desc_trunc);
        try term.write(th.reset);

        // Fill rest of dropdown line
        const used_in_dropdown = 5 + label_w + label_pad + desc_w;
        if (used_in_dropdown < dropdown_width) {
            try writeRepeat(term, " ", dropdown_width - used_in_dropdown);
        }

        try term.write(th.text_dim);
        try term.write("\u{2502}");
        try term.write(th.reset);

        const final_pad = width -| (4 + dropdown_width + 2);
        try writeRepeat(term, " ", final_pad);

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    // Bottom border with count hint
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("   ");
    try term.write(th.text_dim);
    try term.write("\u{2570}");

    if (suggestions.len > max_show) {
        var count_buf: [16]u8 = undefined;
        const count_str = std.fmt.bufPrint(&count_buf, " {d}/{d} ", .{
            max_show,
            suggestions.len,
        }) catch "";
        const hint_len = unicode.displayWidth(count_str);
        const bar_len = (dropdown_width -| hint_len) / 2;
        try writeRepeat(term, "\u{2500}", bar_len);
        try term.write(count_str);
        try writeRepeat(term, "\u{2500}", dropdown_width -| bar_len -| hint_len);
    } else {
        try writeRepeat(term, "\u{2500}", dropdown_width);
    }

    try term.write("\u{256f}");
    try term.write(th.reset);

    const remaining2 = width -| (4 + dropdown_width + 2);
    try writeRepeat(term, " ", remaining2);
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");
}
