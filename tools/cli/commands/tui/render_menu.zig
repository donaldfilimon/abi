//! Menu list renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const items = state.items;
    const indices = state.filtered_indices.items;
    const visible = state.visible_rows;
    const start = state.scroll_offset;
    const end = @min(start + visible, indices.len);
    const inner = width -| 2;

    if (start > 0) {
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write("  ^ more above");
        try writeRepeat(term, " ", inner -| 13);
        try term.write(th.reset);
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    const prefix_cols: usize = 8;
    const label_cols: usize = @max(@as(usize, 12), @min(@as(usize, 24), inner -| 20));
    const desc_cols: usize = inner -| prefix_cols -| label_cols;

    for (start..end) |i| {
        const idx = indices[i];
        const item = items[idx];
        const is_selected = i == state.selected;

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);

        if (is_selected) {
            try term.write(chrome.selection_rail);
            try term.write("▌ ");
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        } else {
            try term.write("  ");
        }

        if (item.shortcut) |num| {
            try term.write(chrome.keycap_bg);
            try term.write(chrome.keycap_fg);
            try term.write(" ");
            var buf: [1]u8 = undefined;
            buf[0] = '0' + num;
            try term.write(&buf);
            try term.write(" ");
            try term.write(th.reset);
            if (is_selected) {
                try term.write(chrome.selection_bg);
                try term.write(chrome.selection_fg);
            }
        } else {
            try term.write("    ");
        }

        try term.write(" ");

        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        const label = unicode.truncateToWidth(item.label, label_cols);
        const label_w = unicode.displayWidth(label);
        try term.write(label);
        try term.write(th.reset);

        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (label_w < label_cols) {
            try writeRepeat(term, " ", label_cols - label_w);
        }

        try term.write(" ");
        if (is_selected) {
            try term.write(chrome.selection_fg);
        } else {
            try term.write(th.text_dim);
        }
        const desc = unicode.truncateToWidth(item.description, desc_cols);
        const desc_w = unicode.displayWidth(desc);
        try term.write(desc);
        try term.write(th.reset);

        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        if (desc_w < desc_cols) {
            try writeRepeat(term, " ", desc_cols - desc_w);
        }

        if (is_selected) {
            try term.write(th.reset);
        }

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    if (end < indices.len) {
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write("  v more below");
        try writeRepeat(term, " ", inner -| 13);
        try term.write(th.reset);
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    const rendered = end - start +
        @as(usize, if (start > 0) 1 else 0) +
        @as(usize, if (end < indices.len) 1 else 0);
    if (rendered < visible) {
        for (0..(visible - rendered)) |_| {
            try term.write(chrome.frame);
            try term.write(box.v);
            try term.write(th.reset);
            try writeRepeat(term, " ", inner);
            try term.write(chrome.frame);
            try term.write(box.v);
            try term.write(th.reset);
            try term.write("\n");
        }
    }
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
