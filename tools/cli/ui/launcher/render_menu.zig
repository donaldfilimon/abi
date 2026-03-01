//! Menu list renderer for command launcher TUI.

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

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const items = state.items;
    const indices = state.filtered_indices.items;
    const visible = state.visible_rows;
    const start = state.scroll_offset;
    const end = @min(start + visible, indices.len);
    const inner = layout.frameInnerWidth(width);

    if (start > 0) {
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try renderClippedLine(term, th, chrome, inner, " ^ more above");
        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    const prefix_cols: usize = if (inner < 8) inner else 8;
    const row_budget = layout.safeSub(inner, prefix_cols);
    const show_desc = row_budget >= 12;
    const label_budget = if (row_budget == 0)
        0
    else if (show_desc)
        @min(@as(usize, 24), layout.safeSub(row_budget, 10))
    else
        row_budget;

    for (start..end) |i| {
        const idx = indices[i];
        const item = items[idx];
        const is_selected = i == state.selected;

        try term.write(chrome.frame);
        try term.write(box.v);
        try term.write(th.reset);

        if (inner >= 8) {
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
        } else if (inner > 0) {
            try writeRepeat(term, " ", inner);
        }

        const desc_budget = if (show_desc) layout.safeSub(row_budget, label_budget) else 0;
        const label = if (label_budget > 0)
            unicode.truncateToWidth(item.label, label_budget)
        else
            "";
        const label_w = unicode.displayWidth(label);
        if (is_selected) {
            try term.write(chrome.selection_bg);
            try term.write(chrome.selection_fg);
        }
        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        try writeClipped(term, label, label_budget);
        try term.write(th.reset);
        if (label_w < label_budget) {
            try writeRepeat(term, " ", label_budget - label_w);
        }

        if (desc_budget > 0 and inner > prefix_cols + label_budget) {
            try term.write(" ");
            const desc = unicode.truncateToWidth(item.description, desc_budget);
            const desc_w = unicode.displayWidth(desc);
            try term.write(if (is_selected) chrome.selection_fg else th.text_dim);
            try writeClipped(term, desc, desc_budget);
            try term.write(th.reset);
            if (desc_w < desc_budget) {
                try writeRepeat(term, " ", desc_budget - desc_w);
            }
            if (is_selected) try term.write(chrome.selection_bg);
            if (is_selected) try term.write(chrome.selection_fg);
        } else if (inner > 0) {
            if (is_selected) {
                try term.write(chrome.selection_bg);
            }
            if (inner > prefix_cols + label_w) {
                try writeRepeat(term, " ", layout.safeSub(inner, prefix_cols + label_w));
            }
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
        try renderClippedLine(term, th, chrome, inner, " v more below");
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

fn renderClippedLine(
    term: *tui.Terminal,
    th: *const tui.Theme,
    chrome: style_adapter.ChromeStyle,
    inner: usize,
    text: []const u8,
) !void {
    try term.write(chrome.text_dim);
    const clipped = unicode.truncateToWidth(text, inner);
    try writeClipped(term, clipped, inner);
    try term.write(th.reset);
    const used = unicode.displayWidth(clipped);
    if (used < inner) try writeRepeat(term, " ", inner - used);
}

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
