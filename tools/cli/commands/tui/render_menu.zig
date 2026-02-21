//! Menu list renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const items = state.items;
    const indices = state.filtered_indices.items;
    const visible = state.visible_rows;
    const start = state.scroll_offset;
    const end = @min(start + visible, indices.len);

    if (start > 0) {
        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write("   \xE2\x96\xB2 more above");
        try writeRepeat(term, " ", width -| 18);
        try term.write(th.reset);
        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    for (start..end) |i| {
        const idx = indices[i];
        const item = items[idx];
        const is_selected = i == state.selected;

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);

        if (is_selected) {
            try term.write(th.selection_bg);
            try term.write(th.selection_fg);
            try term.write(" \xE2\x96\xB8 ");
        } else {
            try term.write("   ");
        }

        if (item.shortcut) |num| {
            try term.write(th.text_dim);
            var buf: [2]u8 = undefined;
            buf[0] = '0' + num;
            buf[1] = 0;
            try term.write(buf[0..1]);
            try term.write(th.reset);
            if (is_selected) {
                try term.write(th.selection_bg);
                try term.write(th.selection_fg);
            }
            try term.write(" ");
        } else {
            try term.write("  ");
        }

        try term.write(item.categoryColor(th));
        if (is_selected) try term.write(th.bold);
        try term.write(item.label);
        try term.write(th.reset);
        if (is_selected) {
            try term.write(th.selection_bg);
            try term.write(th.selection_fg);
        }

        const label_w = unicode.displayWidth(item.label);
        const padding = 16 -| @min(label_w, 16);
        try writeRepeat(term, " ", padding);

        try term.write(th.text_dim);
        if (is_selected) try term.write(th.selection_fg);
        const desc_max = width -| 24;
        const desc_trunc = unicode.truncateToWidth(item.description, desc_max);
        const desc_w = unicode.displayWidth(desc_trunc);
        try term.write(desc_trunc);
        try term.write(th.reset);

        const used = 6 + label_w + padding + desc_w;
        if (used < width - 1) {
            try writeRepeat(term, " ", width - 1 - used);
        }

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    if (end < indices.len) {
        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write(th.text_dim);
        try term.write("   \xE2\x96\xBC more below");
        try writeRepeat(term, " ", width -| 18);
        try term.write(th.reset);
        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    const rendered = end - start +
        @as(usize, if (start > 0) 1 else 0) +
        @as(usize, if (end < indices.len) 1 else 0);
    if (rendered < visible) {
        for (0..(visible - rendered)) |_| {
            try term.write(th.border);
            try term.write(box.v);
            try term.write(th.reset);
            try writeRepeat(term, " ", width - 2);
            try term.write(th.border);
            try term.write(box.v);
            try term.write(th.reset);
            try term.write("\n");
        }
    }
}
