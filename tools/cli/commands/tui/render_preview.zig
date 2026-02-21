//! Command preview renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const item = state.selectedItem() orelse return;

    try term.write("\n");
    try term.write(th.primary);
    try term.write(tui.widgets.box.dtl);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(tui.widgets.box.dtr);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.bold);
    try term.write(item.categoryColor(th));
    try term.write(item.label);
    try term.write(th.reset);

    const title_w = unicode.displayWidth(item.label) + 2;
    if (title_w < width - 1) {
        try writeRepeat(term, " ", width - 1 - title_w);
    }
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.text_dim);
    try term.write(item.description);
    try term.write(th.reset);

    const desc_w = unicode.displayWidth(item.description) + 2;
    if (desc_w < width - 1) {
        try writeRepeat(term, " ", width - 1 - desc_w);
    }
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(th.primary);
    try term.write(box.lsep);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");

    if (item.usage.len > 0) {
        try renderSection(term, th, "Usage", width);
        try renderLine(term, th, item.usage, width);
    }

    if (item.examples.len > 0) {
        try renderSection(term, th, "Examples", width);
        for (item.examples) |example| {
            try term.write(th.primary);
            try term.write(tui.widgets.box.dv);
            try term.write(th.reset);
            try term.write("   ");
            try term.write(th.success);
            try term.write("$ ");
            try term.write(th.reset);
            try term.write(example);

            const ex_w = unicode.displayWidth(example) + 5;
            if (ex_w < width - 1) {
                try writeRepeat(term, " ", width - 1 - ex_w);
            }
            try term.write(th.primary);
            try term.write(tui.widgets.box.dv);
            try term.write(th.reset);
            try term.write("\n");
        }
    }

    if (item.related.len > 0) {
        try renderSection(term, th, "Related", width);
        try term.write(th.primary);
        try term.write(tui.widgets.box.dv);
        try term.write(th.reset);
        try term.write("   ");

        var total_w: usize = 3;
        for (item.related, 0..) |rel, i| {
            if (i > 0) {
                try term.write(", ");
                total_w += 2;
            }
            try term.write(th.accent);
            try term.write(rel);
            try term.write(th.reset);
            total_w += unicode.displayWidth(rel);
        }

        if (total_w < width - 1) {
            try writeRepeat(term, " ", width - 1 - total_w);
        }
        try term.write(th.primary);
        try term.write(tui.widgets.box.dv);
        try term.write(th.reset);
        try term.write("\n");
    }

    try term.write(th.primary);
    try term.write(tui.widgets.box.dbl);
    try writeRepeat(term, tui.widgets.box.dh, width - 2);
    try term.write(tui.widgets.box.dbr);
    try term.write(th.reset);
    try term.write("\n\n");

    try term.write(th.text_dim);
    try term.write(" Press ");
    try term.write(th.reset);
    try term.write(th.accent);
    try term.write("Enter");
    try term.write(th.reset);
    try term.write(th.text_dim);
    try term.write(" to run, ");
    try term.write(th.reset);
    try term.write(th.accent);
    try term.write("Esc");
    try term.write(th.reset);
    try term.write(th.text_dim);
    try term.write(" to go back\n");
    try term.write(th.reset);
}

fn renderSection(term: *tui.Terminal, th: *const tui.Theme, title: []const u8, width: usize) !void {
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try writeRepeat(term, " ", width - 2);
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("\n");

    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.bold);
    try term.write(th.primary);
    try term.write(title);
    try term.write(":");
    try term.write(th.reset);

    const sect_w = unicode.displayWidth(title) + 3;
    if (sect_w < width - 1) {
        try writeRepeat(term, " ", width - 1 - sect_w);
    }
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("\n");
}

fn renderLine(term: *tui.Terminal, th: *const tui.Theme, text: []const u8, width: usize) !void {
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("   ");
    try term.write(text);

    const line_w = unicode.displayWidth(text) + 3;
    if (line_w < width - 1) {
        try writeRepeat(term, " ", width - 1 - line_w);
    }
    try term.write(th.primary);
    try term.write(tui.widgets.box.dv);
    try term.write(th.reset);
    try term.write("\n");
}
