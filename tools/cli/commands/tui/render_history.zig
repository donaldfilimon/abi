//! History panel renderer for command launcher TUI.

const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");
const menu_mod = @import("menu.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();

    // Header
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.bold);
    try term.write(th.accent);
    try term.write("Recent Commands:");
    try term.write(th.reset);

    const header_used: usize = 2 + 1 + unicode.displayWidth("Recent Commands:");
    if (header_used < width - 1) {
        try writeRepeat(term, " ", width - 1 - header_used);
    }
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);
    try term.write("\n");

    // Show up to 5 recent commands
    const max_show = @min(state.history.items.len, 5);
    for (0..max_show) |i| {
        const entry = state.history.items[i];
        const cmd_name = menu_mod.commandName(entry.command);

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("   ");
        try term.write(th.text_dim);
        var num_buf: [2]u8 = undefined;
        num_buf[0] = '1' + @as(u8, @intCast(i));
        num_buf[1] = '.';
        try term.write(&num_buf);
        try term.write(" ");
        try term.write(th.reset);
        try term.write(th.secondary);
        try term.write(cmd_name);
        try term.write(th.reset);

        const cmd_w = unicode.displayWidth(cmd_name);
        const used = 7 + cmd_w;
        if (used < width - 1) {
            try writeRepeat(term, " ", width - 1 - used);
        }

        try term.write(th.border);
        try term.write(box.v);
        try term.write(th.reset);
        try term.write("\n");
    }

    // Separator
    try term.write(th.border);
    try term.write(box.lsep);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.rsep);
    try term.write(th.reset);
    try term.write("\n");
}
