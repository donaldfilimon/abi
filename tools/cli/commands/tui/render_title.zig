//! Title bar renderer for command launcher TUI.

const abi = @import("abi");
const tui = @import("../../tui/mod.zig");
const types = @import("types.zig");
const state_mod = @import("state.zig");

const TuiState = state_mod.TuiState;
const box = types.box;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();

    // Top border
    try term.write(th.border);
    try term.write(box.tl);
    try writeRepeat(term, box.h, width - 2);
    try term.write(box.tr);
    try term.write(th.reset);
    try term.write("\n");

    // Title
    try term.write(th.border);
    try term.write(box.v);
    try term.write(th.reset);

    const title = " ABI Framework ";
    const version_str = abi.version();
    const theme_indicator = state.theme_manager.current.name;

    const title_w = unicode.displayWidth(title);
    const version_w = unicode.displayWidth(version_str);
    const theme_w = unicode.displayWidth(theme_indicator);
    const title_len = title_w + version_w + theme_w + 6; // " vX.X.X [theme]"
    const left_pad = (width - 2 -| title_len) / 2;
    const right_pad = width - 2 -| title_len -| left_pad;

    try writeRepeat(term, " ", left_pad);
    try term.write(th.bold);
    try term.write(th.primary);
    try term.write(title);
    try term.write(th.text_dim);
    try term.write("v");
    try term.write(version_str);
    try term.write(th.reset);
    try term.write(" ");
    try term.write(th.text_muted);
    try term.write("[");
    try term.write(theme_indicator);
    try term.write("]");
    try term.write(th.reset);
    try writeRepeat(term, " ", right_pad);

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
