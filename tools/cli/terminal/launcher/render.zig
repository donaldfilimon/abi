//! TUI frame renderer orchestrator.
//!
//! Keeps top-level render flow while delegating individual sections
//! to dedicated renderer modules.

const tui = @import("../mod.zig");
const state_mod = @import("state");
const tui_layout = @import("layout");

const title_renderer = @import("render_title");
const notification_renderer = @import("render_notification");
const search_renderer = @import("render_search");
const history_renderer = @import("render_history");
const menu_renderer = @import("render_menu");
const footer_renderer = @import("render_footer");
const preview_renderer = @import("render_preview");

const TuiState = state_mod.TuiState;

pub fn renderFrame(state: *TuiState) !void {
    const term = state.terminal;
    const width = @as(usize, @intCast(state.term_size.cols));

    if (state.preview_mode) {
        try preview_renderer.render(term, state, width);
        return;
    }

    try title_renderer.render(term, state, width);

    if (state.notification) |msg| {
        try notification_renderer.render(term, state, width, msg);
    }

    if (state.search_mode or state.search_len > 0) {
        try search_renderer.renderBar(term, state, width);

        if (state.search_mode and state.completion_state.active) {
            try search_renderer.renderDropdown(term, state, width);
        }
    }

    if (state.show_history and state.history.items.len > 0) {
        try history_renderer.render(term, state, width);
    }

    try menu_renderer.render(term, state, width);
    try footer_renderer.renderStatus(term, state, width);
    try footer_renderer.renderHelp(term, state, width);
}

test {
    _ = @import("render_title");
    _ = @import("render_notification");
    _ = @import("render_search");
    _ = @import("render_history");
    _ = @import("render_menu");
    _ = @import("render_footer");
    _ = @import("render_preview");
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
