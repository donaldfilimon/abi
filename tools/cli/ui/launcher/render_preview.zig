//! Command preview renderer for command launcher TUI.

const tui = @import("../core/mod.zig");
const state_mod = @import("state.zig");
const style_adapter = @import("style_adapter.zig");

const TuiState = state_mod.TuiState;
const unicode = tui.unicode;
const writeRepeat = tui.render_utils.writeRepeat;

pub fn render(term: *tui.Terminal, state: *TuiState, width: usize) !void {
    const th = state.theme();
    const chrome = style_adapter.launcher(th);
    const item = state.selectedItem() orelse return;
    const inner = width -| 2;

    try term.write("\n");
    try term.write(chrome.frame);
    try term.write("╭");
    try writeRepeat(term, "─", inner);
    try term.write("╮");
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" PREVIEW ");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(item.categoryColor(th));
    try term.write(th.bold);
    const title_max = inner -| 12;
    const title = unicode.truncateToWidth(item.label, title_max);
    try term.write(title);
    try term.write(th.reset);
    const title_w = 11 + unicode.displayWidth(title);
    if (title_w < inner) try writeRepeat(term, " ", inner - title_w);
    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.frame);
    try term.write("├");
    try writeRepeat(term, "─", inner);
    try term.write("┤");
    try term.write(th.reset);
    try term.write("\n");

    try renderSection(term, th, chrome, "DESCRIPTION", item.description, inner);
    if (item.usage.len > 0) {
        try renderSection(term, th, chrome, "USAGE", item.usage, inner);
    }

    if (item.examples.len > 0) {
        try renderSectionHeader(term, th, chrome, "EXAMPLES", inner);
        for (item.examples) |example| {
            try renderBodyLine(term, th, chrome, example, inner, "$ ");
        }
    }

    if (item.related.len > 0) {
        try renderSectionHeader(term, th, chrome, "RELATED", inner);
        for (item.related) |related| {
            try renderBodyLine(term, th, chrome, related, inner, "- ");
        }
    }

    try term.write(chrome.frame);
    try term.write("╰");
    try writeRepeat(term, "─", inner);
    try term.write("╯");
    try term.write(th.reset);
    try term.write("\n");

    try term.write(chrome.subtitle);
    try term.write(" ");
    try term.write(chrome.keycap_bg);
    try term.write(chrome.keycap_fg);
    try term.write(" Enter ");
    try term.write(th.reset);
    try term.write(chrome.subtitle);
    try term.write(" run  ");
    try term.write(chrome.keycap_bg);
    try term.write(chrome.keycap_fg);
    try term.write(" Esc ");
    try term.write(th.reset);
    try term.write(chrome.subtitle);
    try term.write(" back");
    try term.write(th.reset);
    try term.write("\n");
}

fn renderSection(
    term: *tui.Terminal,
    th: *const tui.Theme,
    chrome: style_adapter.ChromeStyle,
    title: []const u8,
    body: []const u8,
    inner: usize,
) !void {
    try renderSectionHeader(term, th, chrome, title, inner);
    try renderBodyLine(term, th, chrome, body, inner, "");
}

fn renderSectionHeader(
    term: *tui.Terminal,
    th: *const tui.Theme,
    chrome: style_adapter.ChromeStyle,
    title: []const u8,
    inner: usize,
) !void {
    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write(" ");
    try term.write(chrome.chip_bg);
    try term.write(chrome.chip_fg);
    try term.write(" ");
    try term.write(title);
    try term.write(" ");
    try term.write(th.reset);
    const used = title.len + 4;
    if (used < inner) try writeRepeat(term, " ", inner - used);
    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write("\n");
}

fn renderBodyLine(
    term: *tui.Terminal,
    th: *const tui.Theme,
    chrome: style_adapter.ChromeStyle,
    text: []const u8,
    inner: usize,
    prefix: []const u8,
) !void {
    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write(" ");
    if (prefix.len > 0) {
        try term.write(chrome.title);
        try term.write(prefix);
        try term.write(th.reset);
    }
    const body_max = inner -| 1 -| prefix.len;
    const body = unicode.truncateToWidth(text, body_max);
    try term.write(th.text_dim);
    try term.write(body);
    try term.write(th.reset);
    const used = 1 + prefix.len + unicode.displayWidth(body);
    if (used < inner) try writeRepeat(term, " ", inner - used);
    try term.write(chrome.frame);
    try term.write("│");
    try term.write(th.reset);
    try term.write("\n");
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
