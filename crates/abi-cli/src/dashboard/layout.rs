//! Text-layout primitives for the diagnostics dashboard.
//!
//! Fixed-width cell fitting, box-drawing borders, and the focused-pane
//! highlight. Pure string building: nothing here reads state or touches a
//! terminal.

use super::{DIAG_WIDTH, LABEL_WIDTH, VALUE_WIDTH};

pub(super) fn fit(s: &str, width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= width {
        let mut out = s.to_string();
        out.extend(std::iter::repeat_n(' ', width - chars.len()));
        out
    } else if width == 0 {
        String::new()
    } else if width == 1 {
        "~".to_string()
    } else {
        let mut out: String = chars.into_iter().take(width - 1).collect();
        out.push('~');
        out
    }
}

fn append_rule(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push('─');
    }
}

pub(super) fn append_border(out: &mut String, left: &str, title: &str, right: &str) {
    out.push_str(left);
    if title.is_empty() {
        append_rule(out, DIAG_WIDTH);
    } else {
        out.push(' ');
        let title_w = title.chars().count().min(DIAG_WIDTH.saturating_sub(4));
        out.push_str(&fit(title, title_w));
        out.push(' ');
        let used = title_w + 2;
        if used < DIAG_WIDTH {
            append_rule(out, DIAG_WIDTH - used);
        }
    }
    out.push_str(right);
    out.push('\n');
}

/// Reverse video plus bold red — how the dashboard marks the focused pane.
///
/// Kept as one constant because it is contract: `tools/run_tui_smoke.sh` asserts
/// this exact byte sequence precedes the selected pane's title.
pub(super) const FOCUS_STYLE: &str = "\x1b[7m\x1b[1;31m";
/// SGR reset, closing [`FOCUS_STYLE`].
pub(super) const STYLE_RESET: &str = "\x1b[0m";

/// Render a pane's top border, highlighting it when it is the focused pane.
///
/// The reset lands *before* the newline: leaving it after would let reverse video
/// bleed into the following row, which is the usual way this kind of highlight
/// goes wrong. With `color` false (`--plain` / `--no-color`) no escapes are
/// emitted at all, which is what makes those flags meaningful for the text
/// render rather than metadata that only shows up in `--json`.
pub(super) fn append_pane_header(out: &mut String, title: &str, selected: bool, color: bool) {
    let mut line = String::new();
    append_border(&mut line, "┌", title, "┐");

    if selected && color {
        out.push_str(FOCUS_STYLE);
        out.push_str(line.trim_end_matches('\n'));
        out.push_str(STYLE_RESET);
        out.push('\n');
    } else {
        out.push_str(&line);
    }
}

pub(super) fn append_row(out: &mut String, label: &str, value: &str) {
    out.push_str("│ ");
    out.push_str(&fit(label, LABEL_WIDTH));
    out.push(' ');
    out.push_str(&fit(value, VALUE_WIDTH));
    out.push_str(" │\n");
}

pub(super) fn append_metric(out: &mut String, label: &str, value: usize) {
    append_row(out, label, &value.to_string());
}
