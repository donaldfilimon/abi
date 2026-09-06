//! Behavioural tests for the diagnostics dashboard.

use serde_json::Value;

use super::input::{InputAction, apply_interactive_input, cycle_pane};
use super::layout::{FOCUS_STYLE, STYLE_RESET};
use super::options::pane_index_for_token;
use super::render::pane_at_position;
use super::*;

#[test]
fn list_panes_prints_all_five() {
    let outcome = run(&["--list-panes".to_owned()]);
    assert_eq!(outcome.exit_code, 0);
    assert!(outcome.stderr.contains("Dashboard panes:"));
    for name in ["system", "plugins", "storage", "scheduler", "memory"] {
        assert!(outcome.stderr.contains(name), "missing {name}");
    }
}

#[test]
fn once_renders_all_five_panels() {
    let outcome = run(&["--once".to_owned(), "--plain".to_owned()]);
    assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
    assert!(outcome.stderr.contains("ABI Diagnostics Dashboard"));
    assert!(outcome.stderr.contains("System"));
    assert!(outcome.stderr.contains("Plugins"));
    assert!(outcome.stderr.contains("WDBX Storage"));
    assert!(outcome.stderr.contains("Scheduler"));
    assert!(outcome.stderr.contains("Memory"));
    assert!(outcome.stderr.contains("Registered"));
    // Honest GPU disclosure
    assert!(outcome.stderr.contains("accelerated"));
    assert!(outcome.stderr.contains("native linked"));
    if !abi_gpu::metal_kernels::kernels_active() {
        assert!(!outcome.stderr.contains("native linked             yes"));
    }
}

#[test]
fn json_snapshot_is_parseable_and_claim_honest() {
    let outcome = run(&["--json".to_owned()]);
    assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
    let v: Value = serde_json::from_str(outcome.stderr.trim()).expect("json");
    assert_eq!(v["type"], "abi.dashboard");
    assert_eq!(v["plugins"]["count"], 16);
    let linked = abi_gpu::metal_kernels::kernels_linked();
    let active = abi_gpu::metal_kernels::kernels_active();
    assert_eq!(v["gpu"]["linked"], linked);
    assert_eq!(v["gpu"]["accelerated"], active);
    // Health is "nominal" only when kernels are active (linked∧init).
    assert_eq!(v["health"], if active { "nominal" } else { "cpu" });
    assert_eq!(v["scheduler"]["completed"], 2);
    assert_eq!(v["wdbx"]["blocks"], 0);
    assert_eq!(v["wdbx"]["vectors"], 0);
}

#[test]
fn selected_pane_is_highlighted_and_plain_suppresses_it() {
    // The focused pane must be visually marked. Zig did this and the Rust
    // port initially dropped it: `color` was stored and reported in `--json`
    // but never used to emit anything, so `--pane` had no visible effect and
    // `--plain` was a no-op for the text render.
    let colored = run(&["--pane".to_owned(), "memory".to_owned()]);
    assert_eq!(colored.exit_code, 0, "{}", colored.stderr);
    assert!(
        colored.stderr.contains(&format!("{FOCUS_STYLE}┌ Memory")),
        "focused pane must carry the highlight"
    );
    assert!(
        colored.stderr.contains(STYLE_RESET),
        "the highlight must be reset so it does not bleed"
    );
    // Unfocused panes stay unstyled.
    assert!(colored.stderr.contains("┌ System"));
    assert!(!colored.stderr.contains(&format!("{FOCUS_STYLE}┌ System")));

    let plain = run(&[
        "--pane".to_owned(),
        "memory".to_owned(),
        "--plain".to_owned(),
    ]);
    assert_eq!(plain.exit_code, 0, "{}", plain.stderr);
    assert!(
        !plain.stderr.contains(FOCUS_STYLE),
        "--plain must emit no escapes"
    );
    assert!(plain.stderr.contains("┌ Memory"));
}

#[test]
fn highlight_reset_precedes_the_newline() {
    // Resetting after the newline would let reverse video bleed across the
    // following row, which is the usual way this bug shows up in a terminal
    // but is invisible to a "contains" assertion.
    let out = run(&["--pane".to_owned(), "memory".to_owned()]);
    let line = out
        .stderr
        .lines()
        .find(|l| l.contains("┌ Memory"))
        .expect("the memory pane header is rendered");
    assert!(line.starts_with(FOCUS_STYLE), "got {line:?}");
    assert!(line.ends_with(STYLE_RESET), "got {line:?}");
}

#[test]
fn compact_shows_only_selected_pane() {
    let outcome = run(&[
        "--compact".to_owned(),
        "--pane".to_owned(),
        "scheduler".to_owned(),
        "--plain".to_owned(),
    ]);
    assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
    assert!(outcome.stderr.contains("Scheduler"));
    // Other panel titles should not appear as panel headers
    assert!(!outcome.stderr.contains("┌ Plugins"));
    assert!(!outcome.stderr.contains("┌ System"));
}

#[test]
fn pane_token_aliases() {
    assert_eq!(pane_index_for_token("1"), Some(0));
    assert_eq!(pane_index_for_token("wdbx"), Some(2));
    assert_eq!(pane_index_for_token("storage"), Some(2));
    assert_eq!(pane_index_for_token("memory"), Some(4));
    assert_eq!(pane_index_for_token("nope"), None);
}

#[test]
fn pane_cycle_wraps_in_both_directions() {
    assert_eq!(cycle_pane(0, false), 1);
    assert_eq!(cycle_pane(PANES.len() - 1, false), 0);
    assert_eq!(cycle_pane(1, true), 0);
    assert_eq!(cycle_pane(0, true), PANES.len() - 1);
}

#[test]
fn mouse_hit_testing_tracks_rendered_pane_rows_and_bounds() {
    let state = collect_state(&Options::default());
    for (row, expected) in [(5, 0), (10, 1), (20, 2), (27, 3), (34, 4)] {
        assert_eq!(
            pane_at_position(&state, 2, row),
            Some(expected),
            "row={row}"
        );
    }
    assert_eq!(pane_at_position(&state, 2, 39), Some(4));
    assert_eq!(pane_at_position(&state, 2, 40), None);
    assert_eq!(pane_at_position(&state, 0, 5), None);
    let outside = u16::try_from(DIAG_WIDTH + 3).expect("dashboard width fits u16");
    assert_eq!(pane_at_position(&state, outside, 5), None);
}

#[test]
fn compact_mouse_hit_testing_exposes_only_the_visible_pane() {
    let options = Options {
        compact: true,
        initial_pane: 3,
        ..Options::default()
    };
    let state = collect_state(&options);
    assert_eq!(pane_at_position(&state, 2, 5), Some(3));
    assert_eq!(pane_at_position(&state, 2, 11), Some(3));
    assert_eq!(pane_at_position(&state, 2, 12), None);
}

#[test]
fn only_primary_mouse_presses_change_focus() {
    use crate::terminal::{Key, MouseEvent};

    let mut options = Options::default();
    let click = MouseEvent {
        button: 0,
        column: 2,
        row: 10,
        pressed: true,
    };
    assert_eq!(
        apply_interactive_input(Key::Other, Some(click), &mut options),
        InputAction::Repaint
    );
    assert_eq!(options.initial_pane, 1);

    for ignored in [
        MouseEvent {
            pressed: false,
            ..click
        },
        MouseEvent { button: 2, ..click },
        MouseEvent { row: 40, ..click },
    ] {
        assert_eq!(
            apply_interactive_input(Key::Other, Some(ignored), &mut options),
            InputAction::Ignore
        );
        assert_eq!(options.initial_pane, 1);
    }
}

#[test]
fn bad_flag_is_usage() {
    let outcome = run(&["--bogus".to_owned()]);
    assert_eq!(outcome.exit_code, 2);
}
