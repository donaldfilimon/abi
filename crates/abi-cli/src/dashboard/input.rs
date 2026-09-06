//! Raw-mode interactive loop for the diagnostics dashboard.
//!
//! Owns the TTY: keypress/mouse decoding, pane cycling, screen clearing, and
//! terminal restoration on exit. The one-shot renderers stay free of terminal
//! state so they remain testable without a PTY.

use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

use crate::app::Outcome;

use super::render::{pane_at_position, render_text};
use super::state::collect_state;
use super::{Options, PANES};

pub(super) fn clear_screen(out: &mut dyn Write) {
    let _ = write!(out, "\x1b[2J\x1b[H");
    let _ = out.flush();
}

pub(super) fn cycle_pane(selected: usize, backwards: bool) -> usize {
    if backwards {
        selected.checked_sub(1).unwrap_or(PANES.len() - 1)
    } else {
        (selected + 1) % PANES.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputAction {
    Quit,
    Repaint,
    Ignore,
}

pub(super) fn apply_interactive_input(
    key: crate::terminal::Key,
    mouse: Option<crate::terminal::MouseEvent>,
    options: &mut Options,
) -> InputAction {
    use crate::terminal::Key;

    if let Some(event) = mouse
        && event.is_primary_press()
    {
        let state = collect_state(options);
        if let Some(pane) = pane_at_position(&state, event.column, event.row) {
            options.initial_pane = pane;
            return InputAction::Repaint;
        }
    }
    match key {
        Key::Char('q' | 'Q') | Key::Escape | Key::Interrupt | Key::Eof => InputAction::Quit,
        Key::Char('r' | 'R') => InputAction::Repaint,
        Key::Char('h' | 'H') | Key::Left | Key::BackTab => {
            options.initial_pane = cycle_pane(options.initial_pane, true);
            InputAction::Repaint
        }
        Key::Char('l' | 'L') | Key::Right | Key::Tab => {
            options.initial_pane = cycle_pane(options.initial_pane, false);
            InputAction::Repaint
        }
        Key::Char(key @ '1'..='5') => {
            options.initial_pane = usize::from(key as u8 - b'1');
            InputAction::Repaint
        }
        _ => InputAction::Ignore,
    }
}

/// Interactive raw-mode loop for a real TTY. Returns a final one-shot frame on exit
/// (for capture) after restoring the terminal.
pub(super) fn run_interactive(options: Options) -> Outcome {
    #[cfg(not(unix))]
    {
        let state = collect_state(&options);
        let text = render_text(&state);
        return Outcome::stderr(
            format!(
                "{text}\nnote: interactive raw-mode dashboard requires a Unix TTY; rendered one-shot.\n"
            ),
            0,
        );
    }

    #[cfg(unix)]
    {
        let _raw = match crate::terminal::RawMode::enter() {
            Ok(r) => r,
            Err(err) => {
                let state = collect_state(&options);
                let text = render_text(&state);
                return Outcome::stderr(
                    format!("{text}\nnote: raw-mode unavailable ({err}); one-shot fallback.\n"),
                    0,
                );
            }
        };
        // Declared after `_raw` so Rust drops this guard first: mouse reporting
        // is disabled before canonical terminal mode is restored.
        let _mouse = match crate::terminal::MouseCapture::enter(io::stderr()) {
            Ok(capture) => capture,
            Err(err) => {
                let state = collect_state(&options);
                let text = render_text(&state);
                return Outcome::stderr(
                    format!(
                        "{text}\nnote: mouse capture unavailable ({err}); one-shot fallback.\n"
                    ),
                    0,
                );
            }
        };

        let mut options = options;
        let interval =
            Duration::from_millis(u64::try_from(options.refresh_interval_ms).unwrap_or(1000));
        let mut last_paint = Instant::now();
        let mut force_paint = true;
        let mut stderr = io::stderr();
        let mut stdin = io::stdin();
        let mut buf = [0_u8; 16];
        let mut decoder = crate::terminal::KeyDecoder::default();

        loop {
            if force_paint || last_paint.elapsed() >= interval {
                let mut state = collect_state(&options);
                state.interactive = true;
                state.selected_pane = options.initial_pane.min(PANES.len() - 1);
                clear_screen(&mut stderr);
                let frame = render_text(&state);
                let _ = write!(stderr, "{frame}");
                let _ = stderr.flush();
                last_paint = Instant::now();
                force_paint = false;
            }

            match stdin.read(&mut buf) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(n) => {
                    decoder.push(&buf[..n]);
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => {
                    let state = collect_state(&options);
                    return Outcome::stderr(render_text(&state), 0);
                }
            }

            while let Some(key) = decoder.next_key() {
                match apply_interactive_input(key, decoder.take_mouse_event(), &mut options) {
                    InputAction::Quit => {
                        let mut state = collect_state(&options);
                        state.interactive = true;
                        let frame = render_text(&state);
                        return Outcome::stderr(frame, 0);
                    }
                    InputAction::Repaint => force_paint = true,
                    InputAction::Ignore => {}
                }
            }
        }
    }
}
