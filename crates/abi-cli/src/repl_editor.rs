//! Bounded raw-terminal line editing for the ABI agent REPL.

use unicode_width::UnicodeWidthStr;

const PROMPT: &str = "abi> ";
const MAX_LINE_BYTES: usize = 4096;
const MAX_EDITOR_HISTORY: usize = 100;

const SLASH_COMPLETIONS: &[&str] = &[
    "/clear",
    "/cls",
    "/context",
    "/exit",
    "/feat",
    "/features",
    "/h",
    "/help",
    "/hist",
    "/history",
    "/model",
    "/profile",
    "/q",
    "/quit",
    "/reset",
    "/sea",
    "/stat",
    "/status",
];

const MODEL_COMPLETIONS: &[&str] = &[
    "abi-local",
    "apple-fm",
    "claude-fable-5",
    "claude-haiku-4-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "fable-5",
    "fable5",
    "fm",
    "fm-local",
];
const SEA_COMPLETIONS: &[&str] = &["off", "on", "status", "toggle"];

/// One completed editor read.
pub(super) enum InteractiveRead {
    /// A line submitted with Enter.
    Line(String),
    /// Terminal EOF.
    Eof,
    /// User interrupt.
    Interrupt,
}

/// Pure editor state with UTF-8 byte-boundary cursors.
#[derive(Debug, Default)]
pub(super) struct EditorState {
    buffer: String,
    cursor: usize,
    history: Vec<String>,
    history_cursor: Option<usize>,
    draft: String,
}

impl EditorState {
    fn insert(&mut self, ch: char) -> bool {
        if self.buffer.len() + ch.len_utf8() > MAX_LINE_BYTES {
            return false;
        }
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        true
    }

    fn previous_boundary(&self) -> Option<usize> {
        self.buffer[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(idx, _)| idx)
    }

    fn next_boundary(&self) -> Option<usize> {
        self.buffer[self.cursor..]
            .chars()
            .next()
            .map(|ch| self.cursor + ch.len_utf8())
    }

    fn move_left(&mut self) -> bool {
        let Some(previous) = self.previous_boundary() else {
            return false;
        };
        self.cursor = previous;
        true
    }

    fn move_right(&mut self) -> bool {
        let Some(next) = self.next_boundary() else {
            return false;
        };
        self.cursor = next;
        true
    }

    fn backspace(&mut self) -> bool {
        let Some(previous) = self.previous_boundary() else {
            return false;
        };
        self.buffer.drain(previous..self.cursor);
        self.cursor = previous;
        true
    }

    fn delete(&mut self) -> bool {
        let Some(next) = self.next_boundary() else {
            return false;
        };
        self.buffer.drain(self.cursor..next);
        true
    }

    fn home(&mut self) -> bool {
        let changed = self.cursor != 0;
        self.cursor = 0;
        changed
    }

    fn end(&mut self) -> bool {
        let changed = self.cursor != self.buffer.len();
        self.cursor = self.buffer.len();
        changed
    }

    fn set_buffer(&mut self, value: String) {
        self.buffer = value;
        self.cursor = self.buffer.len();
    }

    fn previous_history(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let next = match self.history_cursor {
            None => {
                self.draft.clone_from(&self.buffer);
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(idx) => idx - 1,
        };
        if self.history_cursor == Some(next) {
            return false;
        }
        self.history_cursor = Some(next);
        self.set_buffer(self.history[next].clone());
        true
    }

    fn next_history(&mut self) -> bool {
        let Some(current) = self.history_cursor else {
            return false;
        };
        if current + 1 < self.history.len() {
            let next = current + 1;
            self.history_cursor = Some(next);
            self.set_buffer(self.history[next].clone());
        } else {
            self.history_cursor = None;
            let draft = std::mem::take(&mut self.draft);
            self.set_buffer(draft);
        }
        true
    }

    fn complete(&mut self) -> bool {
        let prefix = &self.buffer[..self.cursor];
        let Some((start, needle, candidates, append_space)) = completion_context(prefix) else {
            return false;
        };
        let matches: Vec<&str> = candidates
            .iter()
            .copied()
            .filter(|candidate| candidate.starts_with(needle))
            .collect();
        let Some(first) = matches.first().copied() else {
            return false;
        };
        let replacement = if matches.len() == 1 {
            first
        } else {
            common_prefix(&matches)
        };
        if replacement.len() <= needle.len() {
            return false;
        }
        let append = usize::from(matches.len() == 1 && append_space);
        let new_len = self.buffer.len() - needle.len() + replacement.len() + append;
        if new_len > MAX_LINE_BYTES {
            return false;
        }
        self.buffer.replace_range(start..self.cursor, replacement);
        self.cursor = start + replacement.len();
        if matches.len() == 1 && append_space && self.buffer[self.cursor..].is_empty() {
            self.buffer.insert(self.cursor, ' ');
            self.cursor += 1;
        }
        true
    }

    fn take_line(&mut self) -> String {
        let line = std::mem::take(&mut self.buffer);
        self.cursor = 0;
        self.history_cursor = None;
        self.draft.clear();
        if !line.is_empty() && self.history.last() != Some(&line) {
            if self.history.len() == MAX_EDITOR_HISTORY {
                self.history.remove(0);
            }
            self.history.push(line.clone());
        }
        line
    }
}

fn completion_context(prefix: &str) -> Option<(usize, &str, &'static [&'static str], bool)> {
    if prefix.starts_with('/') && !prefix.chars().any(char::is_whitespace) {
        return Some((0, prefix, SLASH_COMPLETIONS, true));
    }
    let split = prefix.find(char::is_whitespace)?;
    let command = &prefix[..split];
    let rest = &prefix[split..];
    let trimmed = rest.trim_start();
    if trimmed.chars().any(char::is_whitespace) {
        return None;
    }
    let start = prefix.len() - trimmed.len();
    match command {
        "/model" => Some((start, trimmed, MODEL_COMPLETIONS, false)),
        "/sea" => Some((start, trimmed, SEA_COMPLETIONS, false)),
        _ => None,
    }
}

fn common_prefix<'a>(values: &[&'a str]) -> &'a str {
    let Some(first) = values.first().copied() else {
        return "";
    };
    let mut end = first.len();
    for value in &values[1..] {
        end = first
            .bytes()
            .zip(value.bytes())
            .take_while(|(left, right)| left == right)
            .count()
            .min(end);
    }
    &first[..end]
}

fn trailing_display_width(editor: &EditorState) -> usize {
    UnicodeWidthStr::width(&editor.buffer[editor.cursor..])
}

fn redraw(stdout: &mut std::io::Stdout, editor: &EditorState) {
    use std::io::Write;
    let _ = write!(stdout, "\r\x1b[2K{PROMPT}{}", editor.buffer);
    let trailing = trailing_display_width(editor);
    if trailing > 0 {
        let _ = write!(stdout, "\x1b[{trailing}D");
    }
    let _ = stdout.flush();
}

/// Read one edited line from the raw terminal.
pub(super) fn read_interactive_line(
    stdin: &mut std::io::Stdin,
    stdout: &mut std::io::Stdout,
    editor: &mut EditorState,
    decoder: &mut crate::terminal::KeyDecoder,
) -> InteractiveRead {
    use std::io::{Read, Write};

    redraw(stdout, editor);
    let mut bytes = [0_u8; 64];
    loop {
        match stdin.read(&mut bytes) {
            Ok(n) if n > 0 => decoder.push(&bytes[..n]),
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {
                let _ = write!(stdout, "\r\n");
                return InteractiveRead::Interrupt;
            }
            Err(_) => {
                let _ = write!(stdout, "\r\n");
                return InteractiveRead::Eof;
            }
        }

        while let Some(key) = decoder.next_key() {
            use crate::terminal::Key;
            let changed = match key {
                Key::Char(ch) if !ch.is_control() => editor.insert(ch),
                Key::Left => editor.move_left(),
                Key::Right => editor.move_right(),
                Key::Up => editor.previous_history(),
                Key::Down => editor.next_history(),
                Key::Backspace => editor.backspace(),
                Key::Home => editor.home(),
                Key::End => editor.end(),
                Key::Tab | Key::BackTab => editor.complete(),
                Key::Enter => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    return InteractiveRead::Line(editor.take_line());
                }
                Key::Interrupt => {
                    editor.set_buffer(String::new());
                    let _ = write!(stdout, "^C\r\n");
                    let _ = stdout.flush();
                    return InteractiveRead::Interrupt;
                }
                Key::Eof if editor.buffer.is_empty() => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    return InteractiveRead::Eof;
                }
                Key::Delete | Key::Eof => editor.delete(),
                Key::Escape | Key::Other | Key::Char(_) => false,
            };
            if changed {
                redraw(stdout, editor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutations_are_utf8_safe_and_display_width_is_terminal_aware() {
        let mut editor = EditorState::default();
        for ch in "a界e\u{301}b".chars() {
            assert!(editor.insert(ch));
        }
        editor.home();
        assert_eq!(trailing_display_width(&editor), 5);
        editor.end();
        assert!(editor.move_left());
        assert!(editor.backspace());
        assert_eq!(editor.buffer, "a界eb");
        assert_eq!(trailing_display_width(&editor), 1);
        assert!(editor.backspace());
        assert_eq!(editor.buffer, "a界b");
        assert_eq!(trailing_display_width(&editor), 1);
    }

    #[test]
    fn history_preserves_and_restores_the_draft() {
        let mut editor = EditorState::default();
        editor.set_buffer("first".into());
        assert_eq!(editor.take_line(), "first");
        editor.set_buffer("second".into());
        assert_eq!(editor.take_line(), "second");
        editor.set_buffer("draft".into());
        assert!(editor.previous_history());
        assert_eq!(editor.buffer, "second");
        assert!(editor.previous_history());
        assert_eq!(editor.buffer, "first");
        assert!(editor.next_history());
        assert_eq!(editor.buffer, "second");
        assert!(editor.next_history());
        assert_eq!(editor.buffer, "draft");
    }

    #[test]
    fn tab_completes_commands_models_and_sea_routes() {
        let mut editor = EditorState::default();
        editor.set_buffer("/mod".into());
        assert!(editor.complete());
        assert_eq!(editor.buffer, "/model ");
        editor.set_buffer("/model app".into());
        assert!(editor.complete());
        assert_eq!(editor.buffer, "/model apple-fm");
        editor.set_buffer("/sea tog".into());
        assert!(editor.complete());
        assert_eq!(editor.buffer, "/sea toggle");
    }

    #[test]
    fn line_and_history_growth_are_bounded() {
        let mut editor = EditorState::default();
        for _ in 0..MAX_LINE_BYTES {
            assert!(editor.insert('a'));
        }
        assert!(!editor.insert('b'));
        for idx in 0..=MAX_EDITOR_HISTORY {
            editor.set_buffer(format!("line-{idx}"));
            let _ = editor.take_line();
        }
        assert_eq!(editor.history.len(), MAX_EDITOR_HISTORY);
        assert_eq!(editor.history.first().map(String::as_str), Some("line-1"));
    }
}
