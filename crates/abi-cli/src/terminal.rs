//! Minimal Unix terminal helpers (echo-off secret entry + raw-mode).
//!
//! Uses `libc` termios. The workspace denies `unsafe_code` by default; this
//! module is the deliberate, reviewed exception for TTY control only.

#![allow(unsafe_code)]

use std::io::{self, BufRead, Write};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

const MAX_PENDING_BYTES: usize = 256;
const MAX_MOUSE_SEQUENCE_BYTES: usize = 48;

/// `XTerm` SGR mouse event (`CSI < button ; column ; row M/m`). Coordinates are
/// one-based terminal cells and bounded to `u16` by the decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MouseEvent {
    pub(crate) button: u16,
    pub(crate) column: u16,
    pub(crate) row: u16,
    pub(crate) pressed: bool,
}

impl MouseEvent {
    /// A primary-button press without motion or wheel bits. Modifier bits are
    /// accepted so Shift/Ctrl-click selects the same pane as an ordinary click.
    pub(crate) const fn is_primary_press(self) -> bool {
        self.pressed && self.button.trailing_zeros() >= 2 && self.button & 0b0110_0000 == 0
    }
}

/// Logical terminal keys decoded from stdin's byte stream.
///
/// Keeping this small decoder in the existing termios layer lets both the
/// dashboard and the agent REPL handle fragmented escape/UTF-8 sequences
/// without adding a terminal dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Key {
    Char(char),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Tab,
    BackTab,
    Interrupt,
    Eof,
    Escape,
    Other,
}

const ESCAPE_DELAY: Duration = Duration::from_millis(30);

/// Incremental decoder for raw terminal input.
///
/// Arrow/function sequences and multi-byte UTF-8 characters can be split
/// across `Read::read` calls. A lone Escape is held briefly so it is not
/// mistaken for the prefix of a sequence that is still arriving.
#[derive(Debug, Default)]
pub(crate) struct KeyDecoder {
    pending: Vec<u8>,
    escape_started: Option<Instant>,
    mouse_event: Option<MouseEvent>,
}

impl KeyDecoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.pending.len().saturating_add(bytes.len()) > MAX_PENDING_BYTES {
            self.pending.clear();
            self.escape_started = None;
            self.mouse_event = None;
            let keep_from = bytes.len().saturating_sub(MAX_PENDING_BYTES);
            self.pending.extend_from_slice(&bytes[keep_from..]);
            return;
        }
        self.pending.extend_from_slice(bytes);
    }

    pub(crate) fn take_mouse_event(&mut self) -> Option<MouseEvent> {
        self.mouse_event.take()
    }

    pub(crate) fn next_key(&mut self) -> Option<Key> {
        self.next_key_at(Instant::now())
    }

    fn next_key_at(&mut self, now: Instant) -> Option<Key> {
        let first = *self.pending.first()?;
        if first == 0x1b {
            if self.pending.len() == 1 {
                let started = self.escape_started.get_or_insert(now);
                if now.duration_since(*started) < ESCAPE_DELAY {
                    return None;
                }
                self.pending.remove(0);
                self.escape_started = None;
                return Some(Key::Escape);
            }
            self.escape_started = None;
            return self.decode_escape();
        }
        self.escape_started = None;

        let key = match first {
            b'\r' | b'\n' => Key::Enter,
            0x01 => Key::Home,
            0x03 => Key::Interrupt,
            0x04 => Key::Eof,
            0x05 => Key::End,
            b'\t' => Key::Tab,
            0x08 | 0x7f => Key::Backspace,
            0x20..=0x7e => Key::Char(char::from(first)),
            0x00..=0x1f => Key::Other,
            _ => return self.decode_utf8(),
        };
        self.pending.remove(0);
        Some(key)
    }

    fn decode_escape(&mut self) -> Option<Key> {
        if self.pending.get(1) != Some(&b'[') {
            self.pending.remove(0);
            return Some(Key::Escape);
        }
        let third = *self.pending.get(2)?;
        if third == b'<' {
            return self.decode_mouse();
        }
        let (consumed, key) = match third {
            b'A' => (3, Key::Up),
            b'B' => (3, Key::Down),
            b'C' => (3, Key::Right),
            b'D' => (3, Key::Left),
            b'H' => (3, Key::Home),
            b'F' => (3, Key::End),
            b'Z' => (3, Key::BackTab),
            b'1' | b'7' if self.pending.get(3) == Some(&b'~') => (4, Key::Home),
            b'3' if self.pending.get(3) == Some(&b'~') => (4, Key::Delete),
            b'4' | b'8' if self.pending.get(3) == Some(&b'~') => (4, Key::End),
            b'0'..=b'9' if self.pending.len() < 4 => return None,
            _ => (3, Key::Other),
        };
        self.pending.drain(..consumed);
        Some(key)
    }

    fn decode_mouse(&mut self) -> Option<Key> {
        let search_end = self.pending.len().min(MAX_MOUSE_SEQUENCE_BYTES);
        let terminator = self.pending[3..search_end]
            .iter()
            .position(|byte| matches!(byte, b'M' | b'm'))
            .map(|offset| offset + 3);
        let Some(end) = terminator else {
            if self.pending.len() < MAX_MOUSE_SEQUENCE_BYTES {
                return None;
            }
            // Discard the CSI introducer and let subsequent bytes recover as
            // ordinary input. Malformed reports cannot grow `pending` without
            // bound or wedge every later key behind them.
            self.pending.drain(..3);
            return Some(Key::Other);
        };

        let pressed = self.pending[end] == b'M';
        let event = std::str::from_utf8(&self.pending[3..end])
            .ok()
            .and_then(|body| {
                let mut fields = body.split(';');
                let button = fields.next()?.parse::<u16>().ok()?;
                let column = fields.next()?.parse::<u16>().ok()?;
                let row = fields.next()?.parse::<u16>().ok()?;
                fields.next().is_none().then_some(MouseEvent {
                    button,
                    column,
                    row,
                    pressed,
                })
            });
        self.pending.drain(..=end);
        self.mouse_event = event;
        Some(Key::Other)
    }

    fn decode_utf8(&mut self) -> Option<Key> {
        let first = self.pending[0];
        let width = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => {
                self.pending.remove(0);
                return Some(Key::Other);
            }
        };
        if self.pending.len() < width {
            return None;
        }
        let decoded = std::str::from_utf8(&self.pending[..width])
            .ok()
            .and_then(|s| s.chars().next());
        self.pending.drain(..width);
        Some(decoded.map_or(Key::Other, Key::Char))
    }
}

/// Read one line from stdin with echo disabled (Unix TTY). Restores termios.
pub(crate) fn read_secret_line() -> Result<String, String> {
    let fd = io::stdin().as_raw_fd();
    // SAFETY: operate only on the process stdin fd; always restore on exit.
    let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
    if unsafe { libc::tcgetattr(fd, &raw mut original) } != 0 {
        return Err("failed to read terminal attributes".into());
    }
    let mut raw = original;
    raw.c_lflag &= !(libc::ECHO | libc::ECHOE | libc::ECHOK | libc::ECHONL);
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw const raw) } != 0 {
        return Err("failed to disable terminal echo".into());
    }

    let mut line = String::new();
    let result = io::stdin().lock().read_line(&mut line);
    let _ = unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw const original) };
    let _ = writeln!(io::stderr());

    result.map_err(|e| format!("failed to read secret: {e}"))?;
    let trimmed = line.trim().to_string();
    if trimmed.is_empty() {
        return Err("empty secret".into());
    }
    Ok(trimmed)
}

const ENABLE_MOUSE_CAPTURE: &[u8] = b"\x1b[?1000h\x1b[?1006h";
const DISABLE_MOUSE_CAPTURE: &[u8] = b"\x1b[?1006l\x1b[?1000l";

/// `XTerm` button-event + SGR-coordinate capture guard.
///
/// Capture is enabled only after raw mode succeeds. Every normal return and
/// unwinding path runs `Drop`, which disables both modes before raw termios is
/// restored. A partial enable failure also emits the disable sequence before
/// returning an error.
pub(crate) struct MouseCapture<W: Write> {
    writer: W,
}

impl<W: Write> MouseCapture<W> {
    pub(crate) fn enter(mut writer: W) -> Result<Self, String> {
        if let Err(err) = writer
            .write_all(ENABLE_MOUSE_CAPTURE)
            .and_then(|()| writer.flush())
        {
            let _ = writer.write_all(DISABLE_MOUSE_CAPTURE);
            let _ = writer.flush();
            return Err(format!("failed to enable mouse capture: {err}"));
        }
        Ok(Self { writer })
    }
}

impl<W: Write> Drop for MouseCapture<W> {
    fn drop(&mut self) {
        let _ = self.writer.write_all(DISABLE_MOUSE_CAPTURE);
        let _ = self.writer.flush();
    }
}

/// Unix raw-mode guard: non-canonical, no echo, bounded single-byte reads.
pub(crate) struct RawMode {
    fd: i32,
    original: libc::termios,
}

impl RawMode {
    /// Enter raw mode on stdin.
    pub(crate) fn enter() -> Result<Self, String> {
        let fd = io::stdin().as_raw_fd();
        // SAFETY: termios on live stdin; restored in Drop.
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        if unsafe { libc::tcgetattr(fd, &raw mut original) } != 0 {
            return Err("tcgetattr failed".into());
        }
        let mut raw = original;
        // Disable signal generation so Ctrl-C is returned as byte 0x03 and the
        // caller can unwind through this guard. Otherwise SIGINT can terminate
        // the process before Drop restores the terminal.
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::IEXTEN | libc::ISIG);
        raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
        raw.c_cflag |= libc::CS8;
        raw.c_cc[libc::VMIN] = 0;
        // A one-decisecond ceiling prevents a busy loop while keeping the
        // dashboard refresh and lone-Escape timeout responsive.
        raw.c_cc[libc::VTIME] = 1;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw const raw) } != 0 {
            return Err("tcsetattr failed".into());
        }
        Ok(Self { fd, original })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        // SAFETY: restore attributes captured in enter().
        let _ = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &raw const self.original) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("writer lock").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn decoder_preserves_fragmented_utf8_and_escape_sequences() {
        let mut decoder = KeyDecoder::default();
        decoder.push(&[0xe2]);
        assert_eq!(decoder.next_key(), None);
        decoder.push(&[0x98, 0x83, 0x1b, b'[']);
        assert_eq!(decoder.next_key(), Some(Key::Char('☃')));
        assert_eq!(decoder.next_key(), None);
        decoder.push(b"D");
        assert_eq!(decoder.next_key(), Some(Key::Left));
    }

    #[test]
    fn decoder_distinguishes_tab_backtab_and_delete() {
        let mut decoder = KeyDecoder::default();
        decoder.push(b"\t\x1b[Z\x1b[3~");
        assert_eq!(decoder.next_key(), Some(Key::Tab));
        assert_eq!(decoder.next_key(), Some(Key::BackTab));
        assert_eq!(decoder.next_key(), Some(Key::Delete));
    }

    #[test]
    fn decoder_handles_fragmented_sgr_mouse_press_and_release() {
        let mut decoder = KeyDecoder::default();
        decoder.push(b"\x1b[<0;12;");
        assert_eq!(decoder.next_key(), None);
        decoder.push(b"34M\x1b[<0;12;34m");
        let press = MouseEvent {
            button: 0,
            column: 12,
            row: 34,
            pressed: true,
        };
        assert_eq!(decoder.next_key(), Some(Key::Other));
        assert_eq!(decoder.take_mouse_event(), Some(press));
        assert!(press.is_primary_press());
        assert_eq!(decoder.next_key(), Some(Key::Other));
        assert_eq!(
            decoder.take_mouse_event(),
            Some(MouseEvent {
                pressed: false,
                ..press
            })
        );
    }

    #[test]
    fn malformed_mouse_report_is_bounded_and_decoder_recovers() {
        let mut decoder = KeyDecoder::default();
        let mut malformed = b"\x1b[<".to_vec();
        malformed.resize(MAX_MOUSE_SEQUENCE_BYTES, b'9');
        decoder.push(&malformed);
        assert_eq!(decoder.next_key(), Some(Key::Other));
        while decoder.next_key().is_some() {}
        decoder.push(b"q");
        assert_eq!(decoder.next_key(), Some(Key::Char('q')));
    }

    #[test]
    fn mouse_capture_guard_enables_and_disables_in_drop_order() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        {
            let _guard = MouseCapture::enter(SharedWriter(Arc::clone(&bytes))).expect("capture");
            assert_eq!(
                bytes.lock().expect("writer lock").as_slice(),
                ENABLE_MOUSE_CAPTURE
            );
        }
        let expected = [ENABLE_MOUSE_CAPTURE, DISABLE_MOUSE_CAPTURE].concat();
        assert_eq!(*bytes.lock().expect("writer lock"), expected);
    }
}
