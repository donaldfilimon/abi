//! Minimal Unix terminal helpers (echo-off secret entry + raw-mode).
//!
//! Uses `libc` termios. The workspace denies `unsafe_code` by default; this
//! module is the deliberate, reviewed exception for TTY control only.

#![allow(unsafe_code)]

use std::io::{self, BufRead, Write};
use std::os::fd::AsRawFd;

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

/// Unix raw-mode guard: non-canonical, no echo, non-blocking single-byte reads.
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
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 0;
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
