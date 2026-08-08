//! In-memory structured logger.
//!
//! Ported from `src/foundation/logger.zig`. The Zig logger buffered every entry
//! *and* mirrored it to `std.log`; the buffer is what the TUI diagnostics pane
//! and several tests read back, so it is preserved here.
//!
//! One Zig behaviour is deliberately not carried over: on allocation failure it
//! silently dropped the entry. Rust has no fallible-allocation path here, so a
//! call that returns is a call that recorded.

use crate::time;
use std::fmt;

/// Severity of a log entry, ordered from least to most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// Verbose developer detail.
    Debug,
    /// Normal operational milestones.
    Info,
    /// Something unexpected that did not fail the operation.
    Warn,
    /// The operation failed.
    Error,
}

impl Level {
    /// The uppercase tag used in rendered output (`DEBUG`, `INFO`, …).
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    /// Parse a level from its name, case-insensitively.
    ///
    /// Accepts both `error` and Zig's abbreviated `err`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "debug" => Some(Self::Debug),
            "info" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warn),
            "err" | "error" => Some(Self::Error),
            _ => None,
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// A single recorded log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Severity of the entry.
    pub level: Level,
    /// The subsystem that emitted it.
    pub module: String,
    /// The formatted message.
    pub message: String,
    /// Unix milliseconds at which it was recorded.
    pub timestamp: i64,
}

impl fmt::Display for LogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] [{}] {}",
            self.level.tag(),
            self.module,
            self.message
        )
    }
}

/// A buffering logger that drops entries below its minimum level.
#[derive(Debug, Clone)]
pub struct Logger {
    level: Level,
    entries: Vec<LogEntry>,
    echo: bool,
}

impl Logger {
    /// Create a logger that keeps entries at or above `min_level`.
    #[must_use]
    pub fn new(min_level: Level) -> Self {
        Self {
            level: min_level,
            entries: Vec::new(),
            echo: false,
        }
    }

    /// Also mirror each recorded entry to stderr.
    ///
    /// Off by default. The Zig logger always echoed, which made test output
    /// noisy; making it opt-in keeps the buffered behaviour without the noise.
    #[must_use]
    pub fn with_stderr_echo(mut self, echo: bool) -> Self {
        self.echo = echo;
        self
    }

    /// The configured minimum level.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// Change the minimum level. Already-recorded entries are unaffected.
    pub fn set_level(&mut self, level: Level) {
        self.level = level;
    }

    /// Whether an entry at `level` would be recorded.
    #[must_use]
    pub fn enabled(&self, level: Level) -> bool {
        level >= self.level
    }

    /// Record a message, if `level` passes the filter.
    pub fn log(&mut self, level: Level, module: &str, message: impl Into<String>) {
        if !self.enabled(level) {
            return;
        }
        let entry = LogEntry {
            level,
            module: module.to_string(),
            message: message.into(),
            timestamp: time::unix_ms(),
        };
        if self.echo {
            eprintln!("{entry}");
        }
        self.entries.push(entry);
    }

    /// Record at [`Level::Debug`].
    pub fn debug(&mut self, module: &str, message: impl Into<String>) {
        self.log(Level::Debug, module, message);
    }

    /// Record at [`Level::Info`].
    pub fn info(&mut self, module: &str, message: impl Into<String>) {
        self.log(Level::Info, module, message);
    }

    /// Record at [`Level::Warn`].
    pub fn warn(&mut self, module: &str, message: impl Into<String>) {
        self.log(Level::Warn, module, message);
    }

    /// Record at [`Level::Error`].
    pub fn error(&mut self, module: &str, message: impl Into<String>) {
        self.log(Level::Error, module, message);
    }

    /// All recorded entries, oldest first.
    #[must_use]
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// The number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all recorded entries, keeping the configured level.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Entries at exactly `level`.
    pub fn entries_at(&self, level: Level) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter().filter(move |e| e.level == level)
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new(Level::Info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_logger_starts_empty_at_the_given_level() {
        let logger = Logger::new(Level::Debug);
        assert_eq!(logger.level(), Level::Debug);
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn records_entries_with_level_and_module() {
        let mut logger = Logger::new(Level::Debug);
        logger.info("test", "hello world");
        assert_eq!(logger.len(), 1);
        let entry = &logger.entries()[0];
        assert_eq!(entry.level, Level::Info);
        assert_eq!(entry.module, "test");
        assert_eq!(entry.message, "hello world");
        assert!(entry.timestamp > 0);
    }

    #[test]
    fn filters_entries_below_the_minimum_level() {
        let mut logger = Logger::new(Level::Warn);
        logger.debug("test", "should not appear");
        logger.info("test", "should not appear");
        assert_eq!(logger.len(), 0);

        logger.warn("test", "should appear");
        assert_eq!(logger.len(), 1);

        logger.error("test", "also appears");
        assert_eq!(logger.len(), 2);
    }

    #[test]
    fn clear_drops_entries_but_keeps_the_level() {
        let mut logger = Logger::new(Level::Debug);
        logger.info("test", "entry 1");
        logger.info("test", "entry 2");
        assert_eq!(logger.len(), 2);

        logger.clear();
        assert_eq!(logger.len(), 0);
        assert_eq!(logger.level(), Level::Debug);
    }

    #[test]
    fn set_level_changes_subsequent_filtering_only() {
        let mut logger = Logger::new(Level::Debug);
        logger.debug("test", "kept");
        logger.set_level(Level::Error);
        logger.debug("test", "dropped");
        assert_eq!(logger.len(), 1);
        assert_eq!(logger.entries()[0].message, "kept");
    }

    #[test]
    fn levels_are_ordered_by_severity() {
        assert!(Level::Debug < Level::Info);
        assert!(Level::Info < Level::Warn);
        assert!(Level::Warn < Level::Error);
    }

    #[test]
    fn level_parse_accepts_zig_abbreviation() {
        assert_eq!(Level::parse("debug"), Some(Level::Debug));
        assert_eq!(Level::parse("INFO"), Some(Level::Info));
        assert_eq!(Level::parse("warning"), Some(Level::Warn));
        assert_eq!(Level::parse("err"), Some(Level::Error));
        assert_eq!(Level::parse("error"), Some(Level::Error));
        assert_eq!(Level::parse("nope"), None);
    }

    #[test]
    fn entry_renders_like_the_zig_output() {
        let mut logger = Logger::new(Level::Debug);
        logger.warn("wdbx", "segment rewritten");
        assert_eq!(
            logger.entries()[0].to_string(),
            "[WARN] [wdbx] segment rewritten"
        );
    }

    #[test]
    fn entries_at_filters_by_exact_level() {
        let mut logger = Logger::new(Level::Debug);
        logger.info("a", "one");
        logger.error("b", "two");
        logger.info("c", "three");
        let infos: Vec<_> = logger.entries_at(Level::Info).collect();
        assert_eq!(infos.len(), 2);
        assert_eq!(logger.entries_at(Level::Error).count(), 1);
        assert_eq!(logger.entries_at(Level::Warn).count(), 0);
    }
}
