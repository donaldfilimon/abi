//! Filesystem helpers and I/O accounting.
//!
//! Ported from `src/foundation/io/`. That directory had four modules; three of
//! them — `reader.zig`, `writer.zig`, `filestream.zig` — were buffering wrappers
//! over `std.Io`, which `std::io::BufReader`/`BufWriter`/`File` already provide.
//! They are dropped rather than reimplemented.
//!
//! What survives is the part that carried behaviour of its own: the [`IoStats`]
//! counters (read back by the TUI diagnostics pane and `wdbx_stats`) and the
//! path/file convenience functions.
//!
//! The Zig `async*` prefixes were misleading — those functions were synchronous,
//! reaching for `std.Options.debug_io` to obtain a blocking `Io` handle. The
//! Rust names drop the prefix.

use crate::time;
use std::io::{Read as _, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Default buffer size for streaming reads and writes.
pub const DEFAULT_BUFFER_SIZE: usize = 4096;

/// An immutable view of [`IoStats`] at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStatsSnapshot {
    /// Total bytes read.
    pub bytes_read: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Number of read operations.
    pub read_ops: u64,
    /// Number of write operations.
    pub write_ops: u64,
    /// Number of failed operations.
    pub errors: u64,
    /// Unix milliseconds of the most recent recorded activity.
    pub last_activity_ms: i64,
}

#[derive(Debug, Default)]
struct Counters {
    bytes_read: u64,
    bytes_written: u64,
    read_ops: u64,
    write_ops: u64,
    errors: u64,
    last_activity_ms: i64,
}

/// Thread-safe I/O counters.
///
/// The Zig version guarded these with a hand-rolled spin lock; a `Mutex` is the
/// right primitive for the actual contention profile (rare, short critical
/// sections that already touch the clock).
#[derive(Debug, Default)]
pub struct IoStats {
    inner: Mutex<Counters>,
}

impl IoStats {
    /// Create zeroed counters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn with<R>(&self, f: impl FnOnce(&mut Counters) -> R) -> R {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut guard)
    }

    /// Record a completed read of `bytes` bytes.
    pub fn record_read(&self, bytes: u64) {
        let now = time::unix_ms();
        self.with(|c| {
            c.bytes_read = c.bytes_read.saturating_add(bytes);
            c.read_ops += 1;
            c.last_activity_ms = now;
        });
    }

    /// Record a completed write of `bytes` bytes.
    pub fn record_write(&self, bytes: u64) {
        let now = time::unix_ms();
        self.with(|c| {
            c.bytes_written = c.bytes_written.saturating_add(bytes);
            c.write_ops += 1;
            c.last_activity_ms = now;
        });
    }

    /// Record a failed operation.
    pub fn record_error(&self) {
        let now = time::unix_ms();
        self.with(|c| {
            c.errors += 1;
            c.last_activity_ms = now;
        });
    }

    /// Take a consistent snapshot of all counters.
    #[must_use]
    pub fn snapshot(&self) -> IoStatsSnapshot {
        self.with(|c| IoStatsSnapshot {
            bytes_read: c.bytes_read,
            bytes_written: c.bytes_written,
            read_ops: c.read_ops,
            write_ops: c.write_ops,
            errors: c.errors,
            last_activity_ms: c.last_activity_ms,
        })
    }

    /// Zero every counter.
    pub fn reset(&self) {
        self.with(|c| *c = Counters::default());
    }
}

/// A writer that counts the bytes passing through it.
///
/// Used where a payload must be sized without being buffered twice.
#[derive(Debug)]
pub struct CountingWriter<W> {
    inner: W,
    count: u64,
}

impl<W: Write> CountingWriter<W> {
    /// Wrap `inner`, starting the count at zero.
    pub const fn new(inner: W) -> Self {
        Self { inner, count: 0 }
    }

    /// Bytes written so far.
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Unwrap, discarding the count.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.count += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Read a whole file into a `String`.
pub fn read_file_to_string(path: impl AsRef<Path>) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}

/// Read a whole file into a byte vector.
pub fn read_file(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Write `data` to `path`, truncating any existing contents.
///
/// Parent directories are created if missing, matching the Zig callers that
/// paired `ensureDir` with every write.
pub fn write_file(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, data)
}

/// Append `data` to `path`, creating the file if it does not exist.
pub fn append_file(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(data.as_ref())
}

/// Write `data` to `path` atomically.
///
/// Writes to a sibling temporary file and renames over the target, so a reader
/// never observes a partially written file. This is what the WDBX manifest and
/// credential store need; the Zig side open-coded it at each call site.
pub fn write_file_atomic(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp{}", std::process::id()));
    let tmp = PathBuf::from(tmp);

    // Flush to the OS before the rename; otherwise a crash between write and
    // rename can leave the target intact but the temp file empty, and some
    // filesystems will happily rename the empty version into place.
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(data.as_ref())?;
        file.sync_all()?;
    }

    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            Err(err)
        }
    }
}

/// Create `path` and every missing parent directory.
pub fn ensure_dir(path: impl AsRef<Path>) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

/// Whether `path` exists and is a regular file.
#[must_use]
pub fn file_exists(path: impl AsRef<Path>) -> bool {
    path.as_ref().is_file()
}

/// Whether `path` exists and is a directory.
#[must_use]
pub fn dir_exists(path: impl AsRef<Path>) -> bool {
    path.as_ref().is_dir()
}

/// Resolve `path` to an absolute, canonical path.
///
/// Unlike Zig's `resolvePath`, which returned relative paths unchanged when the
/// target did not exist, this requires the path to exist — canonicalization is
/// the only way to resolve `..` and symlinks correctly.
pub fn resolve_path(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    std::fs::canonicalize(path)
}

/// Read `path` line by line, invoking `f` for each line without buffering the
/// whole file.
///
/// Used for WDBX JSONL segments, which can exceed available memory.
pub fn for_each_line(
    path: impl AsRef<Path>,
    mut f: impl FnMut(&str) -> std::io::Result<()>,
) -> std::io::Result<()> {
    use std::io::BufRead as _;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::with_capacity(DEFAULT_BUFFER_SIZE, file);
    for line in reader.lines() {
        f(&line?)?;
    }
    Ok(())
}

/// Read at most `limit` bytes from `path`.
///
/// Guards against a hostile or corrupt file exhausting memory.
pub fn read_file_limited(path: impl AsRef<Path>, limit: u64) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.take(limit).read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temp_path;

    #[test]
    fn stats_accumulate_and_snapshot() {
        let stats = IoStats::new();
        stats.record_read(100);
        stats.record_read(200);
        stats.record_write(50);
        stats.record_error();

        let snap = stats.snapshot();
        assert_eq!(snap.bytes_read, 300);
        assert_eq!(snap.bytes_written, 50);
        assert_eq!(snap.read_ops, 2);
        assert_eq!(snap.write_ops, 1);
        assert_eq!(snap.errors, 1);
        assert!(snap.last_activity_ms > 0);
    }

    #[test]
    fn stats_reset_zeroes_everything() {
        let stats = IoStats::new();
        stats.record_read(100);
        stats.record_write(10);
        stats.reset();

        assert_eq!(stats.snapshot(), IoStatsSnapshot::default());
    }

    #[test]
    fn stats_are_shareable_across_threads() {
        let stats = std::sync::Arc::new(IoStats::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let stats = std::sync::Arc::clone(&stats);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        stats.record_read(1);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("worker thread panicked");
        }
        let snap = stats.snapshot();
        assert_eq!(snap.read_ops, 800);
        assert_eq!(snap.bytes_read, 800);
    }

    #[test]
    fn write_then_read_roundtrips() {
        let path = temp_path::temp_file_path("abi_io_roundtrip", "txt");
        write_file(&path, "hello io world").expect("write");
        assert_eq!(read_file_to_string(&path).expect("read"), "hello io world");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_extends_and_creates() {
        let path = temp_path::temp_file_path("abi_io_append", "txt");
        append_file(&path, "created\n").expect("append creates");
        append_file(&path, "line2\n").expect("append extends");
        assert_eq!(
            read_file_to_string(&path).expect("read"),
            "created\nline2\n"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn atomic_write_replaces_existing_contents() {
        let path = temp_path::temp_file_path("abi_io_atomic", "json");
        write_file(&path, "old").expect("seed");
        write_file_atomic(&path, "new contents").expect("atomic write");
        assert_eq!(read_file_to_string(&path).expect("read"), "new contents");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn atomic_write_leaves_no_temp_file_behind() {
        let path = temp_path::temp_file_path("abi_io_atomic_clean", "json");
        write_file_atomic(&path, "payload").expect("atomic write");

        let dir = path.parent().expect("temp parent");
        let stem = path.file_name().and_then(|n| n.to_str()).expect("utf8");
        let leftovers = std::fs::read_dir(dir)
            .expect("read temp dir")
            .filter_map(std::result::Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(stem) && n != stem)
            })
            .count();
        assert_eq!(leftovers, 0, "atomic write left a temp file behind");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_file_creates_missing_parents() {
        let root = temp_path::temp_file_path("abi_io_nested", "d");
        let nested = root.join("a").join("b").join("c.txt");
        write_file(&nested, "deep").expect("write nested");
        assert!(file_exists(&nested));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn existence_checks_distinguish_files_from_dirs() {
        let path = temp_path::temp_file_path("abi_io_kind", "txt");
        write_file(&path, "x").expect("write");
        assert!(file_exists(&path));
        assert!(!dir_exists(&path));

        let dir = path.parent().expect("temp parent");
        assert!(dir_exists(dir));
        assert!(!file_exists(dir));

        std::fs::remove_file(&path).ok();
        assert!(!file_exists(&path));
    }

    #[test]
    fn ensure_dir_creates_nested_directories() {
        let root = temp_path::temp_file_path("abi_io_ensure", "d");
        let nested = root.join("a").join("b").join("c");
        ensure_dir(&nested).expect("ensure_dir");
        assert!(dir_exists(&nested));
        // Idempotent.
        ensure_dir(&nested).expect("ensure_dir again");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn for_each_line_streams_every_line() {
        let path = temp_path::temp_file_path("abi_io_lines", "jsonl");
        write_file(&path, "one\ntwo\nthree\n").expect("write");
        let mut seen = Vec::new();
        for_each_line(&path, |line| {
            seen.push(line.to_string());
            Ok(())
        })
        .expect("stream lines");
        assert_eq!(seen, ["one", "two", "three"]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_file_limited_caps_the_read() {
        let path = temp_path::temp_file_path("abi_io_limit", "bin");
        write_file(&path, "0123456789").expect("write");
        let got = read_file_limited(&path, 4).expect("limited read");
        assert_eq!(got, b"0123");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn counting_writer_tracks_byte_total() {
        let mut writer = CountingWriter::new(Vec::new());
        writer.write_all(b"hello").expect("write");
        writer.write_all(b" world").expect("write");
        assert_eq!(writer.count(), 11);
        assert_eq!(writer.into_inner(), b"hello world");
    }

    #[test]
    fn resolve_path_canonicalizes_an_existing_path() {
        let path = temp_path::temp_file_path("abi_io_resolve", "txt");
        write_file(&path, "x").expect("write");
        let resolved = resolve_path(&path).expect("canonicalize");
        assert!(resolved.is_absolute());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn resolve_path_fails_for_a_missing_path() {
        let missing = temp_path::temp_file_path("abi_io_missing", "txt");
        assert!(resolve_path(&missing).is_err());
    }
}
