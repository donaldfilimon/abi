//! Platform temporary-directory and temp-file-path resolution.
//!
//! Ported from `src/foundation/temp_path.zig`, which resolved `$TMPDIR` (Unix),
//! then `$TEMP`, then `$TMP`, then a platform fallback. That precedence is kept
//! rather than delegating to `std::env::temp_dir`, because the framework honours
//! the ABI env overlay in [`crate::env`] and callers depend on the documented
//! order.
//!
//! The Zig `tempFilePath` used only the PID, so two calls in one process
//! collided. A monotonic counter is appended here to make paths unique per call.

use crate::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// The platform temporary directory.
///
/// Precedence: `$TMPDIR` (non-Windows only), `$TEMP`, `$TMP`, then
/// `\Windows\Temp` on Windows or `/tmp` elsewhere.
#[must_use]
pub fn temp_dir() -> PathBuf {
    if !cfg!(windows)
        && let Some(value) = env::get("TMPDIR")
    {
        return PathBuf::from(value);
    }
    if let Some(value) = env::get("TEMP") {
        return PathBuf::from(value);
    }
    if let Some(value) = env::get("TMP") {
        return PathBuf::from(value);
    }
    if cfg!(windows) {
        PathBuf::from(r"\Windows\Temp")
    } else {
        PathBuf::from("/tmp")
    }
}

/// A unique path inside [`temp_dir`], of the form
/// `<temp_dir>/<prefix>_<pid>_<n>.<extension>`.
///
/// The path is *not* created — callers open or create it themselves.
#[must_use]
pub fn temp_file_path(prefix: &str, extension: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    temp_dir().join(format!("{prefix}_{pid}_{n}.{extension}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_dir_is_non_empty_and_exists() {
        // TMPDIR is shared with `tmpdir_override_is_honoured_on_unix`.
        let _guard = env::lock_for_test();
        let dir = temp_dir();
        assert!(!dir.as_os_str().is_empty());
        assert!(dir.is_dir(), "{} should be a directory", dir.display());
    }

    #[test]
    fn temp_file_path_lands_in_temp_dir_with_the_extension() {
        let _guard = env::lock_for_test();
        let path = temp_file_path("abi_test_prefix", "txt");
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("txt"));
        assert_eq!(path.parent(), Some(temp_dir().as_path()));
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf8 name");
        assert!(name.starts_with("abi_test_prefix_"), "got {name}");
    }

    #[test]
    fn successive_calls_do_not_collide() {
        let first = temp_file_path("abi_unique", "tmp");
        let second = temp_file_path("abi_unique", "tmp");
        assert_ne!(first, second, "temp paths must be unique within a process");
    }

    #[test]
    fn tmpdir_override_is_honoured_on_unix() {
        if cfg!(windows) {
            return;
        }
        let _guard = env::lock_for_test();
        env::set_override("TMPDIR", "/custom/tmp/root");
        assert_eq!(temp_dir(), PathBuf::from("/custom/tmp/root"));
        env::clear_override("TMPDIR");
    }
}
