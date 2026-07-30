//! String helpers that Rust's standard library does not already cover.
//!
//! Ported from `src/foundation/utils.zig`. Most of that module reimplemented
//! things Zig's std lacked a direct form of, and they map onto std here:
//!
//! | Zig | Rust |
//! |---|---|
//! | `trimWhitespace` | `str::trim` |
//! | `splitLines` | `str::lines` |
//! | `joinStrings` | `[T]::join` |
//! | `pathJoin` / `pathBasename` / `pathDirname` / `pathExt` | `std::path::Path` |
//! | `startsWith` / `endsWith` / `contains` | `str::starts_with` / `ends_with` / `contains` |
//! | `stringEqlIgnoreCase` | `str::eq_ignore_ascii_case` |
//!
//! Only [`contains_ignore_case`] has no std equivalent. The path helpers are
//! also kept, because their edge-case answers differ from `std::path` in ways
//! the CLI and WDBX callers depend on — see each function's note.

/// Case-insensitive substring search, ASCII only.
///
/// An empty needle always matches, consistent with `str::contains("")`.
#[must_use]
pub fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    haystack
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

/// Whether `haystack` starts with `prefix`, ignoring ASCII case.
#[must_use]
pub fn starts_with_ignore_case(haystack: &str, prefix: &str) -> bool {
    haystack.len() >= prefix.len()
        && haystack.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

/// The final path component.
///
/// Differs from `Path::file_name`: a trailing separator yields `""` rather than
/// the preceding component, and there is no `Option`. `basename("/")` is `""`.
#[must_use]
pub fn basename(path: &str) -> &str {
    match path.rfind(['/', '\\']) {
        Some(index) => &path[index + 1..],
        None => path,
    }
}

/// The directory portion of a path.
///
/// Returns `"."` when there is no separator and `"/"` for a root-level path,
/// matching the Zig behaviour rather than `Path::parent`'s `Option`.
#[must_use]
pub fn dirname(path: &str) -> &str {
    match path.rfind(['/', '\\']) {
        Some(0) => "/",
        Some(index) => &path[..index],
        None => ".",
    }
}

/// The file extension *including* the leading dot, or `""` when there is none.
///
/// Differs from `Path::extension`, which omits the dot and returns an `Option`.
/// A leading-dot name like `.gitignore` is treated as an extension, matching Zig.
#[must_use]
pub fn extension(path: &str) -> &str {
    let base = basename(path);
    match base.rfind('.') {
        Some(index) => &base[index..],
        None => "",
    }
}

/// Join two path fragments with a single separator.
///
/// Collapses a duplicate separator at the boundary; an empty fragment yields the
/// other side unchanged.
#[must_use]
pub fn path_join(base: &str, relative: &str) -> String {
    if base.is_empty() {
        return relative.to_string();
    }
    if relative.is_empty() {
        return base.to_string();
    }
    if base.ends_with('/') || base.ends_with('\\') {
        format!("{base}{relative}")
    } else {
        format!("{base}/{relative}")
    }
}

/// Truncate `text` to at most `max_chars` characters, appending `…` when cut.
///
/// Operates on `char` boundaries, so it cannot split a multi-byte codepoint —
/// the Zig version sliced bytes and could emit invalid UTF-8 into the TUI.
#[must_use]
pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_ignore_case_matches_either_case() {
        assert!(contains_ignore_case("Hello World", "hello"));
        assert!(contains_ignore_case("HELLO WORLD", "hello"));
        assert!(contains_ignore_case("hello world", "WORLD"));
        assert!(!contains_ignore_case("hello", "world"));
    }

    #[test]
    fn contains_ignore_case_treats_empty_needle_as_a_match() {
        assert!(contains_ignore_case("abc", ""));
        assert!(contains_ignore_case("", ""));
        assert!(!contains_ignore_case("", "a"));
    }

    #[test]
    fn starts_with_ignore_case_compares_the_prefix_only() {
        assert!(starts_with_ignore_case("Bearer abc", "bearer "));
        assert!(!starts_with_ignore_case("Basic abc", "bearer "));
        assert!(starts_with_ignore_case("abc", ""));
        assert!(!starts_with_ignore_case("ab", "abc"));
    }

    #[test]
    fn basename_matches_zig_edge_cases() {
        assert_eq!(basename("/home/user/file.txt"), "file.txt");
        assert_eq!(basename("file.txt"), "file.txt");
        assert_eq!(basename("/"), "");
        assert_eq!(basename("/home/user/"), "");
        assert_eq!(basename(r"C:\dir\file.txt"), "file.txt");
    }

    #[test]
    fn dirname_matches_zig_edge_cases() {
        assert_eq!(dirname("/home/user/file.txt"), "/home/user");
        assert_eq!(dirname("file.txt"), ".");
        assert_eq!(dirname("/file.txt"), "/");
    }

    #[test]
    fn extension_keeps_the_leading_dot() {
        assert_eq!(extension("/home/user/file.txt"), ".txt");
        assert_eq!(extension("/home/user/file"), "");
        assert_eq!(extension("src/main.rs"), ".rs");
        assert_eq!(extension("archive.tar.gz"), ".gz");
        // A dot in a parent directory must not leak into the result.
        assert_eq!(extension("/etc/some.dir/file"), "");
    }

    #[test]
    fn path_join_collapses_duplicate_separators() {
        assert_eq!(
            path_join("/home/user", "docs/file.txt"),
            "/home/user/docs/file.txt"
        );
        assert_eq!(path_join("/home/user/", "docs"), "/home/user/docs");
        assert_eq!(path_join("", "docs"), "docs");
        assert_eq!(path_join("/home", ""), "/home");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello w…");
        assert_eq!(truncate("", 5), "");
        assert_eq!(truncate("abc", 0), "");
        // Would have produced invalid UTF-8 under byte slicing.
        let truncated = truncate("héllo wörld", 6);
        assert_eq!(truncated, "héllo…");
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
    }
}
