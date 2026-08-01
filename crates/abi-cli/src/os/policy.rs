//! The `abi agent os` policy file (`~/.abi/os-policy.toml`).
//!
//! # The one invariant: a policy file can only narrow, never widen
//!
//! `allow` is intersected with the compiled-in [`CEILING`]. A name outside the
//! ceiling is **ignored**, not honored. This is deliberate and load-bearing:
//! the file lives in the user's home directory, so anything able to write there
//! would otherwise gain arbitrary command execution through `abi` — a
//! privilege-escalation path that a "configurable allowlist" would quietly
//! introduce. Narrowing has no such hazard, so that is the whole power the file
//! gets. Widening the ceiling remains a code change, gated by review.
//!
//! # A strict TOML subset, not TOML
//!
//! The parser accepts exactly what the documented schema needs — `#` comments,
//! blank lines, `allow = ["a", "b"]`, and `timeout_secs = <integer>` — and
//! **rejects everything else**, including unknown keys. A policy file is a
//! security control, so a typo'd key must be a loud error rather than a setting
//! that silently does nothing. That argues for a strict parser over a
//! permissive full-TOML one, and it avoids a dependency for two keys.

use std::path::PathBuf;
use std::time::Duration;

/// Commands `execute` may ever run. A policy file can subset this and nothing
/// more.
pub(crate) const CEILING: &[&str] = &["true", "pwd", "ls", "whoami", "date"];

/// Timeout applied when no policy file sets one.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Bounds on `timeout_secs`. Zero would make every execute fail instantly and
/// an unbounded value would defeat the timeout, so both ends are errors rather
/// than silently clamped values — same reasoning as rejecting unknown keys.
const MIN_TIMEOUT_SECS: u64 = 1;
const MAX_TIMEOUT_SECS: u64 = 3600;

/// The effective os-control policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Policy {
    /// Always a subset of [`CEILING`].
    pub(crate) allow: Vec<String>,
    pub(crate) timeout: Duration,
    /// Where this came from, for disclosure in `dry-run` output.
    pub(crate) source: Source,
    /// Names the file listed that the ceiling does not contain. Reported so a
    /// user who expected `allow = ["curl"]` to grant `curl` learns that it did
    /// not, instead of finding out when execute refuses.
    pub(crate) ignored: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Source {
    /// No policy file; compiled defaults.
    Builtin,
    /// Loaded from this path.
    File(PathBuf),
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            allow: CEILING.iter().map(|s| (*s).to_string()).collect(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            source: Source::Builtin,
            ignored: Vec::new(),
        }
    }
}

impl Policy {
    pub(crate) fn permits(&self, cmd: &str) -> bool {
        self.allow.iter().any(|a| a == cmd)
    }

    /// The allowlist as it should read in user-facing output.
    pub(crate) fn allow_list(&self) -> String {
        if self.allow.is_empty() {
            "(none)".to_string()
        } else {
            self.allow.join(", ")
        }
    }
}

/// Resolve the policy path: `ABI_OS_POLICY`, else `~/.abi/os-policy.toml`.
///
/// `None` when neither is available, which is treated as "no policy file" —
/// the compiled default — rather than an error.
pub(crate) fn policy_path() -> Option<PathBuf> {
    if let Some(explicit) = abi_foundation::env::get(abi_foundation::env::OS_POLICY_PATH) {
        return Some(PathBuf::from(explicit));
    }
    let home = abi_foundation::env::get("HOME")?;
    Some(PathBuf::from(home).join(".abi").join("os-policy.toml"))
}

/// Load the effective policy.
///
/// # Errors
///
/// A file that exists but does not parse is an error, never a silent fallback
/// to the permissive default: a user who wrote a restricting policy must not
/// end up with a *wider* allowlist because of a typo. Callers fail closed.
pub(crate) fn load() -> Result<Policy, String> {
    let Some(path) = policy_path() else {
        return Ok(Policy::default());
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        // A missing file is the ordinary case, not a misconfiguration.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Policy::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut policy = parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    policy.source = Source::File(path);
    Ok(policy)
}

/// Parse the documented subset. See the module docs for why it is strict.
fn parse(text: &str) -> Result<Policy, String> {
    let mut policy = Policy::default();
    let mut saw_allow = false;
    let mut saw_timeout = false;

    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let n = lineno + 1;
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {n}: expected `key = value`, got `{line}`"))?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "allow" => {
                if saw_allow {
                    return Err(format!("line {n}: duplicate key `allow`"));
                }
                let names = parse_string_array(value).map_err(|e| format!("line {n}: {e}"))?;
                let (kept, ignored) = intersect_with_ceiling(&names);
                policy.allow = kept;
                policy.ignored = ignored;
                saw_allow = true;
            }
            "timeout_secs" => {
                if saw_timeout {
                    return Err(format!("line {n}: duplicate key `timeout_secs`"));
                }
                let secs: u64 = value.parse().map_err(|_| {
                    format!("line {n}: timeout_secs must be an integer, got `{value}`")
                })?;
                if !(MIN_TIMEOUT_SECS..=MAX_TIMEOUT_SECS).contains(&secs) {
                    return Err(format!(
                        "line {n}: timeout_secs must be {MIN_TIMEOUT_SECS}..={MAX_TIMEOUT_SECS}, got {secs}"
                    ));
                }
                policy.timeout = Duration::from_secs(secs);
                saw_timeout = true;
            }
            other => {
                return Err(format!(
                    "line {n}: unknown key `{other}` (supported: allow, timeout_secs)"
                ));
            }
        }
    }

    Ok(policy)
}

/// Drop a trailing `#` comment. Quoted `#` is preserved, so
/// `allow = ["a#b"]` is not truncated mid-array.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_string_array(value: &str) -> Result<Vec<String>, String> {
    let inner = value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| format!("allow must be an array of strings, got `{value}`"))?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            let item = item.trim();
            item.strip_prefix('"')
                .and_then(|i| i.strip_suffix('"'))
                .map(str::to_owned)
                .ok_or_else(|| format!("allow entries must be quoted strings, got `{item}`"))
        })
        .collect()
}

/// Split requested names into (permitted, ignored), preserving ceiling order
/// so the rendered allowlist is stable regardless of how the file ordered it.
fn intersect_with_ceiling(requested: &[String]) -> (Vec<String>, Vec<String>) {
    let kept: Vec<String> = CEILING
        .iter()
        .filter(|c| requested.iter().any(|r| r == *c))
        .map(|c| (*c).to_string())
        .collect();
    let ignored: Vec<String> = requested
        .iter()
        .filter(|r| !CEILING.contains(&r.as_str()))
        .cloned()
        .collect();
    (kept, ignored)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OverrideCleanup(Option<std::path::PathBuf>);

    impl Drop for OverrideCleanup {
        fn drop(&mut self) {
            abi_foundation::env::clear_override(abi_foundation::env::OS_POLICY_PATH);
            if let Some(dir) = &self.0 {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    #[test]
    fn the_default_policy_is_the_whole_ceiling() {
        let p = Policy::default();
        assert_eq!(p.allow.len(), CEILING.len());
        assert!(p.permits("ls"));
        assert!(!p.permits("rm"));
        assert_eq!(p.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
        assert_eq!(p.source, Source::Builtin);
    }

    #[test]
    fn a_policy_file_can_narrow_the_allowlist() {
        let p = parse("allow = [\"pwd\", \"ls\"]\n").expect("parses");
        assert_eq!(p.allow, vec!["pwd", "ls"]);
        assert!(p.permits("pwd"));
        assert!(!p.permits("whoami"), "narrowing must actually remove");
        assert!(p.ignored.is_empty());
    }

    #[test]
    fn a_policy_file_cannot_widen_the_allowlist() {
        // The security property: naming a command outside the compiled ceiling
        // must not grant it, no matter what the file says.
        let p = parse("allow = [\"ls\", \"rm\", \"curl\", \"sh\"]\n").expect("parses");
        assert_eq!(p.allow, vec!["ls"], "only the ceiling member survives");
        assert!(!p.permits("rm"));
        assert!(!p.permits("curl"));
        assert!(!p.permits("sh"));
        assert_eq!(
            p.ignored,
            vec!["rm", "curl", "sh"],
            "widening attempts are reported, not silently dropped"
        );
    }

    #[test]
    fn an_empty_allow_list_denies_everything_and_is_not_a_reset() {
        let p = parse("allow = []\n").expect("parses");
        assert!(p.allow.is_empty());
        for cmd in CEILING {
            assert!(!p.permits(cmd), "{cmd} must be denied");
        }
        assert_eq!(p.allow_list(), "(none)");
    }

    #[test]
    fn timeout_secs_is_read_and_bounded() {
        assert_eq!(
            parse("timeout_secs = 5\n").expect("parses").timeout,
            Duration::from_secs(5)
        );
        assert!(parse("timeout_secs = 0\n").is_err(), "zero is rejected");
        assert!(
            parse("timeout_secs = 99999\n").is_err(),
            "unbounded rejected"
        );
        assert!(
            parse("timeout_secs = soon\n").is_err(),
            "non-integer rejected"
        );
    }

    #[test]
    fn an_unknown_key_is_an_error_rather_than_a_silent_no_op() {
        // The whole reason the parser is strict: `allowed = [...]` must not
        // look like it restricted anything when it did not.
        let err = parse("allowed = [\"ls\"]\n").expect_err("unknown key rejected");
        assert!(err.contains("unknown key `allowed`"), "{err}");
        assert!(err.contains("supported: allow, timeout_secs"), "{err}");
    }

    #[test]
    fn duplicate_keys_are_errors_instead_of_last_value_wins() {
        let allow =
            parse("allow = [\"pwd\"]\nallow = [\"ls\"]\n").expect_err("duplicate allow rejected");
        assert!(allow.contains("duplicate key `allow`"), "{allow}");

        let timeout =
            parse("timeout_secs = 1\ntimeout_secs = 2\n").expect_err("duplicate timeout rejected");
        assert!(
            timeout.contains("duplicate key `timeout_secs`"),
            "{timeout}"
        );
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let p =
            parse("# policy\n\nallow = [\"ls\"]  # only ls\n\ntimeout_secs = 7\n").expect("parses");
        assert_eq!(p.allow, vec!["ls"]);
        assert_eq!(p.timeout, Duration::from_secs(7));
    }

    #[test]
    fn a_hash_inside_quotes_is_not_a_comment() {
        let p = parse("allow = [\"ls\", \"a#b\"]\n").expect("parses");
        assert_eq!(p.allow, vec!["ls"]);
        assert_eq!(p.ignored, vec!["a#b"], "the quoted name survived parsing");
    }

    #[test]
    fn malformed_syntax_is_an_error() {
        assert!(parse("allow\n").is_err(), "missing `=`");
        assert!(parse("allow = ls\n").is_err(), "unquoted, unbracketed");
        assert!(parse("allow = [ls]\n").is_err(), "unquoted entry");
    }

    #[test]
    fn load_reads_a_file_through_the_env_override() {
        let _guard = abi_foundation::env::lock_for_test();
        let dir = std::env::temp_dir().join(format!("abi-os-policy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let _cleanup = OverrideCleanup(Some(dir.clone()));
        let path = dir.join("os-policy.toml");
        std::fs::write(&path, "allow = [\"pwd\"]\ntimeout_secs = 3\n").expect("write");

        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            &path.display().to_string(),
        );
        let p = load().expect("loads");

        assert_eq!(p.allow, vec!["pwd"]);
        assert_eq!(p.timeout, Duration::from_secs(3));
        assert_eq!(p.source, Source::File(path.clone()));
    }

    #[test]
    fn a_missing_file_is_the_default_not_an_error() {
        let _guard = abi_foundation::env::lock_for_test();
        let _cleanup = OverrideCleanup(None);
        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            "/nonexistent/abi/os-policy.toml",
        );
        let p = load().expect("missing file falls back to the default");
        assert_eq!(p.source, Source::Builtin);
        assert_eq!(p.allow.len(), CEILING.len());
    }

    #[test]
    fn a_malformed_file_is_an_error_not_a_permissive_fallback() {
        // Failing open here would hand a user who typo'd a restricting policy
        // the full allowlist they were trying to shrink.
        let _guard = abi_foundation::env::lock_for_test();
        let dir = std::env::temp_dir().join(format!("abi-os-policy-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let _cleanup = OverrideCleanup(Some(dir.clone()));
        let path = dir.join("os-policy.toml");
        std::fs::write(&path, "allow = [\"pwd\"\n").expect("write");

        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            &path.display().to_string(),
        );
        let err = load().expect_err("malformed policy is an error");
        assert!(
            err.contains("os-policy.toml"),
            "error names the file: {err}"
        );
    }
}
