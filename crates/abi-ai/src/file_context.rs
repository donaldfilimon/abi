//! File-aware agent context: `@file` mentions, workspace tree, git diff.
//!
//! Ported from `src/features/ai/file_context.zig`. Path sandboxing rejects `..`
//! and absolute paths. Symlink escape is not fully resolved (same as the pure
//! validate path in Zig); callers should keep `root` as the process cwd.

use std::fs;
use std::path::{Component, Path};
use std::process::Command;

/// Maximum total bytes of file context injected into a prompt.
pub const DEFAULT_BUDGET_BYTES: usize = 8192;
/// Max depth for workspace tree listings.
pub const TREE_MAX_DEPTH: usize = 3;
/// Max file entries in a workspace tree listing.
pub const TREE_MAX_ENTRIES: usize = 64;
/// Default tree budget slice.
pub const TREE_DEFAULT_BUDGET_BYTES: usize = 2048;
/// Default git-diff budget slice.
pub const GIT_DIFF_DEFAULT_BUDGET_BYTES: usize = 2048;

/// A single `@file` mention found in input text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMention {
    /// Path text (borrowed from the original input).
    pub path: String,
    /// Byte offset of `@` in the input.
    pub start: usize,
    /// Byte offset after the path.
    pub end: usize,
}

/// Context budget tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    /// Hard max.
    pub max_bytes: usize,
    /// Bytes already consumed.
    pub used: usize,
}

impl ContextBudget {
    /// Fresh budget.
    #[must_use]
    pub const fn new(max_bytes: usize) -> Self {
        Self { max_bytes, used: 0 }
    }

    /// Bytes still available.
    #[must_use]
    pub fn remaining(self) -> usize {
        self.max_bytes.saturating_sub(self.used)
    }

    /// Whether `bytes` still fit.
    #[must_use]
    pub fn can_fit(self, bytes: usize) -> bool {
        self.used.saturating_add(bytes) <= self.max_bytes
    }

    /// Record consumption (saturates at max).
    pub fn consume(&mut self, bytes: usize) {
        self.used = self.used.saturating_add(bytes).min(self.max_bytes);
    }
}

/// Path validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    /// Empty path.
    EmptyPath,
    /// Absolute path rejected.
    AbsolutePathRejected,
    /// `..` component.
    PathEscape,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPath => f.write_str("empty path"),
            Self::AbsolutePathRejected => f.write_str("absolute path rejected"),
            Self::PathEscape => f.write_str("path escape"),
        }
    }
}

impl std::error::Error for PathError {}

/// Validate that `path` is a safe relative path (no `..`, not absolute).
///
/// # Errors
///
/// Returns [`PathError`] variants for empty, absolute, or escaping paths.
pub fn validate_mention_path(path: &str) -> Result<(), PathError> {
    if path.is_empty() {
        return Err(PathError::EmptyPath);
    }
    let p = Path::new(path);
    if p.is_absolute() {
        return Err(PathError::AbsolutePathRejected);
    }
    for component in p.components() {
        if matches!(component, Component::ParentDir) {
            return Err(PathError::PathEscape);
        }
    }
    Ok(())
}

/// Scan input text for `@file` mentions.
#[must_use]
pub fn parse_file_mentions(input: &str) -> Vec<FileMention> {
    let bytes = input.as_bytes();
    let mut mentions = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        if i > 0 {
            let prev = bytes[i - 1];
            if !matches!(prev, b' ' | b'\t' | b'\n' | b'\r' | b'(' | b'[') {
                i += 1;
                continue;
            }
        }
        let start = i;
        i += 1;
        let path_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if path_start == i {
            continue;
        }
        let path = &input[path_start..i];
        if !path.contains('.') && !path.contains('/') {
            continue;
        }
        mentions.push(FileMention {
            path: path.to_string(),
            start,
            end: i,
        });
    }
    mentions
}

/// Read a file relative to `cwd`, truncated to `max_read_bytes`.
///
/// # Errors
///
/// Path validation failure or I/O error.
pub fn read_file_bounded(
    cwd: &Path,
    path: &str,
    max_read_bytes: usize,
) -> Result<Vec<u8>, PathError> {
    validate_mention_path(path)?;
    let full = cwd.join(path);
    let data = fs::read(&full).map_err(|_| PathError::EmptyPath)?; // map missing to empty-path-like skip
    if data.len() <= max_read_bytes {
        Ok(data)
    } else {
        Ok(data[..max_read_bytes].to_vec())
    }
}

/// Resolve `@file` mentions by injecting file contents.
#[must_use]
pub fn resolve_and_inject(input: &str, root: &Path, budget: &mut ContextBudget) -> String {
    let mentions = parse_file_mentions(input);
    if mentions.is_empty() {
        return input.to_string();
    }
    let mut result = String::new();
    let mut last_end = 0;
    for mention in &mentions {
        result.push_str(&input[last_end..mention.start]);
        let max_read = budget.remaining();
        if max_read == 0 {
            result.push_str(&input[mention.start..mention.end]);
            last_end = mention.end;
            continue;
        }
        match read_file_bounded(root, &mention.path, max_read) {
            Ok(contents) => {
                let header = format!("[file: {}]\n", mention.path);
                result.push_str(&header);
                result.push_str(&String::from_utf8_lossy(&contents));
                result.push('\n');
                budget.consume(header.len() + contents.len() + 1);
            }
            Err(_) => {
                result.push_str(&input[mention.start..mention.end]);
            }
        }
        last_end = mention.end;
    }
    result.push_str(&input[last_end..]);
    result
}

fn should_skip_tree_name(name: &str) -> bool {
    if name.is_empty() || name.starts_with('.') {
        return true;
    }
    matches!(
        name,
        "zig-cache" | ".zig-cache" | "zig-out" | "node_modules" | "target" | ".git"
    )
}

/// Recursively list files under `root`, capped by depth and entry count.
#[must_use]
pub fn list_workspace_tree(root: &Path, max_depth: usize, max_entries: usize) -> String {
    let mut out = String::new();
    let mut count = 0_usize;
    let root_label = root.file_name().and_then(|s| s.to_str()).unwrap_or(".");
    let start_rel = if root == Path::new(".") {
        String::new()
    } else {
        root_label.to_string()
    };
    walk_tree(
        root,
        &start_rel,
        0,
        max_depth,
        max_entries,
        &mut count,
        &mut out,
    );
    if out.is_empty() {
        out.push_str("(empty tree)\n");
    }
    out
}

fn walk_tree(
    abs_dir: &Path,
    rel: &str,
    depth: usize,
    max_depth: usize,
    max_entries: usize,
    count: &mut usize,
    out: &mut String,
) {
    if depth > max_depth || *count >= max_entries {
        return;
    }
    let Ok(entries) = fs::read_dir(abs_dir) else {
        return;
    };
    let mut collected: Vec<(String, bool)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if should_skip_tree_name(&name) {
            continue;
        }
        let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
        let is_file = entry.file_type().is_ok_and(|t| t.is_file());
        if !is_dir && !is_file {
            continue;
        }
        collected.push((name.into_owned(), is_dir));
    }
    collected.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, is_dir) in collected {
        if *count >= max_entries {
            break;
        }
        let child_rel = if rel.is_empty() {
            name.clone()
        } else {
            format!("{rel}/{name}")
        };
        if validate_mention_path(&child_rel).is_err() {
            continue;
        }
        if is_dir {
            if depth >= max_depth {
                continue;
            }
            walk_tree(
                &abs_dir.join(&name),
                &child_rel,
                depth + 1,
                max_depth,
                max_entries,
                count,
                out,
            );
        } else {
            out.push_str(&child_rel);
            out.push('\n');
            *count += 1;
        }
    }
}

/// Truncate `text` to `max_bytes`, preferring a trailing newline boundary.
#[must_use]
pub fn truncate_to_budget(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut cut = max_bytes;
    while cut > 0 && text.as_bytes()[cut - 1] != b'\n' {
        cut -= 1;
    }
    if cut == 0 {
        cut = max_bytes;
    }
    // Avoid splitting mid-char
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text[..cut].to_string()
}

/// Run `git diff --stat` (or full diff) truncated to `max_bytes`.
#[must_use]
pub fn read_git_diff_budgeted(max_bytes: usize, stat_only: bool) -> String {
    if max_bytes == 0 {
        return String::new();
    }
    let args: &[&str] = if stat_only {
        &["diff", "--stat"]
    } else {
        &["diff", "--color=never"]
    };
    let output = Command::new("git").args(args).output();
    let Ok(output) = output else {
        return String::new();
    };
    if !output.status.success() && output.stdout.is_empty() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    truncate_to_budget(&text, max_bytes)
}

/// Priority budget shares (40/35/15/10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetShares {
    /// Open-file tier.
    pub open_bytes: usize,
    /// `@file` tier.
    pub at_file_bytes: usize,
    /// Workspace tree tier.
    pub tree_bytes: usize,
    /// Git tier.
    pub git_bytes: usize,
}

impl BudgetShares {
    /// Split `total` with open > @file > tree > git priority.
    #[must_use]
    pub fn fair_share(total: usize) -> Self {
        if total == 0 {
            return Self {
                open_bytes: 0,
                at_file_bytes: 0,
                tree_bytes: 0,
                git_bytes: 0,
            };
        }
        let open = (total * 40) / 100;
        let at_file = (total * 35) / 100;
        let tree = (total * 15) / 100;
        let git = total - open - at_file - tree;
        Self {
            open_bytes: open,
            at_file_bytes: at_file,
            tree_bytes: tree,
            git_bytes: git,
        }
    }
}

/// Options for [`build_agent_context`].
#[derive(Debug, Clone)]
pub struct AgentContextOptions {
    /// Include workspace tree.
    pub include_tree: bool,
    /// Include git diff/stat.
    pub include_git_diff: bool,
    /// Prefer `git diff --stat` over full diff.
    pub git_stat_only: bool,
    /// Optional open-file path label.
    pub open_path: String,
    /// Optional open-file content.
    pub open_content: String,
    /// Tree depth cap.
    pub tree_max_depth: usize,
    /// Tree entry cap.
    pub tree_max_entries: usize,
}

impl Default for AgentContextOptions {
    fn default() -> Self {
        Self {
            include_tree: true,
            include_git_diff: true,
            git_stat_only: true,
            open_path: String::new(),
            open_content: String::new(),
            tree_max_depth: TREE_MAX_DEPTH,
            tree_max_entries: TREE_MAX_ENTRIES,
        }
    }
}

/// Build a budgeted agent context string.
#[must_use]
pub fn build_agent_context(
    input: &str,
    root: &Path,
    total_budget: usize,
    opts: &AgentContextOptions,
) -> String {
    let shares = BudgetShares::fair_share(total_budget);
    let mut out = String::new();

    // 1. Open file
    let mut open_used = 0_usize;
    if !opts.open_path.is_empty() && !opts.open_content.is_empty() && shares.open_bytes > 0 {
        let body = truncate_to_budget(&opts.open_content, shares.open_bytes);
        let header = format!("[open: {}]\n", opts.open_path);
        out.push_str(&header);
        out.push_str(&body);
        out.push('\n');
        open_used = header.len() + body.len() + 1;
    }

    // 2. @file mentions
    let at_budget_max = shares
        .at_file_bytes
        .saturating_add(shares.open_bytes.saturating_sub(open_used));
    let mut at_budget = ContextBudget::new(at_budget_max);
    let injected = resolve_and_inject(input, root, &mut at_budget);
    out.push_str(&injected);

    // 3. Workspace tree
    if opts.include_tree && shares.tree_bytes > 0 {
        let listing = list_workspace_tree(root, opts.tree_max_depth, opts.tree_max_entries);
        let clipped = truncate_to_budget(&listing, shares.tree_bytes);
        out.push_str("[workspace-tree]\n");
        out.push_str(&clipped);
        if clipped.is_empty() || !clipped.ends_with('\n') {
            out.push('\n');
        }
    }

    // 4. Git
    if opts.include_git_diff && shares.git_bytes > 0 {
        let diff = read_git_diff_budgeted(shares.git_bytes, opts.git_stat_only);
        if !diff.is_empty() {
            out.push_str(if opts.git_stat_only {
                "[git-diff-stat]\n"
            } else {
                "[git-diff]\n"
            });
            out.push_str(&diff);
            if !diff.ends_with('\n') {
                out.push('\n');
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    fn scratch() -> std::path::PathBuf {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("abi_fc_{}_{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn parse_file_mentions_extracts_paths() {
        let m = parse_file_mentions("hello @src/main.zig world");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].path, "src/main.zig");
        assert_eq!(m[0].start, 6);
        assert_eq!(m[0].end, 19);
    }

    #[test]
    fn parse_file_mentions_ignores_email() {
        assert!(parse_file_mentions("contact user@example.com").is_empty());
    }

    #[test]
    fn parse_file_mentions_requires_dot_or_slash() {
        assert!(parse_file_mentions("hello @world end").is_empty());
    }

    #[test]
    fn parse_file_mentions_multiple() {
        let m = parse_file_mentions("@file1.zig and @dir/file2.zig");
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].path, "file1.zig");
        assert_eq!(m[1].path, "dir/file2.zig");
    }

    #[test]
    fn validate_mention_path_rejects_escape() {
        assert_eq!(
            validate_mention_path("../etc/passwd"),
            Err(PathError::PathEscape)
        );
        assert_eq!(
            validate_mention_path("/etc/passwd"),
            Err(PathError::AbsolutePathRejected)
        );
        assert_eq!(validate_mention_path(""), Err(PathError::EmptyPath));
        assert!(validate_mention_path("src/main.zig").is_ok());
    }

    #[test]
    fn context_budget_tracks_consumption() {
        let mut budget = ContextBudget::new(100);
        assert_eq!(budget.remaining(), 100);
        budget.consume(30);
        assert_eq!(budget.remaining(), 70);
        assert!(!budget.can_fit(71));
        budget.consume(100);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn budget_shares_partition() {
        let s = BudgetShares::fair_share(1000);
        assert_eq!(s.open_bytes, 400);
        assert_eq!(s.at_file_bytes, 350);
        assert_eq!(s.tree_bytes, 150);
        assert_eq!(s.git_bytes, 100);
        assert_eq!(
            s.open_bytes + s.at_file_bytes + s.tree_bytes + s.git_bytes,
            1000
        );
    }

    #[test]
    fn resolve_and_inject_reads_real_file() {
        let dir = scratch();
        fs::write(dir.join("note.txt"), b"hello from file").unwrap();
        let mut budget = ContextBudget::new(DEFAULT_BUDGET_BYTES);
        let result = resolve_and_inject("see @note.txt please", &dir, &mut budget);
        assert!(result.contains("[file: note.txt]"));
        assert!(result.contains("hello from file"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_workspace_tree_lists_files() {
        let dir = scratch();
        fs::write(dir.join("a.rs"), b"1").unwrap();
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("sub/b.rs"), b"2").unwrap();
        let listing = list_workspace_tree(&dir, 2, 16);
        assert!(
            listing.contains("a.rs")
                || listing.contains(&format!("{}", dir.file_name().unwrap().to_string_lossy()))
        );
        assert!(!listing.contains(".."));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_agent_context_includes_workspace_tree_marker() {
        let dir = scratch();
        fs::write(dir.join("x.md"), b"# hi").unwrap();
        let ctx = build_agent_context(
            "inspect",
            &dir,
            DEFAULT_BUDGET_BYTES,
            &AgentContextOptions {
                include_git_diff: false,
                ..AgentContextOptions::default()
            },
        );
        assert!(ctx.contains("[workspace-tree]"));
        assert!(ctx.contains("inspect"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncate_to_budget_respects_max() {
        let clipped = truncate_to_budget("aaa\nbbb\nccc\n", 8);
        assert!(clipped.len() <= 8);
    }
}
