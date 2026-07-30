//! MCP process state: the scheduler and WDBX store shared by tool handlers.
//!
//! Ported from `src/mcp/state.zig`, simplified. Zig keeps one long-lived
//! scheduler and one long-lived WDBX session for the process's lifetime,
//! guarded by double-checked atomics. The Rust port instead opens fresh per
//! call: `wdbx_stats` is read-only, and `ai_complete` opens, mutates, and
//! drops its own [`DurableStore`] so concurrent tool calls do not share a
//! live writer. A shared long-lived session would only pay for itself once a
//! tool needs cross-call store state (for example SEA modulator weights) or
//! once lock contention on the WAL becomes measurable.

use std::path::PathBuf;

use abi_core::Scheduler;
use abi_wdbx::{DurableStore, StorePaths};

/// Home-dir env var, Windows-aware.
#[cfg(windows)]
const HOME_VAR: &str = "USERPROFILE";
#[cfg(not(windows))]
const HOME_VAR: &str = "HOME";

const PATH_ENV: &str = "ABI_WDBX_PATH";
const PERSIST_ENV: &str = "ABI_WDBX_PERSIST";
const MEMORY_SENTINEL: &str = ":memory:";
const DEFAULT_SUBPATH: &str = ".abi/wdbx";

fn is_falsey(value: &str) -> bool {
    matches!(value, "0" | "false" | "no" | "off")
}

/// The pure resolution logic behind [`resolve_wdbx_base_path`], taking each
/// input explicitly so it is testable without mutating process-global
/// environment state. `None` means an in-memory store.
///
/// Matches `durable_store.resolveConfig` in Zig: `persist` falsey wins
/// outright; then `path` (with the `:memory:` sentinel); then, off Windows,
/// `xdg`; then `home` joined with the default subpath; then in-memory if
/// `home` is absent too.
fn resolve_from(
    persist: Option<&str>,
    path: Option<&str>,
    xdg: Option<&str>,
    home: Option<&str>,
) -> Option<PathBuf> {
    if persist.is_some_and(is_falsey) {
        return None;
    }
    if let Some(p) = path {
        return if p == MEMORY_SENTINEL {
            None
        } else {
            Some(PathBuf::from(p))
        };
    }
    if !cfg!(windows)
        && let Some(xdg) = xdg
    {
        return Some(PathBuf::from(xdg).join("abi").join("wdbx"));
    }
    home.map(|home| PathBuf::from(home).join(DEFAULT_SUBPATH))
}

/// Resolve the durable-store base path from the real process environment.
fn resolve_wdbx_base_path() -> Option<PathBuf> {
    resolve_from(
        std::env::var(PERSIST_ENV).ok().as_deref(),
        std::env::var(PATH_ENV).ok().as_deref(),
        std::env::var("XDG_DATA_HOME").ok().as_deref(),
        std::env::var(HOME_VAR).ok().as_deref(),
    )
}

/// `wdbx_stats` could not read the durable store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WdbxStatsError;

impl std::fmt::Display for WdbxStatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to open the WDBX store")
    }
}

impl std::error::Error for WdbxStatsError {}

/// Shared server state. Presently stateless — each accessor resolves and
/// opens fresh — but kept as a type so tool dispatch has one thing to hold,
/// matching the shape a future shared session will take.
#[derive(Debug, Default, Clone, Copy)]
pub struct McpState;

impl McpState {
    /// A fresh, stateless handle.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// `scheduler_stats` / `scheduler_info` text, matching
    /// `ai_tools.schedulerStatsText` in Zig.
    #[must_use]
    pub fn scheduler_stats_text(&self) -> String {
        let scheduler = Scheduler::new();
        format!("scheduler {} source=mcp-server", scheduler.stats().render())
    }

    /// `wdbx_stats` text, matching `ai_tools.wdbxStatsText` in Zig — with one
    /// disclosed divergence: Zig reports the linked GPU backend name
    /// (`metal` on this machine); no Rust GPU backend is linked yet, so this
    /// honestly reports `cpu` rather than fabricating an accelerator claim.
    pub fn wdbx_stats_text(&self) -> Result<String, WdbxStatsError> {
        let store = match resolve_wdbx_base_path() {
            Some(base) => DurableStore::open(StorePaths::new(base)).map_err(|_| WdbxStatsError)?,
            None => return Ok(in_memory_stats_text()),
        };
        let stats = store.stats();
        let dims = store
            .snapshot()
            .vector_dimensions()
            .map_or_else(|| "null".to_string(), |d| d.to_string());
        Ok(format!(
            "kv={} vectors={} blocks={} spatial={} dims={dims} backend=cpu source=mcp-store",
            stats.kv_entries, stats.vectors, stats.blocks, stats.spatial_records
        ))
    }
}

fn in_memory_stats_text() -> String {
    "kv=0 vectors=0 blocks=0 spatial=0 dims=null backend=cpu source=mcp-store".to_string()
}

/// Open the durable store at the resolved path, or `None` for in-memory.
///
/// Shared by every write-capable tool so they resolve the base path the same
/// way `wdbx_stats_text` does. Returns `Ok(None)` rather than an error for the
/// in-memory case — callers that need to mutate a store decide separately
/// whether "no persistence configured" is itself an error for them.
pub(crate) fn open_wdbx_store() -> Result<Option<abi_wdbx::DurableStore>, WdbxStatsError> {
    match resolve_wdbx_base_path() {
        Some(base) => DurableStore::open(StorePaths::new(base))
            .map(Some)
            .map_err(|_| WdbxStatsError),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_stats_text_is_all_zero_at_rest() {
        let state = McpState::new();
        assert_eq!(
            state.scheduler_stats_text(),
            "scheduler running=0 pending=0 completed=0 failed=0 cancelled=0 total_tasks=0 source=mcp-server"
        );
    }

    #[test]
    fn falsey_persist_forces_in_memory_regardless_of_path() {
        assert_eq!(
            resolve_from(Some("0"), Some("/tmp/store"), None, Some("/home/x")),
            None
        );
        assert_eq!(
            resolve_from(Some("false"), None, None, Some("/home/x")),
            None
        );
    }

    #[test]
    fn memory_sentinel_path_forces_in_memory() {
        assert_eq!(
            resolve_from(None, Some(MEMORY_SENTINEL), None, Some("/home/x")),
            None
        );
    }

    #[test]
    fn explicit_path_wins_over_xdg_and_home() {
        assert_eq!(
            resolve_from(None, Some("/explicit/path"), Some("/xdg"), Some("/home/x")),
            Some(PathBuf::from("/explicit/path"))
        );
    }

    #[test]
    fn xdg_data_home_wins_over_home_off_windows() {
        if cfg!(windows) {
            return;
        }
        assert_eq!(
            resolve_from(None, None, Some("/xdg"), Some("/home/x")),
            Some(PathBuf::from("/xdg/abi/wdbx"))
        );
    }

    #[test]
    fn home_is_the_last_resort_default() {
        assert_eq!(
            resolve_from(None, None, None, Some("/home/x")),
            Some(PathBuf::from("/home/x/.abi/wdbx"))
        );
    }

    #[test]
    fn no_home_falls_back_to_in_memory() {
        assert_eq!(resolve_from(None, None, None, None), None);
    }
}
