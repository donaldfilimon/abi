//! MCP process state: the scheduler and WDBX store shared by tool handlers.
//!
//! Ported from `src/mcp/state.zig`, simplified. Zig keeps one long-lived
//! scheduler and one long-lived WDBX session for the process's lifetime,
//! guarded by double-checked atomics. The Rust port instead opens fresh per
//! call: `wdbx_stats` is read-only, and `ai_complete` opens, mutates, and
//! drops its own [`VersionedStore`] so concurrent tool calls do not share a
//! live writer. A shared long-lived session would only pay for itself once a
//! tool needs cross-call store state (for example SEA modulator weights) or
//! once lock contention on the WAL becomes measurable.

use std::path::PathBuf;

use abi_core::Scheduler;
use abi_wdbx::{
    StoreLocation, StorePaths, VersionedStore, open_versioned_read_only,
    resolve_store_location_from_env,
};

/// Home-dir env var, Windows-aware.
#[cfg(windows)]
const HOME_VAR: &str = "USERPROFILE";
#[cfg(not(windows))]
const HOME_VAR: &str = "HOME";

/// Resolve the durable-store base path from the real process environment
/// through the resolver the CLI shares. `None` means an in-memory store.
fn resolve_wdbx_base_path() -> Option<PathBuf> {
    let xdg_data_home = abi_foundation::env::get("XDG_DATA_HOME");
    let home = abi_foundation::env::get(HOME_VAR);
    match resolve_store_location_from_env(xdg_data_home.as_deref(), home.as_deref()) {
        StoreLocation::Durable(base) => Some(base),
        StoreLocation::Memory(_) => None,
    }
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
    ///
    /// Open failures (corrupt/missing segment under the resolved path) return a
    /// disclosed in-memory line rather than an MCP internal error — CI runners
    /// and hosts with an unreadable `~/.abi/` must still answer `tools/call`.
    pub fn wdbx_stats_text(&self) -> Result<String, WdbxStatsError> {
        let store = match resolve_wdbx_base_path() {
            Some(base) => match open_versioned_read_only(&StorePaths::new(base)) {
                Ok(snapshot) => snapshot,
                Err(_) => return Ok(unavailable_stats_text()),
            },
            None => return Ok(in_memory_stats_text()),
        };
        let stats = store.stats();
        let dims = store
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

fn unavailable_stats_text() -> String {
    "kv=0 vectors=0 blocks=0 spatial=0 dims=null backend=cpu source=mcp-store-unavailable"
        .to_string()
}

/// Open the durable store at the resolved path, or `None` for in-memory.
///
/// Shared by every write-capable tool so they resolve the base path the same
/// way `wdbx_stats_text` does. Returns `Ok(None)` rather than an error for the
/// in-memory case — callers that need to mutate a store decide separately
/// whether "no persistence configured" is itself an error for them.
pub(crate) fn open_wdbx_store() -> Result<Option<VersionedStore>, WdbxStatsError> {
    match resolve_wdbx_base_path() {
        Some(base) => VersionedStore::open(StorePaths::new(base))
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
}
