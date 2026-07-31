//! Shared CLI utilities.

use abi_wdbx::{DurableStore, StorePaths};

/// Resolve the durable store the same way MCP does, or `None` for in-memory.
pub(crate) fn open_store() -> Option<DurableStore> {
    if let Ok(path) = std::env::var("ABI_WDBX_PATH") {
        if path == ":memory:" {
            return None;
        }
        return DurableStore::open(StorePaths::new(path)).ok();
    }
    if matches!(
        std::env::var("ABI_WDBX_PERSIST").as_deref(),
        Ok("0" | "false" | "no" | "off")
    ) {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).ok()
}
