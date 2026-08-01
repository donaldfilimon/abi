//! Shared CLI utilities.

use abi_wdbx::{DurableStore, StorePaths};

/// Resolve the durable store the same way MCP does.
///
/// `Ok(None)` means persistence was deliberately disabled or no home path is
/// available. An actual open/recovery/lock failure remains an error so callers
/// can disclose it instead of misreporting `no-store`.
pub(crate) fn open_store_result() -> Result<Option<DurableStore>, abi_wdbx::DurableError> {
    if let Some(path) = abi_foundation::env::get(abi_foundation::env::WDBX_PATH) {
        if path == ":memory:" {
            return Ok(None);
        }
        return DurableStore::open(StorePaths::new(path)).map(Some);
    }
    if abi_foundation::env::get(abi_foundation::env::WDBX_PERSIST)
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "no" | "off"))
    {
        return Ok(None);
    }
    let Some(home) = abi_foundation::env::get("HOME") else {
        return Ok(None);
    };
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).map(Some)
}

/// Compatibility helper for completion/training paths that already disclose
/// persistence only as available/unavailable. Security-sensitive callers such
/// as OS audit should use [`open_store_result`] and preserve the error detail.
pub(crate) fn open_store() -> Option<DurableStore> {
    open_store_result().ok().flatten()
}
