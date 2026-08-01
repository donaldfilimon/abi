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
    let Some(home) = default_store_home() else {
        return Ok(None);
    };
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).map(Some)
}

/// Home directory backing the default `~/.abi/wdbx` store.
///
/// `~/.abi/` is the operator's live store, so the lib-test build refuses the
/// fallback outright rather than creating state and holding a writer lock
/// inside real user data. A unit test that wants a store points
/// `ABI_WDBX_PATH` at a scratch directory; one that does not gets `no-store`.
#[cfg(not(test))]
pub(crate) fn default_store_home() -> Option<String> {
    abi_foundation::env::get("HOME")
}

#[cfg(test)]
pub(crate) fn default_store_home() -> Option<String> {
    None
}

/// Compatibility helper for completion/training paths that already disclose
/// persistence only as available/unavailable. Security-sensitive callers such
/// as OS audit should use [`open_store_result`] and preserve the error detail.
pub(crate) fn open_store() -> Option<DurableStore> {
    open_store_result().ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the lib-test build's refusal to fall back to `$HOME/.abi/wdbx`.
    ///
    /// Scope: this pins the [`default_store_home`] gate, not a suite-wide
    /// property. "No abi-cli test touches `~/.abi/`" is established by running
    /// each test binary under a scratch `HOME` and inspecting it; this test is
    /// what makes removing the gate fail loudly instead of silently.
    ///
    /// `HOME` is overridden to a scratch directory so that a regression fails
    /// safely: if the gate is ever deleted, this test writes into the scratch
    /// tree it then asserts on, never into the developer's real store.
    #[test]
    fn the_home_store_fallback_is_refused_in_the_test_build() {
        assert!(
            default_store_home().is_none(),
            "the test build must never resolve the ~/.abi/wdbx home fallback"
        );

        // `ABI_WDBX_PATH` overrides are process-global and os::tests sets one,
        // so take the shared lock rather than racing it.
        let _guard = abi_foundation::env::lock_for_test();
        let home = std::env::temp_dir().join(format!(
            "abi-util-home-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&home).expect("scratch home");
        // An empty override reads back as unset, which also neutralizes any
        // value the developer running the suite has exported.
        abi_foundation::env::set_override(abi_foundation::env::WDBX_PATH, "");
        abi_foundation::env::set_override(abi_foundation::env::WDBX_PERSIST, "");
        abi_foundation::env::set_override("HOME", &home.display().to_string());

        let resolved = open_store_result();
        let created = home.join(".abi");
        let leaked = created.exists();

        abi_foundation::env::clear_override("HOME");
        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PATH);
        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PERSIST);
        let _ = std::fs::remove_dir_all(&home);

        assert!(
            matches!(resolved, Ok(None)),
            "an unconfigured store must report no-store, not open a home store"
        );
        assert!(
            !leaked,
            "no store state may be created under a home directory"
        );
    }
}
