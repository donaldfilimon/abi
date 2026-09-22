//! Shared CLI utilities.

use abi_wdbx::{
    StoreLocation, StorePaths, VersionedError, VersionedStore, resolve_store_location_from_env,
};

/// Open the durable store at the location `abi_wdbx::resolve_store_location_from_env`
/// picks, the resolver MCP also uses, so `ABI_WDBX_PERSIST`, `ABI_WDBX_PATH`,
/// and `XDG_DATA_HOME` mean the same thing on both surfaces.
///
/// `Ok(None)` means persistence was deliberately disabled or no home path is
/// available. An actual open/recovery/lock failure remains an error so callers
/// can disclose it instead of misreporting `no-store`.
pub(crate) fn open_store_result() -> Result<Option<VersionedStore>, VersionedError> {
    let (xdg_data_home, home) = default_store_roots();
    match resolve_store_location_from_env(xdg_data_home.as_deref(), home.as_deref()) {
        StoreLocation::Durable(base) => VersionedStore::open(StorePaths::new(base)).map(Some),
        StoreLocation::Memory(_) => Ok(None),
    }
}

/// Ambient roots `(XDG_DATA_HOME, HOME)` backing the default
/// `$XDG_DATA_HOME/abi/wdbx` / `~/.abi/wdbx` store.
///
/// Both are the operator's live data, so the lib-test build withholds them
/// outright rather than creating state and holding a writer lock inside real
/// user data. A unit test that wants a store points `ABI_WDBX_PATH` at a
/// scratch directory; one that does not gets `no-store`.
#[cfg(not(test))]
pub(crate) fn default_store_roots() -> (Option<String>, Option<String>) {
    (
        abi_foundation::env::get("XDG_DATA_HOME"),
        abi_foundation::env::get("HOME"),
    )
}

#[cfg(test)]
pub(crate) fn default_store_roots() -> (Option<String>, Option<String>) {
    (None, None)
}

/// Compatibility helper for completion/training paths that already disclose
/// persistence only as available/unavailable. Security-sensitive callers such
/// as OS audit should use [`open_store_result`] and preserve the error detail.
pub(crate) fn open_store() -> Option<VersionedStore> {
    open_store_result().ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ABI_WDBX_PERSIST=0` wins over an explicit `ABI_WDBX_PATH`, as it does
    /// in MCP: the CLI once opened the path anyway.
    #[test]
    fn persist_disabled_wins_over_an_explicit_store_path() {
        let _guard = abi_foundation::env::lock_for_test();
        let scratch = std::env::temp_dir().join(format!(
            "abi-util-persist-off-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&scratch);
        let store = scratch.join("wdbx");
        abi_foundation::env::set_override(abi_foundation::env::WDBX_PERSIST, "0");
        abi_foundation::env::set_override(
            abi_foundation::env::WDBX_PATH,
            &store.display().to_string(),
        );

        let resolved = open_store_result();
        let created = scratch.exists();

        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PERSIST);
        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PATH);
        let _ = std::fs::remove_dir_all(&scratch);

        assert!(
            matches!(resolved, Ok(None)),
            "ABI_WDBX_PERSIST=0 must disable the store even with ABI_WDBX_PATH set"
        );
        assert!(
            !created,
            "a disabled store must create nothing at ABI_WDBX_PATH"
        );
    }

    /// Pins the lib-test build's refusal to fall back to `$HOME/.abi/wdbx` or
    /// `$XDG_DATA_HOME/abi/wdbx`.
    ///
    /// Scope: this pins the [`default_store_roots`] gate, not a suite-wide
    /// property. "No abi-cli test touches `~/.abi/`" is established by running
    /// each test binary under a scratch `HOME` and inspecting it; this test is
    /// what makes removing the gate fail loudly instead of silently.
    ///
    /// `HOME` and `XDG_DATA_HOME` are overridden to scratch directories so that
    /// a regression fails safely: if the gate is ever deleted, this test writes
    /// into the scratch tree it then asserts on, never into real user data.
    #[test]
    fn the_home_store_fallback_is_refused_in_the_test_build() {
        assert_eq!(
            default_store_roots(),
            (None, None),
            "the test build must never resolve the XDG or ~/.abi/wdbx fallbacks"
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
        let xdg = home.join("xdg");
        abi_foundation::env::set_override("XDG_DATA_HOME", &xdg.display().to_string());

        let resolved = open_store_result();
        let leaked = home.join(".abi").exists() || xdg.join("abi").exists();

        abi_foundation::env::clear_override("HOME");
        abi_foundation::env::clear_override("XDG_DATA_HOME");
        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PATH);
        abi_foundation::env::clear_override(abi_foundation::env::WDBX_PERSIST);
        let _ = std::fs::remove_dir_all(&home);

        assert!(
            matches!(resolved, Ok(None)),
            "an unconfigured store must report no-store, not open a home store"
        );
        assert!(
            !leaked,
            "no store state may be created under a home or XDG data directory"
        );
    }
}
