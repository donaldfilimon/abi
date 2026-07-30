//! Process-environment access and the canonical ABI variable names.
//!
//! Ported from `src/foundation/env.zig`. The Zig version needed an explicit
//! `install(init.environ_map)` call from every entrypoint because that
//! toolchain exposes the environment only through `std.process.Init` — there is
//! no global `getenv`. Rust's `std::env::var` has no such restriction, so the
//! install ceremony is dropped; `get` reads the process environment directly.
//!
//! The one behaviour worth preserving is that an *empty* value counts as unset,
//! because callers rely on that to fall through to their documented defaults.
//!
//! Tests need to vary the environment without the cross-test interference that
//! `std::env::set_var` causes (it is process-global and, since Rust 2024,
//! `unsafe`). [`set_override`] provides a per-process overlay that `get`
//! consults first.

use std::collections::HashMap;
use std::sync::RwLock;

static OVERRIDES: RwLock<Option<HashMap<String, String>>> = RwLock::new(None);

/// Look up an environment variable.
///
/// Returns `None` when the variable is unset, empty, or not valid UTF-8 —
/// matching the Zig contract where an empty value is indistinguishable from an
/// absent one.
#[must_use]
pub fn get(key: &str) -> Option<String> {
    if let Ok(guard) = OVERRIDES.read()
        && let Some(map) = guard.as_ref()
        && let Some(value) = map.get(key)
    {
        return if value.is_empty() {
            None
        } else {
            Some(value.clone())
        };
    }

    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

/// Look up an environment variable, falling back to `default` when unset.
#[must_use]
pub fn get_or(key: &str, default: &str) -> String {
    get(key).unwrap_or_else(|| default.to_string())
}

/// Interpret a variable as a boolean flag.
///
/// Truthy values are `1`, `true`, `yes` and `on`, case-insensitively. Anything
/// else — including an unset variable — is false. This is the rule the Zig side
/// applied to `ABI_WDBX_ALLOW_MEMORY_FALLBACK`, generalized.
#[must_use]
pub fn get_bool(key: &str) -> bool {
    match get(key) {
        Some(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Parse a variable as an integer, returning `None` when unset or malformed.
#[must_use]
pub fn get_parsed<T: std::str::FromStr>(key: &str) -> Option<T> {
    get(key)?.trim().parse().ok()
}

/// Install a test-local override for `key`.
///
/// Overrides take precedence over the real process environment. Setting an
/// empty value makes `get` report the variable as unset.
pub fn set_override(key: &str, value: &str) {
    let mut guard = OVERRIDES.write().expect("env override lock poisoned");
    guard
        .get_or_insert_with(HashMap::new)
        .insert(key.to_string(), value.to_string());
}

/// Remove a single override, restoring the real process environment for `key`.
pub fn clear_override(key: &str) {
    if let Ok(mut guard) = OVERRIDES.write()
        && let Some(map) = guard.as_mut()
    {
        map.remove(key);
    }
}

/// Drop every override.
pub fn reset_overrides() {
    if let Ok(mut guard) = OVERRIDES.write() {
        *guard = None;
    }
}

/// Serialize tests that mutate a *shared* environment key.
///
/// Overrides are process-global while `cargo test` runs tests in parallel
/// threads, so two tests touching the same key (e.g. `TMPDIR`) would otherwise
/// race. Tests using a key unique to themselves do not need this.
///
/// The guard is deliberately returned rather than taken internally: holding it
/// for the whole test body is what provides the exclusion.
pub fn lock_for_test() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// --- Canonical environment variable names ---
//
// Shared across MCP, the WDBX REST/cluster listeners, and the connectors. These
// names are an external contract: operators set them in shell profiles and
// service units, so they must not be renamed during the port.

/// Port for the MCP HTTP transport.
pub const MCP_HTTP_PORT: &str = "ABI_MCP_HTTP_PORT";
/// Bearer token required by the MCP HTTP transport.
pub const MCP_HTTP_TOKEN: &str = "ABI_MCP_HTTP_TOKEN";
/// Bearer token required by the WDBX REST surface.
pub const WDBX_REST_TOKEN: &str = "ABI_WDBX_REST_TOKEN";
/// Port for the WDBX REST surface.
pub const WDBX_REST_PORT: &str = "ABI_WDBX_REST_PORT";
/// Shared token authenticating WDBX cluster peers.
pub const WDBX_CLUSTER_TOKEN: &str = "ABI_WDBX_CLUSTER_TOKEN";
/// Comma-separated list of WDBX cluster peer addresses.
pub const WDBX_CLUSTER_PEERS: &str = "ABI_WDBX_CLUSTER_PEERS";
/// Path to the TLS certificate used by WDBX listeners.
pub const WDBX_TLS_CERT: &str = "ABI_WDBX_TLS_CERT";
/// Path to the TLS private key used by WDBX listeners.
pub const WDBX_TLS_KEY: &str = "ABI_WDBX_TLS_KEY";
/// Override for the WDBX data directory.
pub const WDBX_PATH: &str = "ABI_WDBX_PATH";
/// Whether WDBX should persist to disk.
pub const WDBX_PERSIST: &str = "ABI_WDBX_PERSIST";
/// When truthy, MCP falls back to an empty in-memory WDBX store if a durable
/// open fails. The default is fail-closed.
pub const WDBX_ALLOW_MEMORY_FALLBACK: &str = "ABI_WDBX_ALLOW_MEMORY_FALLBACK";
/// Token-bucket capacity for WDBX rate limiting.
pub const WDBX_RATE_LIMIT_CAPACITY: &str = "ABI_WDBX_RATE_LIMIT_CAPACITY";
/// Token-bucket refill rate for WDBX rate limiting.
pub const WDBX_RATE_LIMIT_REFILL: &str = "ABI_WDBX_RATE_LIMIT_REFILL";
/// Endpoint for a local llama.cpp server.
pub const LLAMA_CPP_ENDPOINT: &str = "ABI_LLAMA_CPP_ENDPOINT";
/// Endpoint for a local MLX server.
pub const MLX_ENDPOINT: &str = "ABI_MLX_ENDPOINT";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_variable_reports_none() {
        assert!(get("ABI_DEFINITELY_UNSET_VAR_XYZ").is_none());
    }

    #[test]
    fn override_is_visible_and_removable() {
        let key = "ABI_TEST_OVERRIDE_VISIBLE";
        set_override(key, "hello");
        assert_eq!(get(key).as_deref(), Some("hello"));
        clear_override(key);
        assert!(get(key).is_none());
    }

    #[test]
    fn empty_value_counts_as_unset() {
        let key = "ABI_TEST_OVERRIDE_EMPTY";
        set_override(key, "");
        assert!(get(key).is_none());
        clear_override(key);
    }

    #[test]
    fn get_or_supplies_the_default() {
        let key = "ABI_TEST_OVERRIDE_DEFAULT";
        clear_override(key);
        assert_eq!(get_or(key, "fallback"), "fallback");
        set_override(key, "explicit");
        assert_eq!(get_or(key, "fallback"), "explicit");
        clear_override(key);
    }

    #[test]
    fn get_bool_accepts_the_documented_truthy_set() {
        let key = "ABI_TEST_OVERRIDE_BOOL";
        for truthy in ["1", "true", "TRUE", "yes", "Yes", "on", "ON"] {
            set_override(key, truthy);
            assert!(get_bool(key), "{truthy} should be truthy");
        }
        for falsy in ["0", "false", "no", "off", "", "maybe"] {
            set_override(key, falsy);
            assert!(!get_bool(key), "{falsy} should be falsy");
        }
        clear_override(key);
    }

    #[test]
    fn get_parsed_reads_integers_and_rejects_junk() {
        let key = "ABI_TEST_OVERRIDE_PORT";
        set_override(key, "8080");
        assert_eq!(get_parsed::<u16>(key), Some(8080));
        set_override(key, " 8080 ");
        assert_eq!(get_parsed::<u16>(key), Some(8080));
        set_override(key, "not-a-port");
        assert_eq!(get_parsed::<u16>(key), None);
        clear_override(key);
    }

    #[test]
    fn canonical_names_are_stable() {
        // These are an operator-facing contract; a rename is a breaking change.
        assert_eq!(MCP_HTTP_TOKEN, "ABI_MCP_HTTP_TOKEN");
        assert_eq!(WDBX_REST_TOKEN, "ABI_WDBX_REST_TOKEN");
        assert_eq!(WDBX_PATH, "ABI_WDBX_PATH");
        assert_eq!(WDBX_ALLOW_MEMORY_FALLBACK, "ABI_WDBX_ALLOW_MEMORY_FALLBACK");
    }
}
