//! Opt-in macOS login-keychain credential backend.
//!
//! Ported from `src/foundation/keychain.zig` plus
//! `src/foundation/credentials_keychain.zig`. The Zig version reached
//! Security.framework's `SecItem` API through hand-declared CoreFoundation and
//! Security externs; here that comes from the `security-framework` crate, so the
//! `unsafe` FFI and `CFType` lifetime management are maintained upstream.
//!
//! Two properties of the Zig implementation are deliberately preserved:
//!
//! - **It never shells out to `/usr/bin/security`.** A secret passed on argv is
//!   visible to other users through `ps` on a multi-user machine.
//! - **Non-macOS targets return an error, not a silent success.** Windows
//!   Credential Manager and Linux Secret Service were documented as Proposed and
//!   unimplemented. A cross-platform crate such as `keyring` would make them work
//!   — which would be a *widening* of what the project claims to support, so it
//!   is left as a deliberate follow-up rather than folded into the port.
//!
//! Scope, restated from the Zig docs so it does not inflate: this is the macOS
//! login keychain with the OS's default at-rest protection. It is **not**
//! hardware-backed (no Secure Enclave, no biometric gating) and has **not** been
//! independently audited.

use super::{CREDENTIAL_FIELDS, Credentials, Secret};
use crate::env;
use crate::errors::AbiError;

/// Environment variable selecting the credential backend.
pub const BACKEND_ENV: &str = "ABI_CREDENTIALS_BACKEND";

/// Service name under which every credential field is stored.
///
/// Each field's keychain *account* is its JSON key, e.g. `openai_api_key`.
pub const SERVICE: &str = "abi-credentials";

/// Whether `ABI_CREDENTIALS_BACKEND` selects the keychain backend.
///
/// Only the exact value `keychain` counts. Anything else — including an
/// unrecognised value like `vault` — keeps the default file backend, so an
/// existing `~/.abi/credentials.json` user is never silently migrated.
#[must_use]
pub fn backend_is_keychain() -> bool {
    env::get(BACKEND_ENV).is_some_and(|value| value == "keychain")
}

/// Whether this build can actually talk to an OS keychain.
#[must_use]
pub const fn is_supported() -> bool {
    cfg!(target_os = "macos")
}

/// Read every credential field from the keychain.
pub fn load() -> Result<Credentials, AbiError> {
    let mut creds = Credentials::default();
    for field in CREDENTIAL_FIELDS {
        if let Some(secret) = load_field(field.key)? {
            creds.set(*field, Some(secret));
        }
    }
    Ok(creds)
}

/// Mirror `creds` into the keychain.
///
/// Present fields are stored or overwritten; absent fields are actively
/// **deleted**, so clearing a credential does not leave a stale secret behind.
/// That asymmetry was explicitly tested on the Zig side and is retained.
pub fn save(creds: &Credentials) -> Result<(), AbiError> {
    for field in CREDENTIAL_FIELDS {
        match creds.get(*field) {
            Some(secret) => store_field(field.key, secret)?,
            None => delete_field(field.key)?,
        }
    }
    Ok(())
}

/// Delete every credential field from the keychain.
pub fn clear() -> Result<(), AbiError> {
    save(&Credentials::default())
}

#[cfg(target_os = "macos")]
mod backend {
    use super::{AbiError, SERVICE, Secret};
    use security_framework::passwords::{
        delete_generic_password, get_generic_password, set_generic_password,
    };

    /// `errSecItemNotFound` — the only Security.framework status that means
    /// "this credential is not configured" rather than "the operation failed".
    ///
    /// Discriminating on it matters: treating *every* error as absence would
    /// report a locked or inaccessible keychain as "you have no credentials",
    /// which is the silent-empty-success failure this module exists to avoid.
    const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;

    pub fn load_field(account: &str) -> Result<Option<Secret>, AbiError> {
        match get_generic_password(SERVICE, account) {
            Ok(bytes) => {
                let text = String::from_utf8(bytes).map_err(|_| AbiError::ParseError)?;
                Ok(Some(Secret::new(text)))
            }
            Err(err) if err.code() == ERR_SEC_ITEM_NOT_FOUND => Ok(None),
            Err(_) => Err(AbiError::PermissionDenied),
        }
    }

    pub fn store_field(account: &str, secret: &Secret) -> Result<(), AbiError> {
        set_generic_password(SERVICE, account, secret.expose().as_bytes())
            .map_err(|_| AbiError::PermissionDenied)
    }

    pub fn delete_field(account: &str) -> Result<(), AbiError> {
        match delete_generic_password(SERVICE, account) {
            // Idempotent: deleting an item that was never there is success.
            Ok(()) => Ok(()),
            Err(err) if err.code() == ERR_SEC_ITEM_NOT_FOUND => Ok(()),
            Err(_) => Err(AbiError::PermissionDenied),
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod backend {
    use super::{AbiError, Secret};

    pub fn load_field(_account: &str) -> Result<Option<Secret>, AbiError> {
        Err(AbiError::UnsupportedOperation)
    }

    pub fn store_field(_account: &str, _secret: &Secret) -> Result<(), AbiError> {
        Err(AbiError::UnsupportedOperation)
    }

    pub fn delete_field(_account: &str) -> Result<(), AbiError> {
        Err(AbiError::UnsupportedOperation)
    }
}

use backend::{delete_field, load_field, store_field};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::CredentialField;

    #[test]
    fn backend_defaults_to_file() {
        let _guard = env::lock_for_test();
        env::set_override(BACKEND_ENV, "");
        assert!(!backend_is_keychain());
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn backend_selects_keychain_on_exact_match() {
        let _guard = env::lock_for_test();
        env::set_override(BACKEND_ENV, "keychain");
        assert!(backend_is_keychain());
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn unrecognised_backend_values_keep_the_file_store() {
        let _guard = env::lock_for_test();
        for value in ["vault", "Keychain", "KEYCHAIN", "keychain "] {
            env::set_override(BACKEND_ENV, value);
            assert!(
                !backend_is_keychain(),
                "{value:?} must not select the keychain"
            );
        }
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn support_matches_the_documented_platform_boundary() {
        assert_eq!(is_supported(), cfg!(target_os = "macos"));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platforms_error_rather_than_silently_succeed() {
        // The critical property: never report empty success, which would look
        // like "no credentials configured" instead of "no backend here".
        assert_eq!(load().unwrap_err(), AbiError::UnsupportedOperation);
        assert_eq!(clear().unwrap_err(), AbiError::UnsupportedOperation);
    }

    /// Round-trips against the real macOS login keychain.
    ///
    /// Opt-in via `ABI_KEYCHAIN_TEST=1`: it writes to the developer's actual
    /// keychain and can raise a trust prompt, so it must not run by default.
    /// Reads the real process environment rather than the override overlay,
    /// since the gate is about the host, not about simulated config.
    #[cfg(target_os = "macos")]
    #[test]
    fn keychain_roundtrip_including_stale_deletion() {
        if std::env::var_os("ABI_KEYCHAIN_TEST").is_none() {
            return;
        }

        let mut creds = Credentials::default();
        creds.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-keychain-test")),
        );
        creds.set(
            CredentialField::DISCORD_TOKEN,
            Some(Secret::new("discord-keychain-test")),
        );
        save(&creds).expect("save to keychain");

        let loaded = load().expect("load from keychain");
        assert_eq!(
            loaded
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-keychain-test")
        );
        assert_eq!(
            loaded
                .get(CredentialField::DISCORD_TOKEN)
                .map(Secret::expose),
            Some("discord-keychain-test")
        );
        assert!(loaded.get(CredentialField::ANTHROPIC_API_KEY).is_none());

        // Saving without discord_token must DELETE the stale entry, not merely
        // leave it un-overwritten.
        let mut updated = Credentials::default();
        updated.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-keychain-test")),
        );
        save(&updated).expect("save updated");

        let reloaded = load().expect("reload");
        assert_eq!(
            reloaded
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-keychain-test")
        );
        assert!(
            reloaded.get(CredentialField::DISCORD_TOKEN).is_none(),
            "a cleared field must delete its keychain entry"
        );

        clear().expect("clear");
        assert!(load().expect("load after clear").is_empty());
    }
}
