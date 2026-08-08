//! Credential storage: a JSON file backend by default, an opt-in macOS keychain.
//!
//! Ported from `src/foundation/credentials.zig` and its three leaf modules. The
//! Zig `Credentials` struct listed each provider as a separate optional slice and
//! then repeated that list in six places — `deinit`, the JSON reader, the JSON
//! writer, the keychain reader, the keychain writer, and `auth status`. Adding a
//! provider meant touching all six, and missing one was a silent bug.
//!
//! Here the field list lives once in [`CREDENTIAL_FIELDS`] and everything else
//! iterates it, so a new provider is a single-line change.

mod file;
mod keychain;
mod provider;
mod secret;
#[cfg(windows)]
mod windows_acl;

pub use file::{CREDENTIALS_PATH_ENV, credentials_path, write_owner_only_file_new};
pub use keychain::{BACKEND_ENV, backend_is_keychain};
pub use provider::{
    CredentialCapability, CredentialProvider, CredentialScope, FileCredentialProvider,
    LinuxSecretServiceProvider, MacOsKeychainProvider, ProviderImplementation, ProviderKind,
    SecretTransport, WindowsCredentialProtectionProvider, platform_credential_capabilities,
    platform_credential_providers, platform_credential_report,
};
pub use secret::Secret;

pub use file::mode_of;

use crate::errors::AbiError;

/// One credential slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CredentialField {
    /// JSON key and keychain account name, e.g. `openai_api_key`.
    pub key: &'static str,
    /// Human-readable provider name for `abi auth status`.
    pub label: &'static str,
}

impl CredentialField {
    /// `OpenAI` API key.
    pub const OPENAI_API_KEY: Self = Self {
        key: "openai_api_key",
        label: "OpenAI",
    };
    /// Anthropic API key.
    pub const ANTHROPIC_API_KEY: Self = Self {
        key: "anthropic_api_key",
        label: "Anthropic",
    };
    /// Discord bot token.
    pub const DISCORD_TOKEN: Self = Self {
        key: "discord_token",
        label: "Discord",
    };
    /// xAI Grok API key.
    pub const GROK_API_KEY: Self = Self {
        key: "grok_api_key",
        label: "Grok",
    };
    /// Twilio account SID.
    pub const TWILIO_ACCOUNT_SID: Self = Self {
        key: "twilio_account_sid",
        label: "Twilio SID",
    };
    /// Twilio auth token.
    pub const TWILIO_AUTH_TOKEN: Self = Self {
        key: "twilio_auth_token",
        label: "Twilio token",
    };
}

/// Every credential slot, in the order `abi auth status` lists them.
///
/// This is the single source of truth: the JSON reader and writer, both keychain
/// directions, and the status output all iterate it.
pub const CREDENTIAL_FIELDS: &[CredentialField] = &[
    CredentialField::OPENAI_API_KEY,
    CredentialField::ANTHROPIC_API_KEY,
    CredentialField::DISCORD_TOKEN,
    CredentialField::GROK_API_KEY,
    CredentialField::TWILIO_ACCOUNT_SID,
    CredentialField::TWILIO_AUTH_TOKEN,
];

/// The set of configured credentials.
///
/// Secrets are held as [`Secret`], so they are zeroed on drop and redacted in
/// `Debug` output — the Zig struct held raw slices, which any `{}`-format of the
/// struct would have printed in the clear.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Credentials {
    slots: Vec<(CredentialField, Secret)>,
}

impl Credentials {
    /// Empty credentials.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The secret for `field`, if configured.
    #[must_use]
    pub fn get(&self, field: CredentialField) -> Option<&Secret> {
        self.slots
            .iter()
            .find(|(slot, _)| *slot == field)
            .map(|(_, secret)| secret)
    }

    /// Set or clear `field`.
    ///
    /// Replacing a value drops the previous [`Secret`], which zeroes it.
    pub fn set(&mut self, field: CredentialField, value: Option<Secret>) {
        self.slots.retain(|(slot, _)| *slot != field);
        if let Some(secret) = value {
            self.slots.push((field, secret));
        }
    }

    /// Whether no credential is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The number of configured credentials.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Iterate configured credentials in [`CREDENTIAL_FIELDS`] order.
    ///
    /// Insertion order would make `abi auth status` reorder itself depending on
    /// which credential was set last.
    pub fn iter(&self) -> impl Iterator<Item = (CredentialField, &Secret)> {
        CREDENTIAL_FIELDS
            .iter()
            .filter_map(move |field| self.get(*field).map(|secret| (*field, secret)))
    }
}

/// Which backend a load or save will use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// JSON file with owner-only permissions.
    File,
    /// macOS login keychain.
    Keychain,
}

/// The backend that will actually be used.
///
/// `ABI_CREDENTIALS_BACKEND=keychain` selects the keychain *only on macOS*. Off
/// macOS the request is honoured in reporting — `abi auth status` still discloses
/// that keychain was asked for — but the file store is used, because there is no
/// linked backend. This avoids the worst outcome, which is an empty success that
/// reads as "you have no credentials".
#[must_use]
pub fn active_backend() -> Backend {
    if backend_is_keychain() && keychain::is_supported() {
        Backend::Keychain
    } else {
        Backend::File
    }
}

/// Load credentials from the active backend.
pub fn load() -> Result<Credentials, AbiError> {
    match active_backend() {
        Backend::Keychain => keychain::load(),
        Backend::File => file::load_from_path(file::credentials_path()?),
    }
}

/// Save credentials to the active backend.
///
/// There is no dual-write: when the keychain is active, secrets never also land
/// in the plaintext file.
pub fn save(creds: &Credentials) -> Result<(), AbiError> {
    match active_backend() {
        Backend::Keychain => keychain::save(creds),
        Backend::File => file::save_to_path(file::credentials_path()?, creds),
    }
}

/// Clear keychain-held secrets, if the keychain backend is active.
///
/// A no-op success otherwise — there is nothing to clear when no keychain backend
/// is linked. Used by `abi auth logout` so logout clears keychain secrets the
/// same way it clears the file.
pub fn clear_keychain() -> Result<(), AbiError> {
    if active_backend() == Backend::Keychain {
        keychain::clear()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env;

    #[test]
    fn get_and_set_roundtrip() {
        let mut creds = Credentials::new();
        assert_eq!(creds.len(), 0);

        creds.set(CredentialField::OPENAI_API_KEY, Some(Secret::new("sk-1")));
        assert_eq!(creds.len(), 1);
        assert_eq!(
            creds
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-1")
        );
        assert!(creds.get(CredentialField::GROK_API_KEY).is_none());
    }

    #[test]
    fn setting_none_clears_the_field() {
        let mut creds = Credentials::new();
        creds.set(CredentialField::DISCORD_TOKEN, Some(Secret::new("tok")));
        creds.set(CredentialField::DISCORD_TOKEN, None);
        assert!(creds.get(CredentialField::DISCORD_TOKEN).is_none());
        assert_eq!(creds.len(), 0);
    }

    #[test]
    fn setting_twice_replaces_rather_than_duplicates() {
        let mut creds = Credentials::new();
        creds.set(CredentialField::GROK_API_KEY, Some(Secret::new("old")));
        creds.set(CredentialField::GROK_API_KEY, Some(Secret::new("new")));
        assert_eq!(creds.len(), 1);
        assert_eq!(
            creds.get(CredentialField::GROK_API_KEY).map(Secret::expose),
            Some("new")
        );
    }

    #[test]
    fn iter_follows_declaration_order_not_insertion_order() {
        let mut creds = Credentials::new();
        // Insert back-to-front.
        creds.set(CredentialField::TWILIO_AUTH_TOKEN, Some(Secret::new("d")));
        creds.set(CredentialField::DISCORD_TOKEN, Some(Secret::new("c")));
        creds.set(CredentialField::OPENAI_API_KEY, Some(Secret::new("a")));

        let keys: Vec<&str> = creds.iter().map(|(field, _)| field.key).collect();
        assert_eq!(
            keys,
            ["openai_api_key", "discord_token", "twilio_auth_token"]
        );
    }

    #[test]
    fn field_list_is_the_documented_set() {
        let keys: Vec<&str> = CREDENTIAL_FIELDS.iter().map(|f| f.key).collect();
        assert_eq!(
            keys,
            [
                "openai_api_key",
                "anthropic_api_key",
                "discord_token",
                "grok_api_key",
                "twilio_account_sid",
                "twilio_auth_token",
            ]
        );
    }

    #[test]
    fn field_keys_are_unique() {
        // A duplicate key would make `set` silently clobber another provider.
        let mut keys: Vec<&str> = CREDENTIAL_FIELDS.iter().map(|f| f.key).collect();
        let total = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), total);
    }

    #[test]
    fn debug_output_redacts_every_secret() {
        let mut creds = Credentials::new();
        creds.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-do-not-print-me")),
        );
        let rendered = format!("{creds:?}");
        assert!(
            !rendered.contains("do-not-print-me"),
            "Debug leaked a secret: {rendered}"
        );
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn default_backend_is_the_file_store() {
        let _guard = env::lock_for_test();
        env::set_override(BACKEND_ENV, "");
        assert_eq!(active_backend(), Backend::File);
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn keychain_request_resolves_by_platform_support() {
        let _guard = env::lock_for_test();
        env::set_override(BACKEND_ENV, "keychain");
        let expected = if cfg!(target_os = "macos") {
            Backend::Keychain
        } else {
            // Off macOS the request must fall back to the file store rather
            // than reporting an empty keychain.
            Backend::File
        };
        assert_eq!(active_backend(), expected);
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn clear_keychain_is_a_noop_success_under_the_file_backend() {
        let _guard = env::lock_for_test();
        env::set_override(BACKEND_ENV, "");
        assert!(clear_keychain().is_ok());
        env::clear_override(BACKEND_ENV);
    }

    #[test]
    fn file_backend_roundtrips_through_the_public_api() {
        let _guard = env::lock_for_test();
        let path = crate::temp_path::temp_file_path("abi_credentials_public_api", "json");
        env::set_override(BACKEND_ENV, "");
        env::set_override(CREDENTIALS_PATH_ENV, &path.to_string_lossy());

        let mut creds = Credentials::new();
        creds.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-public-api")),
        );
        save(&creds).expect("save via public API");

        let loaded = load().expect("load via public API");
        assert_eq!(
            loaded
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-public-api")
        );

        std::fs::remove_file(&path).ok();
        env::clear_override(CREDENTIALS_PATH_ENV);
        env::clear_override(BACKEND_ENV);
    }
}
