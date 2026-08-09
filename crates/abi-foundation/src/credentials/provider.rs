//! Additive, object-safe credential-provider capabilities.
//!
//! The existing [`super::Backend`] selection remains authoritative: the file
//! backend is still the default and macOS Keychain remains opt-in. These
//! providers expose explicit platform adapters for inventory and callers that
//! deliberately select one; constructing a provider never changes defaults.

use super::{Credentials, file, keychain, linux_secret_service};
use crate::errors::AbiError;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stable identity of a credential provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProviderKind {
    /// Existing owner-only JSON credential file.
    File,
    /// macOS login Keychain through Security.framework.
    MacOsKeychain,
    /// Existing JSON file protected by an owner-only Windows DACL.
    WindowsProtectedFile,
    /// Linux Secret Service adapter over an encrypted in-process D-Bus session.
    LinuxSecretService,
}

impl ProviderKind {
    /// Stable machine-readable provider label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::MacOsKeychain => "macos_keychain",
            Self::WindowsProtectedFile => "windows_protected_file",
            Self::LinuxSecretService => "linux_secret_service",
        }
    }
}

/// Maturity of the provider implementation, independent of host availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderImplementation {
    /// A runtime implementation exists for the provider's target platform.
    Runtime,
    /// The adapter is compile-checked but still lacks cross-target runtime proof.
    CompileChecked,
    /// Capability-only placeholder with no secret-store operations attached.
    Stub,
}

impl ProviderImplementation {
    const fn label(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::CompileChecked => "compile_checked",
            Self::Stub => "stub",
        }
    }
}

/// How secret bytes cross the provider boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretTransport {
    /// Direct Rust/OS API calls; no child-process command line is involved.
    InProcessApi,
}

/// Secret families accepted by this provider interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialScope {
    /// Only the fixed ABI service-credential fields.
    AbiServiceCredentials,
}

/// Claim-honest provider capability evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CredentialCapability {
    /// Provider represented by this record.
    pub provider: ProviderKind,
    /// Implementation maturity.
    pub implementation: ProviderImplementation,
    /// Whether provider-specific code is compiled into this target.
    pub compiled: bool,
    /// Whether this build has an attached implementation it may call.
    pub available: bool,
    /// Whether this provider instance completed an operation successfully.
    pub runtime_verified: bool,
    /// Secrets use in-process APIs and never a child-process argv.
    pub secret_transport: SecretTransport,
    /// WDBX keys are outside the provider's accepted secret family.
    pub scope: CredentialScope,
}

impl CredentialCapability {
    /// Stable, secret-free capability report line.
    #[must_use]
    pub fn status_line(self) -> String {
        format!(
            "provider={} implementation={} compiled={} available={} runtime_verified={} secret_transport=in_process_api scope=abi_service_credentials",
            self.provider.label(),
            self.implementation.label(),
            self.compiled,
            self.available,
            self.runtime_verified
        )
    }
}

/// Object-safe credential storage boundary.
///
/// Provider methods use the fixed [`super::CREDENTIAL_FIELDS`] set. WDBX key
/// material is intentionally outside this interface.
pub trait CredentialProvider: std::fmt::Debug + Send + Sync {
    /// Current capability evidence for this provider instance.
    fn capability(&self) -> CredentialCapability;

    /// Load configured ABI service credentials.
    fn load(&self) -> Result<Credentials, AbiError>;

    /// Replace the provider's ABI service credentials.
    fn save(&self, credentials: &Credentials) -> Result<(), AbiError>;

    /// Remove every ABI service credential from this provider.
    fn clear(&self) -> Result<(), AbiError>;
}

fn capability(
    provider: ProviderKind,
    implementation: ProviderImplementation,
    compiled: bool,
    available: bool,
    runtime_verified: bool,
) -> CredentialCapability {
    CredentialCapability {
        provider,
        implementation,
        compiled,
        available,
        runtime_verified,
        secret_transport: SecretTransport::InProcessApi,
        scope: CredentialScope::AbiServiceCredentials,
    }
}

fn verified<T>(evidence: &AtomicBool, result: Result<T, AbiError>) -> Result<T, AbiError> {
    if result.is_ok() {
        evidence.store(true, Ordering::Release);
    }
    result
}

/// Explicit adapter for the existing default owner-only credential file.
#[derive(Default)]
pub struct FileCredentialProvider {
    runtime_verified: AtomicBool,
}

impl std::fmt::Debug for FileCredentialProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileCredentialProvider")
            .field("capability", &self.capability())
            .finish()
    }
}

impl CredentialProvider for FileCredentialProvider {
    fn capability(&self) -> CredentialCapability {
        capability(
            ProviderKind::File,
            ProviderImplementation::Runtime,
            true,
            true,
            self.runtime_verified.load(Ordering::Acquire),
        )
    }

    fn load(&self) -> Result<Credentials, AbiError> {
        verified(
            &self.runtime_verified,
            file::load_from_path(file::credentials_path()?),
        )
    }

    fn save(&self, credentials: &Credentials) -> Result<(), AbiError> {
        verified(
            &self.runtime_verified,
            file::save_to_path(file::credentials_path()?, credentials),
        )
    }

    fn clear(&self) -> Result<(), AbiError> {
        self.save(&Credentials::default())
    }
}

/// Explicit adapter for the existing macOS login-Keychain implementation.
#[derive(Default)]
pub struct MacOsKeychainProvider {
    runtime_verified: AtomicBool,
}

impl std::fmt::Debug for MacOsKeychainProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MacOsKeychainProvider")
            .field("capability", &self.capability())
            .finish()
    }
}

impl CredentialProvider for MacOsKeychainProvider {
    fn capability(&self) -> CredentialCapability {
        let supported = keychain::is_supported();
        capability(
            ProviderKind::MacOsKeychain,
            ProviderImplementation::Runtime,
            cfg!(target_os = "macos"),
            supported,
            self.runtime_verified.load(Ordering::Acquire),
        )
    }

    fn load(&self) -> Result<Credentials, AbiError> {
        verified(&self.runtime_verified, keychain::load())
    }

    fn save(&self, credentials: &Credentials) -> Result<(), AbiError> {
        verified(&self.runtime_verified, keychain::save(credentials))
    }

    fn clear(&self) -> Result<(), AbiError> {
        verified(&self.runtime_verified, keychain::clear())
    }
}

/// Windows protected-file adapter backed by the existing owner-only DACL path.
///
/// This is not Windows Credential Manager or DPAPI. On non-Windows targets it
/// is unavailable; on Windows, successful file operations supply runtime proof
/// for that provider instance.
#[derive(Default)]
pub struct WindowsCredentialProtectionProvider {
    runtime_verified: AtomicBool,
}

impl std::fmt::Debug for WindowsCredentialProtectionProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WindowsCredentialProtectionProvider")
            .field("capability", &self.capability())
            .finish()
    }
}

impl CredentialProvider for WindowsCredentialProtectionProvider {
    fn capability(&self) -> CredentialCapability {
        let compiled = cfg!(windows);
        capability(
            ProviderKind::WindowsProtectedFile,
            ProviderImplementation::CompileChecked,
            compiled,
            compiled,
            self.runtime_verified.load(Ordering::Acquire),
        )
    }

    fn load(&self) -> Result<Credentials, AbiError> {
        platform_file_load(&self.runtime_verified)
    }

    fn save(&self, credentials: &Credentials) -> Result<(), AbiError> {
        platform_file_save(&self.runtime_verified, credentials)
    }

    fn clear(&self) -> Result<(), AbiError> {
        self.save(&Credentials::default())
    }
}

#[cfg(windows)]
fn platform_file_load(evidence: &AtomicBool) -> Result<Credentials, AbiError> {
    verified(evidence, file::load_from_path(file::credentials_path()?))
}

#[cfg(not(windows))]
fn platform_file_load(_evidence: &AtomicBool) -> Result<Credentials, AbiError> {
    Err(AbiError::UnsupportedOperation)
}

#[cfg(windows)]
fn platform_file_save(evidence: &AtomicBool, credentials: &Credentials) -> Result<(), AbiError> {
    verified(
        evidence,
        file::save_to_path(file::credentials_path()?, credentials),
    )
}

#[cfg(not(windows))]
fn platform_file_save(_evidence: &AtomicBool, _credentials: &Credentials) -> Result<(), AbiError> {
    Err(AbiError::UnsupportedOperation)
}

/// Linux Secret Service adapter.
///
/// The implementation is target-gated and talks to the freedesktop Secret
/// Service over an encrypted in-process D-Bus session. It never invokes
/// `secret-tool` or places a secret on a process command line. Successful
/// operations provide runtime evidence only for this provider instance.
#[derive(Default)]
pub struct LinuxSecretServiceProvider {
    runtime_verified: AtomicBool,
}

impl std::fmt::Debug for LinuxSecretServiceProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LinuxSecretServiceProvider")
            .field("capability", &self.capability())
            .finish()
    }
}

impl CredentialProvider for LinuxSecretServiceProvider {
    fn capability(&self) -> CredentialCapability {
        let compiled = cfg!(target_os = "linux");
        capability(
            ProviderKind::LinuxSecretService,
            ProviderImplementation::Runtime,
            compiled,
            compiled,
            self.runtime_verified.load(Ordering::Acquire),
        )
    }

    fn load(&self) -> Result<Credentials, AbiError> {
        verified(&self.runtime_verified, linux_secret_service::load())
    }

    fn save(&self, credentials: &Credentials) -> Result<(), AbiError> {
        verified(
            &self.runtime_verified,
            linux_secret_service::save(credentials),
        )
    }

    fn clear(&self) -> Result<(), AbiError> {
        verified(&self.runtime_verified, linux_secret_service::clear())
    }
}

/// Platform provider inventory in stable macOS, Windows, Linux order.
#[must_use]
pub fn platform_credential_providers() -> Vec<Box<dyn CredentialProvider>> {
    vec![
        Box::<MacOsKeychainProvider>::default(),
        Box::<WindowsCredentialProtectionProvider>::default(),
        Box::<LinuxSecretServiceProvider>::default(),
    ]
}

/// Claim-honest platform capability records in stable provider order.
#[must_use]
pub fn platform_credential_capabilities() -> Vec<CredentialCapability> {
    platform_credential_providers()
        .iter()
        .map(|provider| provider.capability())
        .collect()
}

/// Stable, secret-free platform capability report.
#[must_use]
pub fn platform_credential_report() -> String {
    let mut report = platform_credential_capabilities()
        .into_iter()
        .map(CredentialCapability::status_line)
        .collect::<Vec<_>>()
        .join("\n");
    report.push('\n');
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::{CREDENTIALS_PATH_ENV, CredentialField, Secret};
    use crate::env;

    #[test]
    fn platform_capabilities_are_stable_and_claim_honest() {
        let capabilities = platform_credential_capabilities();
        assert_eq!(
            capabilities
                .iter()
                .map(|capability| capability.provider)
                .collect::<Vec<_>>(),
            [
                ProviderKind::MacOsKeychain,
                ProviderKind::WindowsProtectedFile,
                ProviderKind::LinuxSecretService,
            ]
        );
        assert_eq!(capabilities[0].compiled, cfg!(target_os = "macos"));
        assert_eq!(capabilities[0].available, cfg!(target_os = "macos"));
        assert_eq!(capabilities[1].compiled, cfg!(windows));
        assert_eq!(capabilities[1].available, cfg!(windows));
        assert_eq!(capabilities[2].compiled, cfg!(target_os = "linux"));
        assert_eq!(capabilities[2].available, cfg!(target_os = "linux"));
        assert!(
            capabilities
                .iter()
                .all(|capability| !capability.runtime_verified
                    && capability.secret_transport == SecretTransport::InProcessApi
                    && capability.scope == CredentialScope::AbiServiceCredentials)
        );
    }

    #[test]
    fn capability_report_is_stable_secret_free_and_complete() {
        let report = platform_credential_report();
        assert_eq!(report.lines().count(), 3);
        assert!(
            report
                .lines()
                .next()
                .unwrap()
                .starts_with("provider=macos_keychain implementation=runtime compiled=")
        );
        assert!(report.contains("provider=windows_protected_file implementation=compile_checked"));
        assert!(report.contains("provider=linux_secret_service implementation=runtime"));
        assert!(report.lines().all(|line| {
            line.contains("runtime_verified=false")
                && line.contains("secret_transport=in_process_api")
                && line.contains("scope=abi_service_credentials")
        }));
    }

    #[test]
    fn provider_trait_is_object_safe_and_debug_is_redacted() {
        let providers = platform_credential_providers();
        assert_eq!(providers.len(), 3);
        let mut credentials = Credentials::default();
        credentials.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-provider-do-not-print")),
        );
        let rendered = format!("{providers:?} {credentials:?}");
        assert!(!rendered.contains("provider-do-not-print"));
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn successful_file_operation_is_the_only_source_of_runtime_verification() {
        let _guard = env::lock_for_test();
        let path = crate::temp_path::temp_file_path("abi-provider-capability", "json");
        env::set_override(CREDENTIALS_PATH_ENV, &path.to_string_lossy());
        let provider = FileCredentialProvider::default();
        assert!(!provider.capability().runtime_verified);
        let mut credentials = Credentials::default();
        credentials.set(
            CredentialField::ANTHROPIC_API_KEY,
            Some(Secret::new("sk-runtime-proof")),
        );
        provider.save(&credentials).expect("save");
        assert!(provider.capability().runtime_verified);
        let loaded = provider.load().expect("load");
        assert_eq!(
            loaded
                .get(CredentialField::ANTHROPIC_API_KEY)
                .map(Secret::expose),
            Some("sk-runtime-proof")
        );
        std::fs::remove_file(path).ok();
        env::clear_override(CREDENTIALS_PATH_ENV);
    }

    #[test]
    fn failed_operation_does_not_invent_runtime_verification() {
        let evidence = AtomicBool::new(false);
        assert_eq!(
            verified::<()>(&evidence, Err(AbiError::InternalError)).unwrap_err(),
            AbiError::InternalError
        );
        assert!(!evidence.load(Ordering::Acquire));
    }

    #[test]
    fn unavailable_adapters_fail_without_inventing_runtime_evidence() {
        if !cfg!(target_os = "linux") {
            let linux = LinuxSecretServiceProvider::default();
            assert_eq!(linux.load().unwrap_err(), AbiError::UnsupportedOperation);
            assert_eq!(linux.clear().unwrap_err(), AbiError::UnsupportedOperation);
            assert!(!linux.capability().runtime_verified);
        }

        if !cfg!(windows) {
            let windows = WindowsCredentialProtectionProvider::default();
            assert_eq!(windows.load().unwrap_err(), AbiError::UnsupportedOperation);
            assert_eq!(
                windows.save(&Credentials::default()).unwrap_err(),
                AbiError::UnsupportedOperation
            );
            assert!(!windows.capability().runtime_verified);
        }
    }

    /// Opt-in real Linux Secret Service round trip.
    ///
    /// Set `ABI_LINUX_SECRET_SERVICE_TEST=1` on a Linux desktop session. The
    /// test refuses to overwrite a non-empty ABI Secret Service namespace.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_secret_service_roundtrip_sets_runtime_evidence() {
        if std::env::var_os("ABI_LINUX_SECRET_SERVICE_TEST").is_none() {
            return;
        }

        let provider = LinuxSecretServiceProvider::default();
        assert!(!provider.capability().runtime_verified);
        let existing = provider.load().expect("connect and inspect Secret Service");
        assert!(provider.capability().runtime_verified);
        if !existing.is_empty() {
            return;
        }

        let mut credentials = Credentials::default();
        credentials.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-linux-secret-service-test")),
        );
        provider.save(&credentials).expect("save");
        let loaded = provider.load().expect("load");
        assert_eq!(
            loaded
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-linux-secret-service-test")
        );
        provider.clear().expect("clear");
        assert!(provider.load().expect("load after clear").is_empty());
    }
}
