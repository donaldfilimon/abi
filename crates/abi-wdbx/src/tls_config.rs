//! Disclosed-partial TLS path configuration for WDBX listeners.
//!
//! This module validates certificate and key paths for an external terminating
//! proxy or future native integration. It does **not** provide TLS termination.

use std::path::{Path, PathBuf};

/// Certificate path environment variable.
pub const TLS_CERT_ENV: &str = abi_foundation::env::WDBX_TLS_CERT;
/// Private-key path environment variable.
pub const TLS_KEY_ENV: &str = abi_foundation::env::WDBX_TLS_KEY;

/// Validated TLS material paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConfig {
    /// PEM certificate path.
    pub cert_path: PathBuf,
    /// PEM private-key path.
    pub key_path: PathBuf,
}

impl TlsConfig {
    /// Read the paired environment variables and require accessible paths.
    ///
    /// Returns `None` when TLS is unconfigured, partially configured, or
    /// either path is inaccessible. A successful value is configuration
    /// metadata only; the Rust WDBX listener remains loopback plain HTTP.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let cert = abi_foundation::env::get(TLS_CERT_ENV)?;
        let key = abi_foundation::env::get(TLS_KEY_ENV)?;
        Self::from_paths(cert, key)
    }

    /// Validate an explicit certificate/key pair.
    #[must_use]
    pub fn from_paths(cert_path: impl Into<PathBuf>, key_path: impl Into<PathBuf>) -> Option<Self> {
        let cert_path = cert_path.into();
        let key_path = key_path.into();
        if !accessible(&cert_path) || !accessible(&key_path) {
            return None;
        }
        Some(Self {
            cert_path,
            key_path,
        })
    }
}

fn accessible(path: &Path) -> bool {
    std::fs::File::open(path).is_ok()
}

#[cfg(test)]
mod tests {
    use super::{TLS_CERT_ENV, TLS_KEY_ENV, TlsConfig};

    struct Fixture {
        directory: std::path::PathBuf,
        cert: std::path::PathBuf,
        key: std::path::PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = abi_foundation::temp_path::temp_file_path("abi_tls_config", "fixture");
            std::fs::create_dir_all(&directory).expect("fixture directory");
            let cert = directory.join("cert.pem");
            let key = directory.join("key.pem");
            std::fs::write(&cert, "certificate").expect("certificate");
            std::fs::write(&key, "private-key").expect("private key");
            Self {
                directory,
                cert,
                key,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            abi_foundation::env::clear_override(TLS_CERT_ENV);
            abi_foundation::env::clear_override(TLS_KEY_ENV);
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[test]
    fn explicit_paths_require_both_files_to_be_accessible() {
        let fixture = Fixture::new();
        assert_eq!(
            TlsConfig::from_paths(&fixture.cert, &fixture.key),
            Some(TlsConfig {
                cert_path: fixture.cert.clone(),
                key_path: fixture.key.clone(),
            })
        );
        assert!(TlsConfig::from_paths(&fixture.cert, fixture.directory.join("missing")).is_none());
    }

    #[test]
    fn environment_requires_a_complete_pair() {
        let fixture = Fixture::new();
        abi_foundation::env::set_override(TLS_CERT_ENV, &fixture.cert.to_string_lossy());
        assert!(TlsConfig::from_env().is_none());

        abi_foundation::env::set_override(TLS_KEY_ENV, &fixture.key.to_string_lossy());
        assert!(TlsConfig::from_env().is_some());
    }
}
