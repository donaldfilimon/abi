//! Bearer-token loading and constant-time verification.

use crate::config::{GatewayError, open_validated_file};
use sha2::{Digest as _, Sha256};
use std::io::Read as _;
use std::path::Path;
use subtle::ConstantTimeEq as _;
use zeroize::Zeroize as _;

/// Owner-protected bearer token whose debug form never exposes bytes.
pub struct BearerToken([u8; 32]);

impl BearerToken {
    /// Read a nonempty token after symlink, ownership, and mode validation.
    pub fn load(path: &Path) -> Result<Self, GatewayError> {
        let file = open_validated_file(path, true, "bearer token")?;
        let mut bytes = Vec::new();
        file.take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| GatewayError::File {
                label: "bearer token".into(),
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        while bytes.last().is_some_and(u8::is_ascii_whitespace) {
            bytes.pop();
        }
        if bytes.is_empty() || bytes.len() > 64 * 1024 {
            bytes.zeroize();
            return Err(GatewayError::File {
                label: "bearer token".into(),
                path: path.to_path_buf(),
                message: "must contain between 1 and 65536 non-whitespace bytes".into(),
            });
        }
        let digest = Sha256::digest(&bytes).into();
        bytes.zeroize();
        Ok(Self(digest))
    }

    /// Hash every candidate and compare fixed-size digests in constant time.
    #[must_use]
    pub fn authorizes(&self, header: Option<&str>) -> bool {
        let candidate = header
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or_default();
        let digest: [u8; 32] = Sha256::digest(candidate.as_bytes()).into();
        digest.ct_eq(&self.0).into()
    }
}

impl std::fmt::Debug for BearerToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("BearerToken([REDACTED])")
    }
}

impl Drop for BearerToken {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
