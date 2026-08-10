//! Publisher-authenticated manifest envelopes.
//!
//! Signatures cover the exact UTF-8 bytes in `manifest_json`. This avoids an
//! implicit JSON canonicalization contract: whitespace is significant, and a
//! verifier parses the manifest only after its publisher signature succeeds.

use crate::{ModelError, ModelManifest};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Supported signed-envelope version.
pub const SIGNED_MANIFEST_VERSION: u32 = 1;

/// Configured publisher keys, indexed by stable key identifier.
#[derive(Debug, Clone, Default)]
pub struct PublisherTrustStore {
    keys: BTreeMap<String, VerifyingKey>,
}

impl PublisherTrustStore {
    /// Create an empty trust store. No publisher is trusted implicitly.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure one publisher key from 32 raw Ed25519 public-key bytes.
    pub fn insert(
        &mut self,
        key_id: impl Into<String>,
        key_bytes: [u8; 32],
    ) -> Result<(), ModelError> {
        let key_id = key_id.into();
        if key_id.trim().is_empty() {
            return Err(ModelError::MalformedSignatureMaterial {
                kind: "key id",
                key_id,
            });
        }
        if self.keys.contains_key(&key_id) {
            return Err(ModelError::DuplicatePublisherKey { key_id });
        }
        let key = VerifyingKey::from_bytes(&key_bytes).map_err(|_| {
            ModelError::MalformedSignatureMaterial {
                kind: "public key",
                key_id: key_id.clone(),
            }
        })?;
        self.keys.insert(key_id, key);
        Ok(())
    }

    /// Configure one publisher key from 64 lowercase hexadecimal characters.
    pub fn insert_hex(
        &mut self,
        key_id: impl Into<String>,
        public_key_hex: &str,
    ) -> Result<(), ModelError> {
        let key_id = key_id.into();
        let bytes = decode_hex::<32>(public_key_hex).ok_or_else(|| {
            ModelError::MalformedSignatureMaterial {
                kind: "public key",
                key_id: key_id.clone(),
            }
        })?;
        self.insert(key_id, bytes)
    }

    /// Number of configured publisher keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether no publisher keys are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// An exact manifest document and its publisher signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedManifestEnvelope {
    /// Envelope schema version.
    version: u32,
    /// Stable publisher key identifier.
    key_id: String,
    /// Exact UTF-8 manifest bytes covered by the signature.
    manifest_json: String,
    /// Ed25519 signature as 128 lowercase hexadecimal characters.
    signature: String,
}

impl SignedManifestEnvelope {
    /// Parse one envelope without trusting or parsing its enclosed manifest.
    pub fn from_json(context: &str, document: &str) -> Result<Self, ModelError> {
        serde_json::from_str(document).map_err(|source| ModelError::json(context, source))
    }

    /// Stable publisher key identifier declared by the envelope.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Exact signed manifest document.
    #[must_use]
    pub fn manifest_json(&self) -> &str {
        &self.manifest_json
    }

    /// Verify publisher identity, then parse and validate the manifest.
    pub fn verify(
        &self,
        context: &str,
        trust_store: &PublisherTrustStore,
    ) -> Result<ModelManifest, ModelError> {
        if self.version != SIGNED_MANIFEST_VERSION {
            return Err(ModelError::InvalidManifest {
                model: "<signed-envelope>".to_owned(),
                reason: format!("unsupported signed manifest version {}", self.version),
            });
        }
        let key = trust_store.keys.get(&self.key_id).ok_or_else(|| {
            ModelError::UntrustedPublisherKey {
                key_id: self.key_id.clone(),
            }
        })?;
        let bytes = decode_hex::<64>(&self.signature).ok_or_else(|| {
            ModelError::MalformedSignatureMaterial {
                kind: "signature",
                key_id: self.key_id.clone(),
            }
        })?;
        let signature = Signature::from_bytes(&bytes);
        key.verify_strict(self.manifest_json.as_bytes(), &signature)
            .map_err(|_| ModelError::InvalidManifestSignature {
                key_id: self.key_id.clone(),
            })?;
        ModelManifest::from_json(context, &self.manifest_json)
    }
}

/// Decode exactly `N` lowercase-hex bytes.
fn decode_hex<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (index, byte) in out.iter_mut().enumerate() {
        let pair = value.as_bytes().get(index * 2..index * 2 + 2)?;
        let high = nibble(pair[0])?;
        let low = nibble(pair[1])?;
        *byte = (high << 4) | low;
    }
    Some(out)
}

/// Decode one lowercase hexadecimal nibble.
const fn nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::manifest;
    use ed25519_dalek::{Signer as _, SigningKey};
    use serde_json::json;

    fn hex(bytes: &[u8]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            output.push(char::from(DIGITS[usize::from(byte >> 4)]));
            output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
        }
        output
    }

    fn signed_document(signing: &SigningKey, manifest_json: &str) -> String {
        let signature = signing.sign(manifest_json.as_bytes());
        json!({
            "version": SIGNED_MANIFEST_VERSION,
            "key_id": "publisher-1",
            "manifest_json": manifest_json,
            "signature": hex(&signature.to_bytes()),
        })
        .to_string()
    }

    fn configured_store(signing: &SigningKey) -> PublisherTrustStore {
        let mut store = PublisherTrustStore::new();
        store
            .insert("publisher-1", signing.verifying_key().to_bytes())
            .expect("valid key");
        store
    }

    #[test]
    fn a_configured_publisher_signature_authorizes_manifest_parsing() {
        let signing = SigningKey::from_bytes(&[7; 32]);
        let manifest_json = manifest().to_json().expect("serializes");
        let envelope = SignedManifestEnvelope::from_json(
            "envelope",
            &signed_document(&signing, &manifest_json),
        )
        .expect("parses");

        let verified = envelope
            .verify("manifest", &configured_store(&signing))
            .expect("signature and manifest validate");
        assert_eq!(verified, manifest());
    }

    #[test]
    fn tampering_with_exact_manifest_bytes_fails_closed() {
        let signing = SigningKey::from_bytes(&[8; 32]);
        let manifest_json = manifest().to_json().expect("serializes");
        let mut value: serde_json::Value =
            serde_json::from_str(&signed_document(&signing, &manifest_json)).expect("json");
        value["manifest_json"] = serde_json::Value::String(format!("{manifest_json} "));
        let envelope = SignedManifestEnvelope::from_json("envelope", &value.to_string())
            .expect("envelope shape remains valid");

        let error = envelope
            .verify("manifest", &configured_store(&signing))
            .expect_err("changed signed bytes must fail");
        assert!(matches!(error, ModelError::InvalidManifestSignature { .. }));
    }

    #[test]
    fn an_unknown_publisher_is_never_implicitly_trusted() {
        let signing = SigningKey::from_bytes(&[9; 32]);
        let manifest_json = manifest().to_json().expect("serializes");
        let envelope = SignedManifestEnvelope::from_json(
            "envelope",
            &signed_document(&signing, &manifest_json),
        )
        .expect("parses");

        let error = envelope
            .verify("manifest", &PublisherTrustStore::new())
            .expect_err("empty trust store must reject every publisher");
        assert!(matches!(error, ModelError::UntrustedPublisherKey { .. }));
    }

    #[test]
    fn a_signed_manifest_still_requires_artifact_hashes() {
        let signing = SigningKey::from_bytes(&[10; 32]);
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&manifest().to_json().expect("serializes")).expect("json");
        manifest_value["artifacts"][0]
            .as_object_mut()
            .expect("artifact")
            .remove("sha256");
        let manifest_json = manifest_value.to_string();
        let envelope = SignedManifestEnvelope::from_json(
            "envelope",
            &signed_document(&signing, &manifest_json),
        )
        .expect("parses");

        let error = envelope
            .verify("manifest", &configured_store(&signing))
            .expect_err("signature cannot replace mandatory content hashes");
        assert!(matches!(error, ModelError::MissingHash { .. }));
    }

    #[test]
    fn publisher_key_ids_are_nonblank_and_cannot_be_reconfigured() {
        let signing = SigningKey::from_bytes(&[11; 32]);
        let mut store = PublisherTrustStore::new();
        let error = store
            .insert(" ", signing.verifying_key().to_bytes())
            .expect_err("blank identifiers are ambiguous");
        assert!(matches!(
            error,
            ModelError::MalformedSignatureMaterial { .. }
        ));

        store
            .insert("publisher-1", signing.verifying_key().to_bytes())
            .expect("first configuration succeeds");
        let error = store
            .insert("publisher-1", signing.verifying_key().to_bytes())
            .expect_err("a key id cannot be silently replaced");
        assert!(matches!(error, ModelError::DuplicatePublisherKey { .. }));
    }
}
