//! Authenticated WDBX v2 object envelopes and permission-checked key files.

use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead as _, Generate as _, Key, KeyInit as _, Payload},
};
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

/// Environment variable naming the raw 32-byte `XChaCha20` encryption key file.
pub const ABI_WDBX_ENCRYPTION_KEY_FILE: &str = "ABI_WDBX_ENCRYPTION_KEY_FILE";
/// Environment variable naming the raw 32-byte Ed25519 signing-key file.
pub const ABI_WDBX_SIGNING_KEY_FILE: &str = "ABI_WDBX_SIGNING_KEY_FILE";
/// Environment variable naming the raw 32-byte Ed25519 verifying-key file.
pub const ABI_WDBX_VERIFY_KEY_FILE: &str = "ABI_WDBX_VERIFY_KEY_FILE";

const OBJECT_MAGIC: &[u8] = b"ABI-WDBX-OBJECT 2\n";
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_OBJECT_BYTES: usize = 128 * 1024 * 1024;

/// V2 immutable object family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    /// Immutable compacted segment.
    Segment,
    /// Committed per-writer journal object.
    Journal,
    /// Versioned index/codebook artifact.
    Artifact,
}

#[derive(Clone, Copy)]
enum KeyKind {
    Encryption,
    Signing,
    Verifying,
}

/// Generated or loaded key bytes. Debug output is always redacted.
pub struct KeyMaterial {
    bytes: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for KeyMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("KeyMaterial([REDACTED])")
    }
}

/// Optional encryption/signing/verification keys for one object operation.
#[derive(Clone)]
pub struct ObjectSecurity {
    encryption: Option<Zeroizing<[u8; 32]>>,
    signing: Option<SigningKey>,
    verifying: Option<VerifyingKey>,
}

impl std::fmt::Debug for ObjectSecurity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObjectSecurity")
            .field("encryption", &self.encryption.is_some())
            .field("signing", &self.signing.is_some())
            .field("verifying", &self.verifying.is_some())
            .finish()
    }
}

impl ObjectSecurity {
    /// Plaintext, unsigned compatibility mode.
    #[must_use]
    pub fn plaintext() -> Self {
        Self {
            encryption: None,
            signing: None,
            verifying: None,
        }
    }

    /// Load only explicitly configured key files from the process environment.
    pub fn from_env() -> Result<Self, SecurityError> {
        let encryption = env_key(ABI_WDBX_ENCRYPTION_KEY_FILE, KeyKind::Encryption)?
            .map(|material| material.bytes);
        let signing = env_key(ABI_WDBX_SIGNING_KEY_FILE, KeyKind::Signing)?
            .map(|material| SigningKey::from_bytes(&material.bytes));
        let configured_verifying = env_key(ABI_WDBX_VERIFY_KEY_FILE, KeyKind::Verifying)?
            .map(|material| {
                VerifyingKey::from_bytes(&material.bytes)
                    .map_err(|error| SecurityError::InvalidKey(error.to_string()))
            })
            .transpose()?;
        validate_signing_pair(signing.as_ref(), configured_verifying.as_ref())?;
        let verifying =
            configured_verifying.or_else(|| signing.as_ref().map(SigningKey::verifying_key));
        Ok(Self {
            encryption,
            signing,
            verifying,
        })
    }

    /// Load an explicit encryption/signing/verifying key-file set.
    pub fn from_key_files(
        encryption: Option<&Path>,
        signing: Option<&Path>,
        verifying: Option<&Path>,
    ) -> Result<Self, SecurityError> {
        let encryption = encryption
            .map(|path| read_key_file(path, KeyKind::Encryption))
            .transpose()?
            .map(|material| material.bytes);
        let signing = signing
            .map(|path| read_key_file(path, KeyKind::Signing))
            .transpose()?
            .map(|material| SigningKey::from_bytes(&material.bytes));
        let configured_verifying = verifying
            .map(|path| read_key_file(path, KeyKind::Verifying))
            .transpose()?
            .map(|material| {
                VerifyingKey::from_bytes(&material.bytes)
                    .map_err(|error| SecurityError::InvalidKey(error.to_string()))
            })
            .transpose()?;
        validate_signing_pair(signing.as_ref(), configured_verifying.as_ref())?;
        let verifying =
            configured_verifying.or_else(|| signing.as_ref().map(SigningKey::verifying_key));
        Ok(Self {
            encryption,
            signing,
            verifying,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_material(
        encryption: Option<&KeyMaterial>,
        signing: Option<&KeyMaterial>,
        verifying: Option<&KeyMaterial>,
    ) -> Result<Self, SecurityError> {
        let signing = signing.map(|key| SigningKey::from_bytes(&key.bytes));
        let configured_verifying = verifying
            .map(|key| {
                VerifyingKey::from_bytes(&key.bytes)
                    .map_err(|error| SecurityError::InvalidKey(error.to_string()))
            })
            .transpose()?;
        validate_signing_pair(signing.as_ref(), configured_verifying.as_ref())?;
        let verifying =
            configured_verifying.or_else(|| signing.as_ref().map(SigningKey::verifying_key));
        Ok(Self {
            encryption: encryption.map(|key| Zeroizing::new(*key.bytes)),
            signing,
            verifying,
        })
    }
}

fn validate_signing_pair(
    signing: Option<&SigningKey>,
    verifying: Option<&VerifyingKey>,
) -> Result<(), SecurityError> {
    if let (Some(signing), Some(verifying)) = (signing, verifying)
        && signing.verifying_key() != *verifying
    {
        return Err(SecurityError::InvalidKey(
            "signing and verifying key files do not form one Ed25519 pair".into(),
        ));
    }
    Ok(())
}

/// Decoded, authenticated object content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenedObject {
    /// Object family.
    pub kind: ObjectKind,
    /// Stable object identity supplied by the caller.
    pub object_id: String,
    /// Authenticated plaintext.
    pub plaintext: Vec<u8>,
    /// Whether the stored object was encrypted.
    pub encrypted: bool,
    /// Whether the stored object carried a verified signature.
    pub signed: bool,
}

/// Object envelope or key-file failure.
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    /// A key file or object could not be read or written.
    #[error("WDBX security I/O failed for {path}: {message}")]
    Io {
        /// Object path.
        path: PathBuf,
        /// Underlying detail.
        message: String,
    },
    /// A private key file was not owner-only.
    #[error("private key file {path} must be owner-only")]
    InsecurePermissions {
        /// Rejected key path.
        path: PathBuf,
    },
    /// Key bytes had the wrong size or did not decode.
    #[error("invalid WDBX key: {0}")]
    InvalidKey(String),
    /// Encrypted content was opened without its encryption key.
    #[error("encrypted WDBX object requires {ABI_WDBX_ENCRYPTION_KEY_FILE}")]
    MissingEncryptionKey,
    /// Signed content was opened without its verifying key.
    #[error("signed WDBX object requires {ABI_WDBX_VERIFY_KEY_FILE}")]
    MissingVerifyKey,
    /// Signature enforcement rejected an unsigned object.
    #[error("WDBX object is unsigned but a signature is required")]
    SignatureRequired,
    /// The envelope framing or authenticated header was malformed.
    #[error("invalid WDBX object envelope: {0}")]
    InvalidEnvelope(String),
    /// Ciphertext or signature authentication failed.
    #[error("WDBX object authentication failed")]
    Authentication,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct IdentityHeader {
    format_version: u8,
    kind: ObjectKind,
    object_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ObjectHeader {
    identity: IdentityHeader,
    encrypted: bool,
    signed: bool,
    nonce: Option<String>,
    ciphertext_sha256: String,
    plaintext_len: usize,
    payload_len: usize,
}

/// Generate independent encryption, signing, and verifying key material.
#[must_use]
pub fn generate_key_material() -> (KeyMaterial, KeyMaterial, KeyMaterial) {
    let encryption = Key::<XChaCha20Poly1305>::generate();
    let signing_seed = Key::<XChaCha20Poly1305>::generate();
    let signing = SigningKey::from_bytes(signing_seed.as_ref());
    let verifying = signing.verifying_key();
    (
        KeyMaterial {
            bytes: Zeroizing::new(encryption.into()),
        },
        KeyMaterial {
            bytes: Zeroizing::new(signing.to_bytes()),
        },
        KeyMaterial {
            bytes: Zeroizing::new(verifying.to_bytes()),
        },
    )
}

/// Create a new key file with owner-only permissions and no overwrite.
pub fn write_key_material(path: impl AsRef<Path>, key: &KeyMaterial) -> Result<(), SecurityError> {
    let path = path.as_ref();
    abi_foundation::credentials::write_owner_only_file_new(path, key.bytes.as_ref()).map_err(
        |error| SecurityError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        },
    )
}

/// Seal one immutable object with optional independent encryption and signing.
pub fn seal_object(
    kind: ObjectKind,
    object_id: &str,
    plaintext: &[u8],
    security: &ObjectSecurity,
) -> Result<Vec<u8>, SecurityError> {
    validate_identity(object_id, plaintext.len())?;
    let identity = IdentityHeader {
        format_version: 2,
        kind,
        object_id: object_id.to_owned(),
    };
    let identity_bytes = serde_json::to_vec(&identity)
        .map_err(|error| SecurityError::InvalidEnvelope(error.to_string()))?;
    let (payload, nonce) = if let Some(key) = security.encryption.as_ref() {
        let cipher = XChaCha20Poly1305::new_from_slice(key.as_ref())
            .map_err(|_| SecurityError::InvalidKey("encryption key must be 32 bytes".into()))?;
        let nonce = XNonce::generate();
        let payload = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad: &identity_bytes,
                },
            )
            .map_err(|_| SecurityError::Authentication)?;
        (payload, Some(hex_encode(nonce.as_ref())))
    } else {
        (plaintext.to_vec(), None)
    };
    let header = ObjectHeader {
        identity,
        encrypted: security.encryption.is_some(),
        signed: security.signing.is_some(),
        nonce,
        ciphertext_sha256: format!("{:x}", Sha256::digest(&payload)),
        plaintext_len: plaintext.len(),
        payload_len: payload.len(),
    };
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|error| SecurityError::InvalidEnvelope(error.to_string()))?;
    let signature = security
        .signing
        .as_ref()
        .map(|key| key.sign(&header_bytes).to_bytes().to_vec())
        .unwrap_or_default();
    encode_envelope(&header_bytes, &payload, &signature)
}

/// Verify, authenticate, and open one immutable object.
pub fn open_object(
    encoded: &[u8],
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<OpenedObject, SecurityError> {
    let (header_bytes, payload, signature_bytes) = decode_envelope(encoded)?;
    let header: ObjectHeader = serde_json::from_slice(header_bytes)
        .map_err(|error| SecurityError::InvalidEnvelope(error.to_string()))?;
    validate_header(&header, payload.len())?;
    if require_signature && !header.signed {
        return Err(SecurityError::SignatureRequired);
    }
    if header.signed {
        let verifying = security
            .verifying
            .as_ref()
            .ok_or(SecurityError::MissingVerifyKey)?;
        let bytes: [u8; 64] = signature_bytes
            .try_into()
            .map_err(|_| SecurityError::Authentication)?;
        let signature = Signature::from_bytes(&bytes);
        verifying
            .verify_strict(header_bytes, &signature)
            .map_err(|_| SecurityError::Authentication)?;
    } else if !signature_bytes.is_empty() {
        return Err(SecurityError::InvalidEnvelope(
            "unsigned header carries signature bytes".into(),
        ));
    }
    let found_hash = format!("{:x}", Sha256::digest(payload));
    if found_hash != header.ciphertext_sha256 {
        return Err(SecurityError::Authentication);
    }
    let identity_bytes = serde_json::to_vec(&header.identity)
        .map_err(|error| SecurityError::InvalidEnvelope(error.to_string()))?;
    let plaintext = if header.encrypted {
        let key = security
            .encryption
            .as_ref()
            .ok_or(SecurityError::MissingEncryptionKey)?;
        let nonce = header.nonce.as_deref().ok_or_else(|| {
            SecurityError::InvalidEnvelope("encrypted header has no nonce".into())
        })?;
        let nonce_bytes = hex_decode::<24>(nonce)?;
        let nonce = XNonce::try_from(nonce_bytes.as_slice())
            .map_err(|_| SecurityError::InvalidEnvelope("invalid nonce".into()))?;
        let cipher = XChaCha20Poly1305::new_from_slice(key.as_ref())
            .map_err(|_| SecurityError::InvalidKey("encryption key must be 32 bytes".into()))?;
        cipher
            .decrypt(
                &nonce,
                Payload {
                    msg: payload,
                    aad: &identity_bytes,
                },
            )
            .map_err(|_| SecurityError::Authentication)?
    } else {
        if header.nonce.is_some() {
            return Err(SecurityError::InvalidEnvelope(
                "plaintext header unexpectedly carries a nonce".into(),
            ));
        }
        payload.to_vec()
    };
    if plaintext.len() != header.plaintext_len {
        return Err(SecurityError::Authentication);
    }
    Ok(OpenedObject {
        kind: header.identity.kind,
        object_id: header.identity.object_id,
        plaintext,
        encrypted: header.encrypted,
        signed: header.signed,
    })
}

fn env_key(name: &str, kind: KeyKind) -> Result<Option<KeyMaterial>, SecurityError> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .map(|path| read_key_file(&path, kind))
        .transpose()
}

fn read_key_file(path: &Path, kind: KeyKind) -> Result<KeyMaterial, SecurityError> {
    let mut file = std::fs::File::open(path).map_err(|error| io_error(path, &error))?;
    let metadata = file.metadata().map_err(|error| io_error(path, &error))?;
    if !metadata.is_file() {
        return Err(SecurityError::InvalidKey(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if !matches!(kind, KeyKind::Verifying) && !private_permissions(path, &metadata) {
        return Err(SecurityError::InsecurePermissions {
            path: path.to_path_buf(),
        });
    }
    if metadata.len() != 32 {
        return Err(SecurityError::InvalidKey(format!(
            "{} must contain exactly 32 bytes",
            path.display()
        )));
    }
    let mut bytes = Zeroizing::new([0_u8; 32]);
    file.read_exact(bytes.as_mut())
        .map_err(|error| io_error(path, &error))?;
    Ok(KeyMaterial { bytes })
}

#[cfg(unix)]
fn private_permissions(_path: &Path, metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode().trailing_zeros() >= 6
}

#[cfg(windows)]
fn private_permissions(path: &Path, _metadata: &std::fs::Metadata) -> bool {
    abi_foundation::credentials::windows_file_is_owner_only(path).unwrap_or(false)
}

#[cfg(all(not(unix), not(windows)))]
fn private_permissions(_path: &Path, _metadata: &std::fs::Metadata) -> bool {
    false
}

fn encode_envelope(
    header: &[u8],
    payload: &[u8],
    signature: &[u8],
) -> Result<Vec<u8>, SecurityError> {
    let header_len = u32::try_from(header.len())
        .map_err(|_| SecurityError::InvalidEnvelope("header is too large".into()))?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| SecurityError::InvalidEnvelope("payload is too large".into()))?;
    let signature_len = u16::try_from(signature.len())
        .map_err(|_| SecurityError::InvalidEnvelope("signature is too large".into()))?;
    let mut encoded = Vec::with_capacity(
        OBJECT_MAGIC.len() + 4 + header.len() + 8 + payload.len() + 2 + signature.len(),
    );
    encoded.extend_from_slice(OBJECT_MAGIC);
    encoded.extend_from_slice(&header_len.to_be_bytes());
    encoded.extend_from_slice(header);
    encoded.extend_from_slice(&payload_len.to_be_bytes());
    encoded.extend_from_slice(payload);
    encoded.extend_from_slice(&signature_len.to_be_bytes());
    encoded.extend_from_slice(signature);
    Ok(encoded)
}

type EnvelopeParts<'a> = (&'a [u8], &'a [u8], &'a [u8]);

fn decode_envelope(encoded: &[u8]) -> Result<EnvelopeParts<'_>, SecurityError> {
    if encoded.len() > MAX_OBJECT_BYTES || !encoded.starts_with(OBJECT_MAGIC) {
        return Err(SecurityError::InvalidEnvelope(
            "bad magic or object exceeds the size limit".into(),
        ));
    }
    let mut offset = OBJECT_MAGIC.len();
    let header_len = usize::try_from(read_u32(encoded, &mut offset)?)
        .map_err(|_| SecurityError::InvalidEnvelope("header length overflow".into()))?;
    if header_len > MAX_HEADER_BYTES {
        return Err(SecurityError::InvalidEnvelope("header is too large".into()));
    }
    let header = take(encoded, &mut offset, header_len)?;
    let payload_len = usize::try_from(read_u64(encoded, &mut offset)?)
        .map_err(|_| SecurityError::InvalidEnvelope("payload length overflow".into()))?;
    let payload = take(encoded, &mut offset, payload_len)?;
    let signature_len = usize::from(read_u16(encoded, &mut offset)?);
    let signature = take(encoded, &mut offset, signature_len)?;
    if offset != encoded.len() {
        return Err(SecurityError::InvalidEnvelope(
            "trailing bytes after signature".into(),
        ));
    }
    Ok((header, payload, signature))
}

fn validate_identity(object_id: &str, plaintext_len: usize) -> Result<(), SecurityError> {
    if object_id.is_empty() || object_id.len() > 256 || object_id.chars().any(char::is_control) {
        return Err(SecurityError::InvalidEnvelope(
            "object id must be 1..=256 printable characters".into(),
        ));
    }
    if plaintext_len > MAX_OBJECT_BYTES {
        return Err(SecurityError::InvalidEnvelope(
            "plaintext exceeds the object size limit".into(),
        ));
    }
    Ok(())
}

fn validate_header(header: &ObjectHeader, payload_len: usize) -> Result<(), SecurityError> {
    if header.identity.format_version != 2 || header.payload_len != payload_len {
        return Err(SecurityError::InvalidEnvelope(
            "header version or payload length mismatch".into(),
        ));
    }
    validate_identity(&header.identity.object_id, header.plaintext_len)
}

fn take<'a>(
    encoded: &'a [u8],
    offset: &mut usize,
    length: usize,
) -> Result<&'a [u8], SecurityError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| SecurityError::InvalidEnvelope("length overflow".into()))?;
    let value = encoded
        .get(*offset..end)
        .ok_or_else(|| SecurityError::InvalidEnvelope("truncated object".into()))?;
    *offset = end;
    Ok(value)
}

fn read_u16(encoded: &[u8], offset: &mut usize) -> Result<u16, SecurityError> {
    let bytes: [u8; 2] = take(encoded, offset, 2)?
        .try_into()
        .expect("take returned exactly two bytes");
    Ok(u16::from_be_bytes(bytes))
}

fn read_u32(encoded: &[u8], offset: &mut usize) -> Result<u32, SecurityError> {
    let bytes: [u8; 4] = take(encoded, offset, 4)?
        .try_into()
        .expect("take returned exactly four bytes");
    Ok(u32::from_be_bytes(bytes))
}

fn read_u64(encoded: &[u8], offset: &mut usize) -> Result<u64, SecurityError> {
    let bytes: [u8; 8] = take(encoded, offset, 8)?
        .try_into()
        .expect("take returned exactly eight bytes");
    Ok(u64::from_be_bytes(bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

fn hex_decode<const N: usize>(value: &str) -> Result<[u8; N], SecurityError> {
    if value.len() != N * 2 {
        return Err(SecurityError::InvalidEnvelope(
            "invalid hexadecimal length".into(),
        ));
    }
    let mut bytes = [0_u8; N];
    for (index, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| SecurityError::InvalidEnvelope("invalid hexadecimal byte".into()))?;
    }
    Ok(bytes)
}

fn io_error(path: &Path, error: &std::io::Error) -> SecurityError {
    SecurityError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> (KeyMaterial, KeyMaterial, KeyMaterial) {
        generate_key_material()
    }

    #[test]
    fn plaintext_and_encrypted_signed_objects_round_trip() {
        let plain = ObjectSecurity::plaintext();
        let encoded = seal_object(ObjectKind::Journal, "writer-1", b"plain", &plain).unwrap();
        let opened = open_object(&encoded, &plain, false).unwrap();
        assert_eq!(opened.plaintext, b"plain");
        assert!(!opened.encrypted);
        assert!(!opened.signed);

        let (encryption, signing, verifying) = keys();
        let secure =
            ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying))
                .unwrap();
        let encoded = seal_object(ObjectKind::Segment, "segment-7", b"secret", &secure).unwrap();
        let opened = open_object(&encoded, &secure, true).unwrap();
        assert_eq!(opened.plaintext, b"secret");
        assert!(opened.encrypted);
        assert!(opened.signed);
    }

    #[test]
    fn mismatched_signing_and_verifying_keys_are_rejected_before_writes() {
        let (_, signing, _) = keys();
        let (_, _, unrelated_verifying) = keys();
        let error = ObjectSecurity::from_material(None, Some(&signing), Some(&unrelated_verifying))
            .expect_err("an inconsistent Ed25519 key pair must fail closed");
        assert!(error.to_string().contains("do not form one Ed25519 pair"));
    }

    #[test]
    fn every_encrypted_object_gets_a_fresh_24_byte_nonce() {
        let (encryption, _, _) = keys();
        let secure = ObjectSecurity::from_material(Some(&encryption), None, None).unwrap();
        let mut nonces = std::collections::BTreeSet::new();
        for index in 0..100 {
            let encoded = seal_object(
                ObjectKind::Journal,
                &format!("journal-{index}"),
                b"same plaintext",
                &secure,
            )
            .unwrap();
            let (header, _, _) = decode_envelope(&encoded).unwrap();
            let header: ObjectHeader = serde_json::from_slice(header).unwrap();
            let nonce = header.nonce.unwrap();
            assert_eq!(nonce.len(), 48);
            assert!(nonces.insert(nonce));
        }
    }

    #[test]
    fn wrong_missing_and_tampered_keys_or_objects_fail_closed() {
        let (encryption, signing, verifying) = keys();
        let secure =
            ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying))
                .unwrap();
        let encoded = seal_object(ObjectKind::Segment, "segment", b"secret", &secure).unwrap();
        let missing = ObjectSecurity::plaintext();
        assert!(matches!(
            open_object(&encoded, &missing, false),
            Err(SecurityError::MissingVerifyKey)
        ));

        let (wrong_encryption, _, wrong_verifying) = keys();
        let wrong =
            ObjectSecurity::from_material(Some(&wrong_encryption), None, Some(&wrong_verifying))
                .unwrap();
        assert!(matches!(
            open_object(&encoded, &wrong, false),
            Err(SecurityError::Authentication)
        ));

        for offset in [OBJECT_MAGIC.len() + 6, encoded.len() / 2, encoded.len() - 1] {
            let mut tampered = encoded.clone();
            tampered[offset] ^= 1;
            assert!(open_object(&tampered, &secure, true).is_err());
        }
    }

    #[test]
    fn signature_requirement_rejects_plaintext_compatibility_objects() {
        let plain = ObjectSecurity::plaintext();
        let encoded = seal_object(ObjectKind::Artifact, "plain", b"bytes", &plain).unwrap();
        assert!(matches!(
            open_object(&encoded, &plain, true),
            Err(SecurityError::SignatureRequired)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn private_key_files_must_be_owner_only() {
        use std::os::unix::fs::PermissionsExt as _;
        let path = abi_foundation::temp_path::temp_file_path("wdbx-v2-key", "bin");
        std::fs::write(&path, [7_u8; 32]).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            read_key_file(&path, KeyKind::Encryption),
            Err(SecurityError::InsecurePermissions { .. })
        ));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_key_file(&path, KeyKind::Encryption).is_ok());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn generated_key_files_are_redacted_owner_only_and_never_overwritten() {
        let (encryption, signing, verifying) = keys();
        let dir = abi_foundation::temp_path::temp_file_path("wdbx-v2-keys", "dir");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, key) in [
            ("encryption.key", &encryption),
            ("signing.key", &signing),
            ("verify.key", &verifying),
        ] {
            let path = dir.join(name);
            write_key_material(&path, key).unwrap();
            assert_eq!(std::fs::metadata(&path).unwrap().len(), 32);
            assert!(write_key_material(&path, key).is_err());
        }
        assert_eq!(format!("{encryption:?}"), "KeyMaterial([REDACTED])");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
