//! Hash type and format errors for the WDBX on-disk format.

use serde::Serialize;
use std::path::PathBuf;

/// Length of a block hash, in bytes.
pub const HASH_LEN: usize = 32;

/// Why reading the store failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// A segment or manifest did not start with its magic line.
    InvalidHeader {
        /// The file that was read.
        path: PathBuf,
        /// What the first line actually was.
        found: String,
    },
    /// A record's `type` was not one of the six known values.
    UnknownRecordType {
        /// The unrecognised type.
        found: String,
    },
    /// A record was missing a required field.
    MissingField {
        /// The record type.
        record: &'static str,
        /// The absent field.
        field: &'static str,
    },
    /// A field had the wrong JSON type or an out-of-range value.
    InvalidField {
        /// The record type.
        record: &'static str,
        /// The offending field.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
    /// A hash field was neither 32 integers nor a 32-character string.
    InvalidHash {
        /// The offending field, `hash` or `prev_hash`.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
    /// The manifest could not be parsed.
    InvalidManifest {
        /// Why it was rejected.
        reason: String,
    },
    /// A checksum trailer did not match the segment body.
    ChecksumMismatch {
        /// The segment that failed verification.
        path: PathBuf,
        /// Digest recorded in the trailer.
        expected: String,
        /// Digest computed from the body.
        actual: String,
    },
    /// A file could not be read or written.
    Io {
        /// The path that failed.
        path: PathBuf,
        /// The OS error text.
        message: String,
    },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader { path, found } => write!(
                f,
                "{} does not start with the expected magic line (found {found:?})",
                path.display()
            ),
            Self::UnknownRecordType { found } => write!(f, "unknown record type {found:?}"),
            Self::MissingField { record, field } => {
                write!(f, "{record} record is missing field {field:?}")
            }
            Self::InvalidField {
                record,
                field,
                reason,
            } => write!(f, "{record} record field {field:?} is invalid: {reason}"),
            Self::InvalidHash { field, reason } => {
                write!(f, "hash field {field:?} is invalid: {reason}")
            }
            Self::InvalidManifest { reason } => write!(f, "invalid manifest: {reason}"),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch for {}: expected {expected}, computed {actual}",
                path.display()
            ),
            Self::Io { path, message } => {
                write!(f, "I/O failed for {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for FormatError {}

/// Shorthand for a format result.
pub type Result<T> = std::result::Result<T, FormatError>;

/// A 32-byte block hash.
///
/// Serializes as a JSON array of integers — the encoding Zig produces for any
/// digest — and *deserializes* from either an array or a string, because Zig
/// emits the string form whenever the bytes happen to be valid UTF-8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash(pub [u8; HASH_LEN]);

impl Hash {
    /// The all-zero hash, used as the genesis block's predecessor.
    pub const GENESIS: Self = Self([0; HASH_LEN]);

    /// The bytes.
    #[must_use]
    pub const fn bytes(&self) -> &[u8; HASH_LEN] {
        &self.0
    }

    /// Whether this is the genesis sentinel.
    #[must_use]
    pub fn is_genesis(&self) -> bool {
        self.0 == [0; HASH_LEN]
    }

    /// Lowercase hex.
    #[must_use]
    pub fn to_hex(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(HASH_LEN * 2);
        for byte in self.0 {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    /// Read a hash from either JSON encoding.
    ///
    /// Accepts an array of 32 integers in `0..=255`, or a string of exactly 32
    /// `char`s each within `0..=255`. The string case is the one that trips up a
    /// from-scratch reader: Zig produced it for every all-zero `prev_hash`.
    pub fn from_json(value: &serde_json::Value, field: &'static str) -> Result<Self> {
        match value {
            serde_json::Value::Array(items) => {
                if items.len() != HASH_LEN {
                    return Err(FormatError::InvalidHash {
                        field,
                        reason: format!("expected {HASH_LEN} bytes, got {}", items.len()),
                    });
                }
                let mut bytes = [0u8; HASH_LEN];
                for (slot, item) in bytes.iter_mut().zip(items) {
                    let n = item.as_u64().ok_or_else(|| FormatError::InvalidHash {
                        field,
                        reason: format!("array element {item} is not an integer"),
                    })?;
                    *slot = u8::try_from(n).map_err(|_| FormatError::InvalidHash {
                        field,
                        reason: format!("array element {n} does not fit in a byte"),
                    })?;
                }
                Ok(Self(bytes))
            }
            serde_json::Value::String(text) => {
                // Zig wrote the raw bytes as a UTF-8 string, so each char is one
                // original byte and must be within 0..=255.
                let chars: Vec<char> = text.chars().collect();
                if chars.len() != HASH_LEN {
                    return Err(FormatError::InvalidHash {
                        field,
                        reason: format!("expected {HASH_LEN} characters, got {}", chars.len()),
                    });
                }
                let mut bytes = [0u8; HASH_LEN];
                for (slot, ch) in bytes.iter_mut().zip(chars) {
                    let code = ch as u32;
                    *slot = u8::try_from(code).map_err(|_| FormatError::InvalidHash {
                        field,
                        reason: format!("character U+{code:04X} is not a single byte"),
                    })?;
                }
                Ok(Self(bytes))
            }
            other => Err(FormatError::InvalidHash {
                field,
                reason: format!("expected an array or string, got {other}"),
            }),
        }
    }
}

impl Serialize for Hash {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        // Always the array form: it round-trips regardless of the byte values,
        // whereas the string form is only valid when the bytes are valid UTF-8.
        self.0.serialize(serializer)
    }
}
