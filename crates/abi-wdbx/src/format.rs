//! The WDBX v1 on-disk format: records, segments, and the manifest.
//!
//! Ported from `src/features/wdbx/persistence.zig`, `persistence_parse.zig` and
//! `segments.zig`. Specified in `tests/golden/wdbx-format.md`, which was derived
//! from a census of the live store rather than from the source alone.
//!
//! **This must read existing data.** The store at `~/.abi/` holds ~300 segments
//! and ~180 MB of the user's real completions and embeddings; a
//! format-incompatible reader silently orphans all of it. Every tolerance below
//! is there because the real data or the Zig serializer requires it, not for
//! defensiveness:
//!
//! - A segment may contain only its magic line. Several real ones do.
//! - The final line may be torn by an interrupted write, and must be dropped
//!   rather than failing the whole segment.
//! - `hash` and `prev_hash` are each either a JSON array of 32 integers **or** a
//!   32-character string. Both are `[32]u8` in Zig and written identically, but
//!   Zig's stringify emits a byte array as a string when the bytes are valid
//!   UTF-8. A digest never is; an all-zero genesis `prev_hash` always is. So a
//!   reader that fixes one encoding per field fails on the genesis block of every
//!   segment — 40 of them in the observed store.
//! - All six record types must be handled, though the live store contains three.
//! - An optional `# checksum:<hex>` trailer may follow the records.

use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Magic first line of a segment file.
pub const SEGMENT_HEADER: &str = "# ABI-WDBX v1";

/// Magic first line of the manifest.
pub const MANIFEST_HEADER: &str = "# ABI-WDBX-SEGMENTS v1";

/// Magic first line of the compatibility-mirror epoch sidecar.
pub const MIRROR_EPOCH_HEADER: &str = "# ABI-WDBX-MIRROR-EPOCH v1";

/// Prefix of the optional checksum trailer line.
pub const CHECKSUM_PREFIX: &str = "# checksum:";

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

/// A key/value entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRecord {
    /// Entry key.
    pub key: String,
    /// Entry value, an **opaque** string.
    ///
    /// Frequently holds JSON encoded as a string, so it is double-encoded.
    /// Parsing it here would fail on the values that are not JSON.
    pub value: String,
}

/// A stored vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord {
    /// Vector id.
    pub id: u64,
    /// Components. Dimensionality is a property of the data, not the format.
    pub values: Vec<f32>,
}

/// An audit-chain block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRecord {
    /// This block's hash.
    pub hash: Hash,
    /// The predecessor's hash, or [`Hash::GENESIS`].
    pub prev_hash: Hash,
    /// Unix milliseconds.
    pub timestamp_ms: i64,
    /// Position in the chain.
    pub sequence: u64,
    /// The AI profile that produced the entry.
    pub profile: String,
    /// Query vector id.
    pub query_id: u64,
    /// Response vector id.
    pub response_id: u64,
    /// Opaque metadata string.
    pub metadata: String,
}

/// A 3-D spatial record.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialRecord {
    /// Record id.
    pub id: u64,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
    /// Opaque payload.
    pub payload: String,
}

/// A temporal-graph node or edge.
///
/// The live store contains none, but the Zig serializer emits them and its parser
/// accepts them, so a segment written by a build that used the temporal graph must
/// still load. The payload is kept as raw JSON rather than modelled, because there
/// is no sample to validate a stricter shape against — inventing one risks
/// rejecting real data.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalRecord {
    /// Whether this is a node or an edge.
    pub kind: TemporalKind,
    /// The record's fields, verbatim.
    pub fields: serde_json::Map<String, serde_json::Value>,
}

/// Which kind of temporal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalKind {
    /// A graph node.
    Node,
    /// A graph edge.
    Edge,
}

/// One record from a segment.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    /// A key/value entry.
    Kv(KvRecord),
    /// A vector.
    Vector(VectorRecord),
    /// An audit-chain block.
    Block(BlockRecord),
    /// A spatial record.
    Spatial(SpatialRecord),
    /// A temporal node or edge.
    Temporal(TemporalRecord),
}

impl Record {
    /// The `type` discriminator this record serializes with.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Kv(_) => "kv",
            Self::Vector(_) => "vector",
            Self::Block(_) => "block",
            Self::Spatial(_) => "spatial",
            Self::Temporal(record) => match record.kind {
                TemporalKind::Node => "temporal_node",
                TemporalKind::Edge => "temporal_edge",
            },
        }
    }

    /// Parse one JSONL record.
    pub fn parse(line: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| FormatError::InvalidField {
                record: "record",
                field: "<line>",
                reason: e.to_string(),
            })?;
        let object = value.as_object().ok_or_else(|| FormatError::InvalidField {
            record: "record",
            field: "<line>",
            reason: "not a JSON object".to_string(),
        })?;

        let type_name = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or(FormatError::MissingField {
                record: "record",
                field: "type",
            })?;

        match type_name {
            "kv" => Ok(Self::Kv(KvRecord {
                key: string_field(object, "kv", "key")?,
                value: string_field(object, "kv", "value")?,
            })),
            "vector" => {
                let items = object
                    .get("values")
                    .and_then(serde_json::Value::as_array)
                    .ok_or(FormatError::MissingField {
                        record: "vector",
                        field: "values",
                    })?;
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    let n = item.as_f64().ok_or_else(|| FormatError::InvalidField {
                        record: "vector",
                        field: "values",
                        reason: format!("element {item} is not a number"),
                    })?;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "the format stores f32; f64 is only the JSON transport type"
                    )]
                    values.push(n as f32);
                }
                Ok(Self::Vector(VectorRecord {
                    id: u64_field(object, "vector", "id")?,
                    values,
                }))
            }
            "block" => Ok(Self::Block(BlockRecord {
                hash: Hash::from_json(
                    object.get("hash").ok_or(FormatError::MissingField {
                        record: "block",
                        field: "hash",
                    })?,
                    "hash",
                )?,
                prev_hash: Hash::from_json(
                    object.get("prev_hash").ok_or(FormatError::MissingField {
                        record: "block",
                        field: "prev_hash",
                    })?,
                    "prev_hash",
                )?,
                timestamp_ms: i64_field(object, "block", "timestamp_ms")?,
                sequence: u64_field(object, "block", "sequence")?,
                profile: string_field(object, "block", "profile")?,
                query_id: u64_field(object, "block", "query_id")?,
                response_id: u64_field(object, "block", "response_id")?,
                metadata: string_field(object, "block", "metadata")?,
            })),
            "spatial" => Ok(Self::Spatial(SpatialRecord {
                id: u64_field(object, "spatial", "id")?,
                x: f32_field(object, "spatial", "x")?,
                y: f32_field(object, "spatial", "y")?,
                z: f32_field(object, "spatial", "z")?,
                payload: string_field(object, "spatial", "payload")?,
            })),
            "temporal_node" | "temporal_edge" => Ok(Self::Temporal(TemporalRecord {
                kind: if type_name == "temporal_node" {
                    TemporalKind::Node
                } else {
                    TemporalKind::Edge
                },
                fields: object.clone(),
            })),
            other => Err(FormatError::UnknownRecordType {
                found: other.to_string(),
            }),
        }
    }
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<String> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a string".to_string(),
        })
}

fn u64_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<u64> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_u64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a non-negative integer".to_string(),
        })
}

fn i64_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<i64> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_i64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not an integer".to_string(),
        })
}

fn f32_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<f32> {
    let n = object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_f64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a number".to_string(),
        })?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the format stores f32; f64 is only the JSON transport type"
    )]
    Ok(n as f32)
}

/// The outcome of reading one segment file.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// The epoch this segment represents.
    pub epoch: u64,
    /// Records, in file order.
    pub records: Vec<Record>,
    /// The checksum from a `# checksum:` trailer, if present.
    pub checksum: Option<String>,
    /// Whether the final line was truncated and therefore dropped.
    ///
    /// Surfaced rather than swallowed so `wdbx_stats` and recovery can report that
    /// a write was interrupted.
    pub truncated_tail: bool,
}

/// Parse a segment body.
///
/// `path` is used only for error messages.
pub fn parse_segment(epoch: u64, path: &Path, content: &str) -> Result<Segment> {
    let mut lines = content.lines();

    let header = lines.next().unwrap_or_default();
    if header != SEGMENT_HEADER {
        return Err(FormatError::InvalidHeader {
            path: path.to_path_buf(),
            found: header.to_string(),
        });
    }

    verify_checksum(path, content)?;

    // A trailing newline means the last record line is complete. Without one, the
    // final line may be a torn write.
    let ends_cleanly = content.ends_with('\n');
    let body: Vec<&str> = lines.collect();

    let mut records = Vec::new();
    let mut checksum = None;
    let mut truncated_tail = false;

    for (index, line) in body.iter().enumerate() {
        let is_last = index + 1 == body.len();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix(CHECKSUM_PREFIX) {
            checksum = Some(rest.trim().to_string());
            continue;
        }
        match Record::parse(line) {
            Ok(record) => records.push(record),
            Err(err) => {
                // Only the final line of a file with no trailing newline may be
                // torn. A parse failure anywhere else is real corruption and must
                // not be silently dropped.
                if is_last && !ends_cleanly {
                    truncated_tail = true;
                } else {
                    return Err(err);
                }
            }
        }
    }

    Ok(Segment {
        epoch,
        records,
        checksum,
        truncated_tail,
    })
}

fn verify_checksum(path: &Path, content: &str) -> Result<()> {
    let marker = format!("\n{CHECKSUM_PREFIX}");
    let Some(marker_start) = content.rfind(&marker) else {
        return Ok(());
    };

    let expected_start = marker_start + marker.len();
    let expected = content[expected_start..].trim();
    let body_start = SEGMENT_HEADER.len() + 1;
    let body_end = marker_start + 1;
    let actual = hex_digest(Sha256::digest(&content.as_bytes()[body_start..body_end]));

    if expected != actual {
        return Err(FormatError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    use std::fmt::Write as _;
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Read and parse a segment file.
pub fn read_segment(epoch: u64, path: &Path) -> Result<Segment> {
    let content = std::fs::read_to_string(path).map_err(|e| FormatError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    parse_segment(epoch, path, &content)
}

/// The segment manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The epoch the next new segment will take.
    pub next_epoch: u64,
    /// Live epochs, ascending and deduplicated.
    ///
    /// A segment file whose epoch is absent is garbage awaiting collection and
    /// must not be read.
    pub active: Vec<u64>,
}

impl Manifest {
    /// An empty manifest.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            next_epoch: 0,
            active: Vec::new(),
        }
    }

    /// Parse manifest text.
    pub fn parse(content: &str) -> Result<Self> {
        let mut lines = content.lines();
        let header = lines.next().unwrap_or_default();
        if header != MANIFEST_HEADER {
            return Err(FormatError::InvalidManifest {
                reason: format!("expected {MANIFEST_HEADER:?}, found {header:?}"),
            });
        }

        let mut next_epoch = None;
        let mut active = BTreeSet::new();

        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(FormatError::InvalidManifest {
                    reason: format!("line {line:?} is not key=value"),
                });
            };
            match key.trim() {
                "next_epoch" => {
                    next_epoch = Some(value.trim().parse::<u64>().map_err(|_| {
                        FormatError::InvalidManifest {
                            reason: format!("next_epoch {value:?} is not a number"),
                        }
                    })?);
                }
                "active" => {
                    for token in value.split(',') {
                        let token = token.trim();
                        if token.is_empty() {
                            continue;
                        }
                        active.insert(token.parse::<u64>().map_err(|_| {
                            FormatError::InvalidManifest {
                                reason: format!("active epoch {token:?} is not a number"),
                            }
                        })?);
                    }
                }
                // Forward compatibility: a newer writer may add keys, and
                // refusing to open the store over one would be worse than
                // ignoring it.
                _ => {}
            }
        }

        let active: Vec<u64> = active.into_iter().collect();
        Ok(Self {
            next_epoch: next_epoch.unwrap_or_else(|| active.last().map_or(0, |last| last + 1)),
            active,
        })
    }

    /// Render manifest text.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "{MANIFEST_HEADER}");
        let _ = writeln!(out, "next_epoch={}", self.next_epoch);
        let joined: Vec<String> = self.active.iter().map(u64::to_string).collect();
        let _ = writeln!(out, "active={}", joined.join(","));
        out
    }
}

/// Paths within a WDBX store directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePaths {
    /// The store directory.
    pub dir: PathBuf,
    /// Base name of the store files, `wdbx` by default.
    pub base: String,
}

impl StorePaths {
    /// Paths for `dir` using the default `wdbx` base name.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            base: "wdbx".to_string(),
        }
    }

    /// The binary index file.
    #[must_use]
    pub fn index(&self) -> PathBuf {
        self.dir.join(&self.base)
    }

    /// The manifest file.
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        self.dir.join(format!("{}.manifest", self.base))
    }

    /// Epoch provenance for the monolithic compatibility mirror.
    #[must_use]
    pub fn mirror_epoch(&self) -> PathBuf {
        self.dir.join(format!("{}.mirror-epoch", self.base))
    }

    /// The segment file for `epoch`.
    #[must_use]
    pub fn segment(&self, epoch: u64) -> PathBuf {
        self.dir.join(format!("{}.seg.{epoch}.jsonl", self.base))
    }

    /// Read the manifest, or an empty one if the store does not exist yet.
    ///
    /// A missing manifest is "no store yet", not an error — that is what a first
    /// run looks like.
    pub fn read_manifest(&self) -> Result<Manifest> {
        let path = self.manifest();
        match std::fs::read_to_string(&path) {
            Ok(content) => Manifest::parse(&content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Manifest::empty()),
            Err(e) => Err(FormatError::Io {
                path,
                message: e.to_string(),
            }),
        }
    }

    /// Read compatibility-mirror epoch provenance.
    ///
    /// A missing sidecar means epoch 0, which preserves readability of every
    /// Zig-written monolithic snapshot.
    pub fn read_mirror_epoch(&self) -> Result<u64> {
        let path = self.mirror_epoch();
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => {
                return Err(FormatError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let mut lines = content.lines();
        if lines.next() != Some(MIRROR_EPOCH_HEADER) {
            return Err(FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has an invalid header",
                    path.display()
                ),
            });
        }
        let epoch = lines
            .next()
            .and_then(|line| line.strip_prefix("epoch="))
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has an invalid epoch",
                    path.display()
                ),
            })?;
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has unexpected content",
                    path.display()
                ),
            });
        }
        Ok(epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_hash_is_all_zero_and_reports_itself() {
        assert!(Hash::GENESIS.is_genesis());
        assert_eq!(Hash::GENESIS.bytes(), &[0u8; HASH_LEN]);
        assert_eq!(Hash::GENESIS.to_hex(), "0".repeat(64));
        assert!(!Hash([1; HASH_LEN]).is_genesis());
    }

    #[test]
    fn hash_reads_the_array_encoding() {
        let value = serde_json::json!((0..32).collect::<Vec<u8>>());
        let hash = Hash::from_json(&value, "hash").expect("array form");
        assert_eq!(hash.bytes()[0], 0);
        assert_eq!(hash.bytes()[31], 31);
    }

    #[test]
    fn hash_reads_the_string_encoding_used_by_genesis_blocks() {
        // The case a from-scratch reader misses. Zig emits this for every
        // all-zero prev_hash because zero bytes are valid UTF-8.
        let zeros: String = std::iter::repeat_n('\u{0}', HASH_LEN).collect();
        let hash =
            Hash::from_json(&serde_json::Value::String(zeros), "prev_hash").expect("string form");
        assert_eq!(hash, Hash::GENESIS);
    }

    #[test]
    fn hash_reads_a_non_zero_string_encoding() {
        // Any byte sequence that is valid UTF-8 takes the string form.
        let text: String = (1u8..=32).map(char::from).collect();
        let hash = Hash::from_json(&serde_json::Value::String(text), "hash").expect("string form");
        assert_eq!(hash.bytes()[0], 1);
        assert_eq!(hash.bytes()[31], 32);
    }

    #[test]
    fn hash_rejects_the_wrong_length_in_either_encoding() {
        let short_array = serde_json::json!([1, 2, 3]);
        assert!(matches!(
            Hash::from_json(&short_array, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
        let short_string = serde_json::Value::String("abc".to_string());
        assert!(matches!(
            Hash::from_json(&short_string, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
    }

    #[test]
    fn hash_rejects_out_of_range_values() {
        let too_big = serde_json::json!((0..32).map(|_| 300).collect::<Vec<u32>>());
        assert!(matches!(
            Hash::from_json(&too_big, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
        // A codepoint above U+00FF cannot have come from one original byte.
        let wide: String = std::iter::repeat_n('世', HASH_LEN).collect();
        assert!(matches!(
            Hash::from_json(&serde_json::Value::String(wide), "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
    }

    #[test]
    fn latin1_range_characters_decode_to_their_byte_values() {
        // Not an error case, despite looking like one: Zig wrote raw byte 0xE9 as
        // the two-byte UTF-8 encoding of U+00E9, so reading that char back as 233
        // is the correct round-trip. Every byte in 0x80..=0xFF behaves this way.
        let text: String = std::iter::repeat_n('é', HASH_LEN).collect();
        let hash = Hash::from_json(&serde_json::Value::String(text), "prev_hash")
            .expect("U+00E9 is byte 233");
        assert_eq!(hash.bytes(), &[233u8; HASH_LEN]);
    }

    #[test]
    fn hash_serializes_as_an_array_so_it_always_round_trips() {
        // The string form only works for UTF-8-valid bytes; the array form
        // always works, so writing should use it unconditionally.
        let hash = Hash([200; HASH_LEN]);
        let text = serde_json::to_string(&hash).expect("serialize");
        assert!(text.starts_with('['));
        let back = Hash::from_json(&serde_json::from_str(&text).expect("parse"), "hash")
            .expect("round trip");
        assert_eq!(back, hash);
    }

    #[test]
    fn parses_a_kv_record_keeping_the_value_opaque() {
        let line = r#"{"type":"kv","key":"completion:1","value":"{\"kind\":\"completion\"}"}"#;
        let Record::Kv(record) = Record::parse(line).expect("parses") else {
            panic!("expected a kv record");
        };
        assert_eq!(record.key, "completion:1");
        // Double-encoded: still a string, not parsed.
        assert_eq!(record.value, r#"{"kind":"completion"}"#);
    }

    #[test]
    fn parses_a_kv_record_whose_value_is_not_json() {
        let line = r#"{"type":"kv","key":"k","value":"not json at all"}"#;
        let Record::Kv(record) = Record::parse(line).expect("parses") else {
            panic!("expected a kv record");
        };
        assert_eq!(record.value, "not json at all");
    }

    #[test]
    fn parses_a_vector_record_of_any_dimension() {
        for dims in [1usize, 3, 32, 128] {
            let values: Vec<f32> = (0..dims)
                .map(|i| f32::from(u8::try_from(i).unwrap_or(0)))
                .collect();
            let line = serde_json::json!({"type":"vector","id":7,"values":values}).to_string();
            let Record::Vector(record) = Record::parse(&line).expect("parses") else {
                panic!("expected a vector record");
            };
            assert_eq!(record.id, 7);
            assert_eq!(record.values.len(), dims);
        }
    }

    #[test]
    fn parses_a_block_with_either_hash_encoding() {
        let zeros: String = std::iter::repeat_n('\u{0}', HASH_LEN).collect();
        let line = serde_json::json!({
            "type":"block",
            "hash": (0..32).collect::<Vec<u8>>(),
            "prev_hash": zeros,
            "timestamp_ms": 1_753_000_000_000_i64,
            "sequence": 0,
            "profile": "abi",
            "query_id": 1,
            "response_id": 2,
            "metadata": "m"
        })
        .to_string();
        let Record::Block(block) = Record::parse(&line).expect("parses") else {
            panic!("expected a block record");
        };
        assert!(block.prev_hash.is_genesis());
        assert_eq!(block.timestamp_ms, 1_753_000_000_000);
        assert_eq!(block.profile, "abi");
    }

    #[test]
    fn parses_spatial_and_temporal_records() {
        let spatial = r#"{"type":"spatial","id":7,"x":1.5,"y":-2.5,"z":0.0,"payload":"p"}"#;
        let Record::Spatial(record) = Record::parse(spatial).expect("parses") else {
            panic!("expected a spatial record");
        };
        assert_eq!(record.id, 7);
        assert!((record.x - 1.5).abs() < f32::EPSILON);

        for (line, kind) in [
            (r#"{"type":"temporal_node","id":1}"#, TemporalKind::Node),
            (
                r#"{"type":"temporal_edge","from":1,"to":2}"#,
                TemporalKind::Edge,
            ),
        ] {
            let Record::Temporal(record) = Record::parse(line).expect("parses") else {
                panic!("expected a temporal record");
            };
            assert_eq!(record.kind, kind);
        }
    }

    #[test]
    fn rejects_an_unknown_record_type() {
        let err = Record::parse(r#"{"type":"nope"}"#).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnknownRecordType {
                found: "nope".to_string()
            }
        );
    }

    #[test]
    fn reports_a_missing_field_by_name() {
        let err = Record::parse(r#"{"type":"kv","key":"k"}"#).unwrap_err();
        assert_eq!(
            err,
            FormatError::MissingField {
                record: "kv",
                field: "value"
            }
        );
    }

    #[test]
    fn record_type_names_round_trip() {
        assert_eq!(
            Record::Kv(KvRecord {
                key: String::new(),
                value: String::new()
            })
            .type_name(),
            "kv"
        );
        assert_eq!(
            Record::Temporal(TemporalRecord {
                kind: TemporalKind::Edge,
                fields: serde_json::Map::new()
            })
            .type_name(),
            "temporal_edge"
        );
    }

    fn segment_path() -> PathBuf {
        PathBuf::from("wdbx.seg.0.jsonl")
    }

    #[test]
    fn parses_a_segment_containing_only_its_magic_line() {
        // Several real segments look exactly like this.
        let segment = parse_segment(3, &segment_path(), "# ABI-WDBX v1\n").expect("parses");
        assert!(segment.records.is_empty());
        assert!(!segment.truncated_tail);
        assert_eq!(segment.epoch, 3);
    }

    #[test]
    fn rejects_a_segment_with_a_bad_magic_line() {
        let err = parse_segment(0, &segment_path(), "# WRONG\n{}\n").unwrap_err();
        assert!(matches!(err, FormatError::InvalidHeader { .. }));
    }

    #[test]
    fn rejects_an_empty_segment_file() {
        // No magic line at all is corruption, not emptiness.
        assert!(matches!(
            parse_segment(0, &segment_path(), "").unwrap_err(),
            FormatError::InvalidHeader { .. }
        ));
    }

    #[test]
    fn drops_a_torn_final_line_and_reports_it() {
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}\n",
            "{\"type\":\"kv\",\"key\":\"b\",\"valu"
        );
        let segment = parse_segment(0, &segment_path(), content).expect("parses");
        assert_eq!(segment.records.len(), 1);
        assert!(
            segment.truncated_tail,
            "an interrupted write must be reported, not silently swallowed"
        );
    }

    #[test]
    fn corruption_that_is_not_the_final_line_is_an_error() {
        // The distinction that keeps "tolerate truncation" from becoming
        // "ignore all corruption".
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"valu\n",
            "{\"type\":\"kv\",\"key\":\"b\",\"value\":\"2\"}\n"
        );
        assert!(parse_segment(0, &segment_path(), content).is_err());
    }

    #[test]
    fn a_complete_final_line_is_not_treated_as_torn() {
        // No trailing newline, but the line parses — so it is complete.
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}"
        );
        let segment = parse_segment(0, &segment_path(), content).expect("parses");
        assert_eq!(segment.records.len(), 1);
        assert!(!segment.truncated_tail);
    }

    #[test]
    fn reads_the_optional_checksum_trailer() {
        let body = "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}\n";
        let checksum = hex_digest(Sha256::digest(body.as_bytes()));
        let content = format!("{SEGMENT_HEADER}\n{body}{CHECKSUM_PREFIX}{checksum}\n");
        let segment = parse_segment(0, &segment_path(), &content).expect("parses");
        assert_eq!(segment.checksum.as_deref(), Some(checksum.as_str()));
        assert_eq!(segment.records.len(), 1);
    }

    #[test]
    fn parses_a_manifest() {
        let manifest =
            Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=301\nactive=0,1,2,299\n")
                .expect("parses");
        assert_eq!(manifest.next_epoch, 301);
        assert_eq!(manifest.active, [0, 1, 2, 299]);
    }

    #[test]
    fn manifest_active_is_sorted_and_deduplicated() {
        // Ascending order is what makes epoch replay produce the right shadowing.
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=5\nactive=3,1,2,1\n")
            .expect("parses");
        assert_eq!(manifest.active, [1, 2, 3]);
    }

    #[test]
    fn manifest_tolerates_a_sparse_active_list() {
        // Compaction leaves gaps; assuming density would read collected segments.
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=10\nactive=0,4,9\n")
            .expect("parses");
        assert_eq!(manifest.active, [0, 4, 9]);
    }

    #[test]
    fn manifest_rejects_a_bad_magic_line() {
        assert!(matches!(
            Manifest::parse("# WRONG\nnext_epoch=1\n").unwrap_err(),
            FormatError::InvalidManifest { .. }
        ));
    }

    #[test]
    fn manifest_ignores_unknown_keys_for_forward_compatibility() {
        // Refusing to open the store because a newer writer added a key would be
        // worse than ignoring it.
        let manifest = Manifest::parse(
            "# ABI-WDBX-SEGMENTS v1\nnext_epoch=2\nactive=0,1\nfuture_key=whatever\n",
        )
        .expect("parses");
        assert_eq!(manifest.next_epoch, 2);
        assert_eq!(manifest.active, [0, 1]);
    }

    #[test]
    fn manifest_infers_next_epoch_when_absent() {
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nactive=0,1,2\n").expect("parses");
        assert_eq!(manifest.next_epoch, 3);
    }

    #[test]
    fn manifest_renders_in_the_readable_form() {
        let manifest = Manifest {
            next_epoch: 4,
            active: vec![0, 2, 3],
        };
        assert_eq!(
            manifest.render(),
            "# ABI-WDBX-SEGMENTS v1\nnext_epoch=4\nactive=0,2,3\n"
        );
        assert_eq!(
            Manifest::parse(&manifest.render()).expect("parses"),
            manifest
        );
    }

    #[test]
    fn empty_manifest_renders_and_reparses() {
        let manifest = Manifest::empty();
        assert_eq!(
            Manifest::parse(&manifest.render()).expect("parses"),
            manifest
        );
    }

    #[test]
    fn store_paths_match_the_observed_layout() {
        let paths = StorePaths::new("/home/u/.abi");
        assert_eq!(paths.index(), PathBuf::from("/home/u/.abi/wdbx"));
        assert_eq!(
            paths.manifest(),
            PathBuf::from("/home/u/.abi/wdbx.manifest")
        );
        assert_eq!(
            paths.mirror_epoch(),
            PathBuf::from("/home/u/.abi/wdbx.mirror-epoch")
        );
        assert_eq!(
            paths.segment(42),
            PathBuf::from("/home/u/.abi/wdbx.seg.42.jsonl")
        );
    }

    #[test]
    fn a_missing_manifest_is_an_empty_store_not_an_error() {
        // What a first run looks like.
        let dir = abi_foundation::temp_path::temp_file_path("abi_wdbx_no_manifest", "d");
        let paths = StorePaths::new(&dir);
        assert_eq!(paths.read_manifest().expect("no error"), Manifest::empty());
    }

    #[test]
    fn mirror_epoch_defaults_legacy_to_zero_and_rejects_corruption() {
        let dir = abi_foundation::temp_path::temp_file_path("abi_wdbx_mirror_epoch", "d");
        let paths = StorePaths::new(&dir);
        assert_eq!(paths.read_mirror_epoch().expect("legacy default"), 0);

        std::fs::create_dir_all(&dir).expect("fixture directory");
        std::fs::write(paths.mirror_epoch(), "not provenance\n").expect("corrupt sidecar");
        assert!(matches!(
            paths.read_mirror_epoch(),
            Err(FormatError::InvalidManifest { .. })
        ));
    }
}
