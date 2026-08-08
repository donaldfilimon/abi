//! Public WDBX v2 identities, causal values, and immutable snapshot model.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

/// Historical numeric identities and v2 UUID identities share one public type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RecordId {
    /// Identity read from a v1 store or accepted at a compatibility boundary.
    Legacy(u64),
    /// Stable v2 identity, serialized as a UUID string.
    V2(Uuid),
}

impl RecordId {
    /// Allocate a new v2 identity.
    #[must_use]
    pub fn new_v2() -> Self {
        Self::V2(Uuid::new_v4())
    }
}

impl std::fmt::Display for RecordId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy(id) => write!(formatter, "{id}"),
            Self::V2(id) => write!(formatter, "{id}"),
        }
    }
}

/// Highest committed sequence observed for each writer.
pub type CausalHeads = BTreeMap<Uuid, u64>;

/// Maximum vector width accepted by v2 stores.
pub const MAX_V2_VECTOR_DIMENSIONS: usize = 4096;

/// Spatial value stored under a stable public identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SpatialRecord {
    /// Stable record identity.
    pub id: RecordId,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
    /// Opaque payload.
    pub payload: String,
}

/// Temporal record kind preserved from v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum V2TemporalKind {
    /// Temporal graph node.
    Node,
    /// Temporal graph edge.
    Edge,
}

/// Temporal value keyed independently from its raw fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2TemporalRecord {
    /// Node or edge.
    pub kind: V2TemporalKind,
    /// Forward-compatible raw fields.
    pub fields: serde_json::Map<String, serde_json::Value>,
}

/// One immutable audit DAG block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2AuditBlock {
    /// Lowercase SHA-256 identity. Migrated v1 blocks retain their original hash.
    pub hash: String,
    /// All causal audit heads observed by the writer. Migrated v1 blocks have
    /// zero or one parent, mapping the historical chain into a DAG.
    pub parents: Vec<String>,
    /// Unix milliseconds.
    pub timestamp_ms: i64,
    /// Historical sequence, retained for migration equivalence.
    pub sequence: u64,
    /// Producing profile.
    pub profile: String,
    /// Query vector identity.
    pub query_id: RecordId,
    /// Response vector identity.
    pub response_id: RecordId,
    /// Opaque metadata.
    pub metadata: String,
}

/// A mutation carried by one causal transaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum V2Mutation {
    /// Write one key/value version.
    PutKv {
        /// Logical key.
        key: String,
        /// Versioned value.
        value: String,
    },
    /// Write one vector under a stable public identity.
    PutVector {
        /// Stable public vector identity.
        id: RecordId,
        /// Finite vector components.
        values: Vec<f32>,
    },
    /// Write one spatial version.
    PutSpatial {
        /// Spatial value.
        record: V2SpatialRecord,
    },
    /// Write one temporal version under a deterministic logical key.
    PutTemporal {
        /// Logical node/edge key.
        key: String,
        /// Temporal value.
        record: V2TemporalRecord,
    },
    /// Append one immutable audit DAG block.
    PutAudit {
        /// Audit block.
        block: V2AuditBlock,
    },
}

/// One committed multi-version value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Version<T> {
    /// Stable version identity used by explicit conflict resolution.
    pub version_id: Uuid,
    /// Writer that created this value.
    pub writer_id: Uuid,
    /// Writer-local committed sequence.
    pub sequence: u64,
    /// Causal frontier observed before the transaction was written.
    pub observed_heads: CausalHeads,
    /// Stored value.
    pub value: T,
}

impl<T> Version<T> {
    pub(super) fn dominates<U>(&self, other: &Version<U>) -> bool {
        self.writer_id == other.writer_id && self.sequence >= other.sequence
            || self
                .observed_heads
                .get(&other.writer_id)
                .is_some_and(|sequence| *sequence >= other.sequence)
    }
}

/// Preferred current version plus every unresolved concurrent version.
#[derive(Debug, Clone, PartialEq)]
pub struct ConflictSet<T> {
    /// Deterministic preferred version. Preference is presentation only; it
    /// does not resolve or discard the other versions.
    pub preferred: Version<T>,
    /// Concurrent current versions in deterministic identity order.
    pub conflicts: Vec<Version<T>>,
}

/// Immutable recovered v2 state. Views retain an `Arc` to this snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct V2Snapshot {
    pub(super) heads: CausalHeads,
    pub(super) kv: BTreeMap<String, Vec<Version<String>>>,
    pub(super) vectors: BTreeMap<RecordId, Vec<Version<Vec<f32>>>>,
    pub(super) spatial: BTreeMap<RecordId, Vec<Version<V2SpatialRecord>>>,
    pub(super) temporal: BTreeMap<String, Vec<Version<V2TemporalRecord>>>,
    pub(super) audit: BTreeMap<String, V2AuditBlock>,
    pub(super) audit_order: Vec<String>,
    pub(super) committed_transactions: usize,
}

impl V2Snapshot {
    /// Causal frontier represented by this snapshot.
    #[must_use]
    pub fn heads(&self) -> &CausalHeads {
        &self.heads
    }

    /// Number of verified committed transactions replayed.
    #[must_use]
    pub fn committed_transactions(&self) -> usize {
        self.committed_transactions
    }

    /// Return the maximal causal versions for a key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ConflictSet<String>> {
        current_versions(self.kv.get(key)?)
    }

    /// Return the maximal causal versions for one vector identity.
    #[must_use]
    pub fn get_vector(&self, id: RecordId) -> Option<ConflictSet<Vec<f32>>> {
        current_versions(self.vectors.get(&id)?)
    }

    /// Borrow the deterministic preferred vector without cloning it.
    #[must_use]
    pub fn preferred_vector(&self, id: RecordId) -> Option<&[f32]> {
        preferred_version(self.vectors.get(&id)?).map(|version| version.value.as_slice())
    }

    /// Return the maximal causal versions for one spatial identity.
    #[must_use]
    pub fn get_spatial(&self, id: RecordId) -> Option<ConflictSet<V2SpatialRecord>> {
        current_versions(self.spatial.get(&id)?)
    }

    /// Return the maximal causal versions for one temporal key.
    #[must_use]
    pub fn get_temporal(&self, key: &str) -> Option<ConflictSet<V2TemporalRecord>> {
        current_versions(self.temporal.get(key)?)
    }

    /// All immutable audit blocks in deterministic replay order.
    pub fn audit_blocks(&self) -> impl Iterator<Item = &V2AuditBlock> {
        self.audit_order
            .iter()
            .filter_map(|hash| self.audit.get(hash))
    }

    /// All stable vector identities in this immutable view.
    pub fn vector_ids(&self) -> impl Iterator<Item = RecordId> + '_ {
        self.vectors.keys().copied()
    }

    /// Retain this snapshot behind an `Arc` for mutation-safe consumers.
    #[must_use]
    pub fn retained(self) -> Arc<Self> {
        Arc::new(self)
    }
}

pub(super) fn current_versions<T: Clone>(versions: &[Version<T>]) -> Option<ConflictSet<T>> {
    let mut current: Vec<_> = versions
        .iter()
        .filter(|candidate| {
            !versions
                .iter()
                .any(|other| other.version_id != candidate.version_id && other.dominates(candidate))
        })
        .cloned()
        .collect();
    current.sort_by_key(|version| version.version_id);
    let preferred = current.pop()?;
    Some(ConflictSet {
        preferred,
        conflicts: current,
    })
}

fn preferred_version<T>(versions: &[Version<T>]) -> Option<&Version<T>> {
    versions
        .iter()
        .filter(|candidate| {
            !versions
                .iter()
                .any(|other| other.version_id != candidate.version_id && other.dominates(candidate))
        })
        .max_by_key(|version| version.version_id)
}

/// V2 persistence or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum V2Error {
    /// Filesystem operation failed.
    #[error("WDBX v2 I/O failed for {path}: {message}")]
    Io {
        /// Object being accessed.
        path: std::path::PathBuf,
        /// Underlying I/O detail.
        message: String,
    },
    /// A committed journal transaction was malformed or failed verification.
    #[error("WDBX v2 journal {path} is corrupt at line {line}: {reason}")]
    CorruptJournal {
        /// Journal being replayed.
        path: std::path::PathBuf,
        /// One-based frame line.
        line: usize,
        /// Verification failure.
        reason: String,
    },
    /// An authenticated journal object could not be opened or verified.
    #[error("WDBX v2 security failure for {path}: {reason}")]
    Security {
        /// Journal object stream being accessed.
        path: std::path::PathBuf,
        /// Redacted key, framing, or authentication failure.
        reason: String,
    },
    /// Exclusive maintenance could not start or a writer raced it.
    #[error("WDBX v2 maintenance refused: {0}")]
    Maintenance(String),
    /// A mutation was rejected before any bytes were appended.
    #[error("invalid WDBX v2 mutation: {0}")]
    InvalidMutation(String),
    /// The directory does not declare the supported v2 format.
    #[error("unsupported WDBX v2 version marker in {path}: {found:?}")]
    UnsupportedVersion {
        /// Version marker path.
        path: std::path::PathBuf,
        /// Bytes decoded lossily for diagnostics.
        found: String,
    },
    /// Explicit resolution did not name exactly the current conflicting set.
    #[error("conflict resolution set does not match the current versions")]
    StaleResolution,
}
