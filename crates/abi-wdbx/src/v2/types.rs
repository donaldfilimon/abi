//! Public WDBX v2 identities, causal values, and immutable snapshot model.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

/// Historical numeric identities and v2 UUID identities share one public type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(untagged)]
pub enum RecordId {
    /// Identity read from a v1 store or accepted at a compatibility boundary.
    Legacy(u64),
    /// Stable v2 identity, serialized as a UUID string.
    V2(Uuid),
}

impl<'de> Deserialize<'de> for RecordId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RecordIdVisitor;

        impl serde::de::Visitor<'_> for RecordIdVisitor {
            type Value = RecordId;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a legacy integer ID or UUID string")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(RecordId::Legacy(value))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                value
                    .parse::<u64>()
                    .map(RecordId::Legacy)
                    .or_else(|_| Uuid::parse_str(value).map(RecordId::V2).map_err(E::custom))
            }
        }

        deserializer.deserialize_any(RecordIdVisitor)
    }
}

impl RecordId {
    /// Allocate a new v2 identity.
    #[must_use]
    pub fn new_v2() -> Self {
        Self::V2(Uuid::new_v4())
    }
}

impl From<u64> for RecordId {
    fn from(id: u64) -> Self {
        Self::Legacy(id)
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
        self.writer_id == other.writer_id && self.sequence > other.sequence
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

    /// Number of logical key/value entries. Concurrent versions count as one key.
    #[must_use]
    pub fn kv_count(&self) -> usize {
        self.kv.len()
    }

    /// Number of stable vector identities. Concurrent versions count as one identity.
    #[must_use]
    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    /// Number of stable spatial identities. Concurrent versions count as one identity.
    #[must_use]
    pub fn spatial_count(&self) -> usize {
        self.spatial.len()
    }

    /// Number of temporal logical keys of one kind, using the preferred current
    /// version only for presentation while preserving conflicts in the store.
    #[must_use]
    pub fn temporal_count(&self, kind: V2TemporalKind) -> usize {
        self.temporal
            .values()
            .filter_map(|versions| preferred_version(versions))
            .filter(|version| version.value.kind == kind)
            .count()
    }

    /// Number of immutable audit DAG blocks.
    #[must_use]
    pub fn audit_count(&self) -> usize {
        self.audit.len()
    }

    /// Shared vector dimensionality when all preferred current vectors agree.
    /// Empty or mixed-dimension snapshots return `None`.
    #[must_use]
    pub fn vector_dimensions(&self) -> Option<usize> {
        let mut dimensions = None;
        for versions in self.vectors.values() {
            let width = preferred_version(versions)?.value.len();
            match dimensions {
                None => dimensions = Some(width),
                Some(existing) if existing == width => {}
                Some(_) => return None,
            }
        }
        dimensions
    }

    /// Return the maximal causal versions for a key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ConflictSet<String>> {
        current_versions(self.kv.get(key)?)
    }

    /// Borrow the deterministic preferred key/value version without resolving
    /// or discarding concurrent values.
    #[must_use]
    pub fn preferred_value(&self, key: &str) -> Option<&str> {
        preferred_version(self.kv.get(key)?).map(|version| version.value.as_str())
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

    /// Current audit-DAG heads in deterministic hash order.
    #[must_use]
    pub fn audit_heads(&self) -> Vec<String> {
        let parents = self
            .audit
            .values()
            .flat_map(|block| block.parents.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>();
        self.audit
            .keys()
            .filter(|hash| !parents.contains(*hash))
            .cloned()
            .collect()
    }

    /// Verify that every audit parent exists and that parent edges form a DAG.
    pub fn verify_audit_dag(&self) -> Result<(), String> {
        fn visit(
            hash: &str,
            blocks: &BTreeMap<String, V2AuditBlock>,
            visiting: &mut std::collections::BTreeSet<String>,
            visited: &mut std::collections::BTreeSet<String>,
        ) -> Result<(), String> {
            if visited.contains(hash) {
                return Ok(());
            }
            if !visiting.insert(hash.to_owned()) {
                return Err(format!("audit cycle reaches {hash}"));
            }
            let block = blocks
                .get(hash)
                .ok_or_else(|| format!("missing audit block {hash}"))?;
            for parent in &block.parents {
                if !blocks.contains_key(parent) {
                    return Err(format!("audit block {hash} has missing parent {parent}"));
                }
                visit(parent, blocks, visiting, visited)?;
            }
            visiting.remove(hash);
            visited.insert(hash.to_owned());
            Ok(())
        }

        let mut visiting = std::collections::BTreeSet::new();
        let mut visited = std::collections::BTreeSet::new();
        for hash in self.audit.keys() {
            visit(hash, &self.audit, &mut visiting, &mut visited)?;
        }
        Ok(())
    }

    /// Preferred current spatial values in stable identity order.
    #[must_use]
    pub fn preferred_spatial_records(&self) -> Vec<&V2SpatialRecord> {
        self.spatial
            .values()
            .filter_map(|versions| preferred_version(versions))
            .map(|version| &version.value)
            .collect()
    }

    /// Preferred current temporal values in logical-key order.
    #[must_use]
    pub fn preferred_temporal_records(&self) -> Vec<&V2TemporalRecord> {
        self.temporal
            .values()
            .filter_map(|versions| preferred_version(versions))
            .map(|version| &version.value)
            .collect()
    }

    /// All stable vector identities in this immutable view.
    pub fn vector_ids(&self) -> impl Iterator<Item = RecordId> + '_ {
        self.vectors.keys().copied()
    }

    /// Choose a deterministic vector at the recovered causal frontier.
    ///
    /// A later same-writer version or a version that observed another writer's
    /// head dominates that older vector. Concurrent maximal vectors are all
    /// retained; this presentation-only selector breaks their tie by version
    /// identity and then stable record identity.
    #[must_use]
    pub fn causal_focus_vector_id(&self) -> Option<RecordId> {
        let candidates = self
            .vectors
            .iter()
            .filter_map(|(id, versions)| preferred_version(versions).map(|version| (*id, version)))
            .collect::<Vec<_>>();
        candidates
            .iter()
            .copied()
            .filter(|(_, candidate)| {
                !candidates.iter().any(|(_, other)| {
                    other.version_id != candidate.version_id && other.dominates(candidate)
                })
            })
            .max_by_key(|(id, version)| (version.version_id, *id))
            .map(|(id, _)| id)
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

#[cfg(test)]
mod record_id_tests {
    use super::*;

    #[test]
    fn record_ids_preserve_value_encoding_and_accept_json_map_keys() {
        let uuid = Uuid::from_u128(0xAB1);
        assert_eq!(serde_json::to_string(&RecordId::Legacy(42)).unwrap(), "42");
        assert_eq!(
            serde_json::to_string(&RecordId::V2(uuid)).unwrap(),
            format!("\"{uuid}\"")
        );
        assert_eq!(
            serde_json::from_str::<RecordId>("42").unwrap(),
            RecordId::Legacy(42)
        );
        assert_eq!(
            serde_json::from_str::<RecordId>("\"42\"").unwrap(),
            RecordId::Legacy(42)
        );
        assert_eq!(
            serde_json::from_str::<RecordId>(&format!("\"{uuid}\"")).unwrap(),
            RecordId::V2(uuid)
        );

        let map = BTreeMap::from([(RecordId::Legacy(42), "legacy"), (RecordId::V2(uuid), "v2")]);
        let encoded = serde_json::to_string(&map).unwrap();
        assert_eq!(
            serde_json::from_str::<BTreeMap<RecordId, &str>>(&encoded).unwrap(),
            map
        );
    }
}
