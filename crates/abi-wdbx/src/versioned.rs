//! Product-facing WDBX facade over read-compatible v1 and writable v2 stores.
//!
//! [`DurableStore`](crate::DurableStore) remains the legacy engine used by
//! migration fixtures. Product callers should use [`VersionedStore`] for writes
//! and [`open_versioned_read_only`](crate::v2::open_versioned_read_only) for
//! inspection so v1 reads remain non-mutating and the first write performs the
//! verified v1-to-v2 activation.

use crate::hnsw::{HnswError, HnswIndex};
use crate::v2::{
    CompactionReport, ConflictSet, MigrationError, MigrationReport, RecordId, V2AuditBlock,
    V2Error, V2Mutation, V2SearchResult, V2Snapshot, V2SpatialRecord, V2Store, V2TemporalKind,
    V2TemporalRecord, V2VectorIndex, VersionedSnapshot, open_versioned_writable,
};
use crate::{Snapshot, StorePaths};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::sync::Arc;
use uuid::Uuid;

/// Format-neutral counters used by CLI, MCP, and gateway presentation layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VersionedStats {
    /// Distinct logical key/value keys.
    pub kv_entries: usize,
    /// Distinct stable vector identities.
    pub vectors: usize,
    /// Immutable audit blocks.
    pub blocks: usize,
    /// Distinct spatial identities.
    pub spatial_records: usize,
    /// Preferred current temporal nodes.
    pub temporal_nodes: usize,
    /// Preferred current temporal edges.
    pub temporal_edges: usize,
}

/// Product-facade failure.
#[derive(Debug, thiserror::Error)]
pub enum VersionedError {
    /// Version detection, migration, or activation failed.
    #[error(transparent)]
    Migration(#[from] MigrationError),
    /// A v2 mutation or recovery operation failed.
    #[error(transparent)]
    V2(#[from] V2Error),
    /// Vector index construction or search failed.
    #[error(transparent)]
    Hnsw(#[from] HnswError),
    /// An audit block sequence cannot be represented.
    #[error("WDBX audit block sequence is exhausted")]
    AuditSequenceOverflow,
}

/// A version-neutral search result retaining its immutable backing snapshot.
#[derive(Debug, Clone)]
pub struct VersionedSearchResult {
    /// Stable public identity: a JSON number for migrated v1 or UUID string for v2.
    pub id: RecordId,
    /// Cosine similarity.
    pub score: f32,
    vector: VersionedVectorView,
}

#[derive(Debug, Clone)]
enum VersionedVectorView {
    V1 { snapshot: Arc<Snapshot>, id: u64 },
    V2(V2SearchResult),
}

impl VersionedSearchResult {
    /// Zero-copy vector view whose backing snapshot remains alive with this result.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        match &self.vector {
            VersionedVectorView::V1 { snapshot, id } => snapshot
                .vectors
                .get(id)
                .expect("the retained v1 snapshot contains every returned id"),
            VersionedVectorView::V2(result) => result.vector(),
        }
    }
}

impl VersionedSnapshot {
    /// On-disk format represented by this immutable view.
    #[must_use]
    pub const fn format_version(&self) -> u8 {
        match self {
            Self::Empty | Self::V2(_) => 2,
            Self::V1(_) => 1,
        }
    }

    /// Format-neutral logical counters.
    #[must_use]
    pub fn stats(&self) -> VersionedStats {
        match self {
            Self::Empty => VersionedStats::default(),
            Self::V1(snapshot) => VersionedStats {
                kv_entries: snapshot.stats.kv_entries,
                vectors: snapshot.stats.vectors,
                blocks: snapshot.stats.blocks,
                spatial_records: snapshot.stats.spatial_records,
                temporal_nodes: snapshot.stats.temporal_nodes,
                temporal_edges: snapshot.stats.temporal_edges,
            },
            Self::V2(snapshot) => VersionedStats {
                kv_entries: snapshot.kv_count(),
                vectors: snapshot.vector_count(),
                blocks: snapshot.audit_count(),
                spatial_records: snapshot.spatial_count(),
                temporal_nodes: snapshot.temporal_count(V2TemporalKind::Node),
                temporal_edges: snapshot.temporal_count(V2TemporalKind::Edge),
            },
        }
    }

    /// Shared vector width when every preferred current vector agrees.
    #[must_use]
    pub fn vector_dimensions(&self) -> Option<usize> {
        match self {
            Self::Empty => None,
            Self::V1(snapshot) => snapshot.vector_dimensions(),
            Self::V2(snapshot) => snapshot.vector_dimensions(),
        }
    }

    /// Legacy successor identity. V2 UUID spaces intentionally have no successor.
    #[must_use]
    pub fn next_legacy_vector_id(&self) -> Option<u64> {
        match self {
            Self::V1(snapshot) => snapshot.max_vector_id().unwrap_or(0).checked_add(1),
            Self::Empty | Self::V2(_) => None,
        }
    }

    /// Deterministic preferred key/value presentation. V2 conflicts remain
    /// available through [`VersionedStore::get_conflicts`].
    #[must_use]
    pub fn preferred_value(&self, key: &str) -> Option<&str> {
        match self {
            Self::Empty => None,
            Self::V1(snapshot) => snapshot.kv.get(key).map(String::as_str),
            Self::V2(snapshot) => snapshot.preferred_value(key),
        }
    }

    /// Stable vector identities in deterministic order.
    #[must_use]
    pub fn vector_ids(&self) -> Vec<RecordId> {
        match self {
            Self::Empty => Vec::new(),
            Self::V1(snapshot) => snapshot
                .vectors
                .keys()
                .copied()
                .map(RecordId::Legacy)
                .collect(),
            Self::V2(snapshot) => snapshot.vector_ids().collect(),
        }
    }

    /// Choose a deterministic vector at the snapshot's causal frontier.
    ///
    /// Legacy v1 has one total append order, so its largest numeric identity is
    /// the latest vector. V2 delegates to the causal selector and never treats
    /// UUID lexical order as time.
    #[must_use]
    pub fn causal_focus_vector_id(&self) -> Option<RecordId> {
        match self {
            Self::Empty => None,
            Self::V1(snapshot) => snapshot
                .vectors
                .keys()
                .next_back()
                .copied()
                .map(RecordId::Legacy),
            Self::V2(snapshot) => snapshot.causal_focus_vector_id(),
        }
    }

    /// Search either format while retaining the immutable source snapshot.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VersionedSearchResult>, VersionedError> {
        match self {
            Self::Empty => {
                validate_empty_query(query, crate::v2::MAX_V2_VECTOR_DIMENSIONS)?;
                Ok(Vec::new())
            }
            Self::V1(snapshot) if snapshot.vectors.is_empty() => {
                validate_empty_query(query, crate::wal::MAX_VECTOR_DIMENSIONS)?;
                Ok(Vec::new())
            }
            Self::V1(snapshot) => {
                let index = HnswIndex::from_snapshot(snapshot)?;
                Ok(index
                    .search(query, limit)?
                    .into_iter()
                    .map(|result| VersionedSearchResult {
                        id: RecordId::Legacy(result.id),
                        score: result.score,
                        vector: VersionedVectorView::V1 {
                            snapshot: Arc::clone(snapshot),
                            id: result.id,
                        },
                    })
                    .collect())
            }
            Self::V2(snapshot) if snapshot.vector_count() == 0 => {
                validate_empty_query(query, crate::v2::MAX_V2_VECTOR_DIMENSIONS)?;
                Ok(Vec::new())
            }
            Self::V2(snapshot) => Ok(V2VectorIndex::from_snapshot(Arc::clone(snapshot))?
                .search(query, limit)?
                .into_iter()
                .map(|result| VersionedSearchResult {
                    id: result.id,
                    score: result.score,
                    vector: VersionedVectorView::V2(result),
                })
                .collect()),
        }
    }
}

fn validate_empty_query(query: &[f32], maximum: usize) -> Result<(), HnswError> {
    if query.is_empty() || query.len() > maximum || query.iter().any(|value| !value.is_finite()) {
        return Err(HnswError::InvalidDimensions {
            dimensions: query.len(),
        });
    }
    Ok(())
}

/// Writable product store. Opening always yields v2, after verified migration
/// when the authoritative source was v1.
pub struct VersionedStore {
    paths: StorePaths,
    inner: V2Store,
    migration: MigrationReport,
}

impl std::fmt::Debug for VersionedStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VersionedStore")
            .field("paths", &self.paths)
            .field("writer_id", &self.inner.writer_id())
            .field("migration", &self.migration)
            .finish_non_exhaustive()
    }
}

impl VersionedStore {
    /// Open a writable product store, performing the specified safe migration.
    pub fn open(paths: StorePaths) -> Result<Self, VersionedError> {
        let (inner, migration) = open_versioned_writable(&paths)?;
        Ok(Self {
            paths,
            inner,
            migration,
        })
    }

    /// Original versioned base paths, including activation metadata location.
    #[must_use]
    pub fn paths(&self) -> &StorePaths {
        &self.paths
    }

    /// Migration/activation evidence from this open.
    #[must_use]
    pub fn migration_report(&self) -> &MigrationReport {
        &self.migration
    }

    /// Retain the current immutable v2 view.
    #[must_use]
    pub fn snapshot(&self) -> Arc<V2Snapshot> {
        self.inner.snapshot()
    }

    /// Format-neutral counters.
    #[must_use]
    pub fn stats(&self) -> VersionedStats {
        VersionedSnapshot::V2(self.snapshot()).stats()
    }

    /// Write one causal key/value version.
    pub fn put(&mut self, key: &str, value: &str) -> Result<Uuid, VersionedError> {
        Ok(self.inner.commit(vec![V2Mutation::PutKv {
            key: key.to_owned(),
            value: value.to_owned(),
        }])?)
    }

    /// Deterministic preferred value without hiding its concurrent siblings.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        self.snapshot().get(key).map(|set| set.preferred.value)
    }

    /// Current causal versions for explicit conflict presentation/resolution.
    #[must_use]
    pub fn get_conflicts(&self, key: &str) -> Option<ConflictSet<String>> {
        self.snapshot().get(key)
    }

    /// Resolve exactly the named current conflicting versions.
    pub fn resolve(
        &mut self,
        key: &str,
        version_ids: &[Uuid],
        value: String,
    ) -> Result<Uuid, VersionedError> {
        Ok(self.inner.resolve(key, version_ids, value)?)
    }

    /// Insert a vector under a fresh UUID identity.
    pub fn put_vector(&mut self, values: &[f32]) -> Result<RecordId, VersionedError> {
        let id = RecordId::new_v2();
        self.inner.commit(vec![V2Mutation::PutVector {
            id,
            values: values.to_vec(),
        }])?;
        Ok(id)
    }

    /// Search the current immutable v2 snapshot.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VersionedSearchResult>, VersionedError> {
        VersionedSnapshot::V2(self.snapshot()).search(query, limit)
    }

    /// Return every current vector version for an identity.
    #[must_use]
    pub fn get_vector(&self, id: RecordId) -> Option<ConflictSet<Vec<f32>>> {
        self.snapshot().get_vector(id)
    }

    /// Upsert one causal spatial version.
    pub fn put_spatial(&mut self, record: V2SpatialRecord) -> Result<Uuid, VersionedError> {
        Ok(self.inner.commit(vec![V2Mutation::PutSpatial { record }])?)
    }

    /// Upsert one temporal node.
    pub fn add_temporal_node(
        &mut self,
        id: impl Into<RecordId>,
        timestamp_ms: i64,
    ) -> Result<Uuid, VersionedError> {
        let id = id.into();
        let mut fields = serde_json::Map::new();
        fields.insert(
            "id".into(),
            serde_json::to_value(id).expect("RecordId serializes"),
        );
        fields.insert("timestamp_ms".into(), serde_json::json!(timestamp_ms));
        Ok(self.inner.commit(vec![V2Mutation::PutTemporal {
            key: format!("node:{id}"),
            record: V2TemporalRecord {
                kind: V2TemporalKind::Node,
                fields,
            },
        }])?)
    }

    /// Append one temporal causal edge.
    pub fn add_temporal_edge(
        &mut self,
        cause: impl Into<RecordId>,
        effect: impl Into<RecordId>,
    ) -> Result<Uuid, VersionedError> {
        let cause = cause.into();
        let effect = effect.into();
        let mut fields = serde_json::Map::new();
        fields.insert(
            "cause".into(),
            serde_json::to_value(cause).expect("RecordId serializes"),
        );
        fields.insert(
            "effect".into(),
            serde_json::to_value(effect).expect("RecordId serializes"),
        );
        Ok(self.inner.commit(vec![V2Mutation::PutTemporal {
            key: format!("edge:{cause}:{effect}:{}", Uuid::new_v4()),
            record: V2TemporalRecord {
                kind: V2TemporalKind::Edge,
                fields,
            },
        }])?)
    }

    /// Append one audit DAG block whose parents are all currently observed heads.
    pub fn add_block(
        &mut self,
        profile: &str,
        query_id: RecordId,
        response_id: RecordId,
        metadata: &str,
        timestamp_ms: i64,
    ) -> Result<V2AuditBlock, VersionedError> {
        let mut committed = None;
        self.inner
            .commit_with_snapshot::<VersionedError, _>(|snapshot| {
                let sequence = u64::try_from(snapshot.audit_count())
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or(VersionedError::AuditSequenceOverflow)?;
                let parents = snapshot.audit_heads();
                let hash = audit_hash(
                    &parents,
                    timestamp_ms,
                    sequence,
                    profile,
                    query_id,
                    response_id,
                    metadata,
                );
                let block = V2AuditBlock {
                    hash,
                    parents,
                    timestamp_ms,
                    sequence,
                    profile: profile.to_owned(),
                    query_id,
                    response_id,
                    metadata: metadata.to_owned(),
                };
                committed = Some(block.clone());
                Ok(vec![V2Mutation::PutAudit { block }])
            })?;
        committed.ok_or_else(|| {
            V2Error::InvalidMutation("audit transaction did not construct a block".into()).into()
        })
    }

    /// Publish an immutable segment covering the observed causal frontier.
    pub fn compact(&mut self) -> Result<CompactionReport, VersionedError> {
        Ok(self.inner.compact()?)
    }
}

#[derive(Serialize)]
struct AuditHashInput<'a> {
    parents: &'a [String],
    timestamp_ms: i64,
    sequence: u64,
    profile: &'a str,
    query_id: RecordId,
    response_id: RecordId,
    metadata: &'a str,
}

fn audit_hash(
    parents: &[String],
    timestamp_ms: i64,
    sequence: u64,
    profile: &str,
    query_id: RecordId,
    response_id: RecordId,
    metadata: &str,
) -> String {
    let input = AuditHashInput {
        parents,
        timestamp_ms,
        sequence,
        profile,
        query_id,
        response_id,
        metadata,
    };
    let bytes = serde_json::to_vec(&input).expect("audit hash input serializes");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DurableStore, SpatialRecord};

    fn scratch(label: &str) -> StorePaths {
        StorePaths::new(abi_foundation::temp_path::temp_file_path(label, "store"))
    }

    #[test]
    fn first_product_write_migrates_v1_and_allocates_uuid_ids() {
        let paths = scratch("wdbx-versioned-migrate");
        let legacy_id = {
            let mut legacy = DurableStore::open(paths.clone()).unwrap();
            let id = legacy.put_vector(&[1.0, 0.0]).unwrap();
            legacy.put("legacy", "visible").unwrap();
            legacy.checkpoint().unwrap();
            id
        };

        let read_only = crate::v2::open_versioned_read_only(&paths).unwrap();
        assert_eq!(read_only.format_version(), 1);
        assert!(!paths.dir.join(format!("{}.active", paths.base)).exists());

        let mut store = VersionedStore::open(paths.clone()).unwrap();
        assert!(store.migration_report().migrated);
        assert_eq!(store.get("legacy").as_deref(), Some("visible"));
        assert!(store.get_vector(RecordId::Legacy(legacy_id)).is_some());
        let new_id = store.put_vector(&[0.0, 1.0]).unwrap();
        assert!(matches!(new_id, RecordId::V2(_)));

        let view = crate::v2::open_versioned_read_only(&paths).unwrap();
        assert_eq!(view.format_version(), 2);
        assert_eq!(view.stats().vectors, 2);
        std::fs::remove_dir_all(paths.dir).unwrap();
    }

    #[test]
    fn result_views_survive_mutations_and_audit_blocks_form_a_dag() {
        let paths = scratch("wdbx-versioned-lifetime");
        let mut store = VersionedStore::open(paths.clone()).unwrap();
        let first = store.put_vector(&[1.0, 0.0]).unwrap();
        let result = store.search(&[1.0, 0.0], 1).unwrap().remove(0);
        assert_eq!(result.id, first);
        assert_eq!(result.vector(), [1.0, 0.0]);

        store.put_vector(&[0.0, 1.0]).unwrap();
        assert_eq!(result.vector(), [1.0, 0.0]);
        let genesis = store.add_block("abbey", first, first, "first", 1).unwrap();
        let child = store.add_block("abbey", first, first, "second", 2).unwrap();
        assert_eq!(genesis.parents, Vec::<String>::new());
        assert_eq!(child.parents, [genesis.hash]);
        std::fs::remove_dir_all(paths.dir).unwrap();
    }

    #[test]
    fn audit_block_uses_the_same_refreshed_frontier_as_its_commit() {
        let paths = scratch("wdbx-versioned-audit-frontier");
        let mut stale = VersionedStore::open(paths.clone()).unwrap();
        let mut peer = VersionedStore::open(paths.clone()).unwrap();
        let id = peer.put_vector(&[1.0]).unwrap();
        let peer_block = peer.add_block("abi", id, id, "peer", 1).unwrap();

        let stale_block = stale.add_block("abi", id, id, "stale", 2).unwrap();
        assert_eq!(stale_block.parents, [peer_block.hash]);
        assert_eq!(stale.snapshot().audit_heads(), [stale_block.hash]);
        std::fs::remove_dir_all(paths.dir).unwrap();
    }

    #[test]
    fn versioned_focus_preserves_v1_order_and_v2_causality() {
        let paths = scratch("wdbx-versioned-focus-v1");
        {
            let mut legacy = DurableStore::open(paths.clone()).unwrap();
            assert_eq!(legacy.put_vector(&[1.0]).unwrap(), 1);
            assert_eq!(legacy.put_vector(&[2.0]).unwrap(), 2);
            legacy.checkpoint().unwrap();
        }
        let legacy = crate::v2::open_versioned_read_only(&paths).unwrap();
        assert_eq!(legacy.causal_focus_vector_id(), Some(RecordId::Legacy(2)));
        std::fs::remove_dir_all(paths.dir).unwrap();

        let root = scratch("wdbx-versioned-focus-v2");
        let mut store = VersionedStore::open(root.clone()).unwrap();
        let first = store.put_vector(&[1.0]).unwrap();
        let second = store.put_vector(&[2.0]).unwrap();
        assert_ne!(first, second);
        assert_eq!(
            VersionedSnapshot::V2(store.snapshot()).causal_focus_vector_id(),
            Some(second)
        );
        std::fs::remove_dir_all(root.dir).unwrap();
    }

    #[test]
    fn facade_exposes_all_causal_record_families() {
        let paths = scratch("wdbx-versioned-families");
        let mut store = VersionedStore::open(paths.clone()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let generation = &store.migration_report().generation;
            assert_eq!(
                std::fs::metadata(generation).unwrap().permissions().mode() & 0o077,
                0
            );
        }
        let id = store.put_vector(&[1.0]).unwrap();
        store
            .put_spatial(V2SpatialRecord {
                id,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                payload: "point".into(),
            })
            .unwrap();
        store.add_temporal_node(id, 10).unwrap();
        store.add_temporal_edge(id, id).unwrap();
        let stats = store.stats();
        assert_eq!(stats.vectors, 1);
        assert_eq!(stats.spatial_records, 1);
        assert_eq!(stats.temporal_nodes, 1);
        assert_eq!(stats.temporal_edges, 1);

        // The legacy record type remains usable only at the migration boundary.
        let _legacy_shape = SpatialRecord {
            id: 1,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            payload: String::new(),
        };
        std::fs::remove_dir_all(paths.dir).unwrap();
    }
}
