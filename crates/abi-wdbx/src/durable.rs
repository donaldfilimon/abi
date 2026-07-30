//! Durable WDBX store integrating snapshots, WAL recovery and HNSW search.

use crate::SearchResult;
use crate::format::{SpatialRecord, StorePaths, TemporalKind, TemporalRecord};
use crate::hnsw::{HnswError, HnswIndex};
use crate::segments::{CompactionResult, SegmentError};
use crate::store::{Snapshot, SnapshotStats};
use crate::wal::{
    RecoverySource, WalError, append_block, append_kv, append_spatial, append_temporal_edge,
    append_temporal_node, append_vector, build_block, checkpoint, create_with_epoch, recover,
    wal_path,
};

/// A durable-store mutation or recovery failure.
#[derive(Debug)]
pub enum DurableError {
    /// Checkpoint/WAL failure.
    Wal(WalError),
    /// HNSW graph failure.
    Hnsw(HnswError),
    /// Segment listing or compaction failure.
    Segment(SegmentError),
    /// Empty keys are rejected like the Zig store.
    InvalidKey,
    /// No next vector id can be represented.
    VectorIdOverflow,
}

impl std::fmt::Display for DurableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wal(error) => write!(formatter, "{error}"),
            Self::Hnsw(error) => write!(formatter, "{error}"),
            Self::Segment(error) => write!(formatter, "{error}"),
            Self::InvalidKey => formatter.write_str("key must not be empty"),
            Self::VectorIdOverflow => formatter.write_str("vector id space is exhausted"),
        }
    }
}

impl std::error::Error for DurableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wal(error) => Some(error),
            Self::Hnsw(error) => Some(error),
            Self::Segment(error) => Some(error),
            Self::InvalidKey | Self::VectorIdOverflow => None,
        }
    }
}

impl From<WalError> for DurableError {
    fn from(error: WalError) -> Self {
        Self::Wal(error)
    }
}

impl From<HnswError> for DurableError {
    fn from(error: HnswError) -> Self {
        Self::Hnsw(error)
    }
}

impl From<SegmentError> for DurableError {
    fn from(error: SegmentError) -> Self {
        Self::Segment(error)
    }
}

/// Durable result alias.
pub type Result<T> = std::result::Result<T, DurableError>;

/// A recovered WDBX snapshot with WAL-backed mutations and HNSW search.
#[derive(Debug)]
pub struct DurableStore {
    paths: StorePaths,
    snapshot: Snapshot,
    index: Option<HnswIndex>,
    next_vector_id: u64,
    checkpoint_epoch: u64,
    recovery_source: RecoverySource,
    frames_applied: usize,
}

impl DurableStore {
    /// Recover a durable store and rebuild its HNSW graph.
    ///
    /// A torn WAL is immediately folded into a fresh checkpoint before further
    /// appends, so a partial final frame can never be extended into corruption.
    pub fn open(paths: StorePaths) -> Result<Self> {
        let recovered = recover(&paths)?;
        let mut snapshot = recovered.snapshot;
        let mut checkpoint_epoch = recovered.checkpoint_epoch;
        let mut recovery_source = recovered.source;
        let mut frames_applied = recovered.frames_applied;

        if recovered.torn_wal_tail {
            checkpoint_epoch = checkpoint(&paths, &snapshot)?;
            recovery_source = RecoverySource::Segment;
            frames_applied = 0;
        } else {
            create_with_epoch(wal_path(&paths), checkpoint_epoch)?;
        }

        let index = if snapshot.vectors.is_empty() {
            None
        } else {
            Some(HnswIndex::from_snapshot(&snapshot)?)
        };
        let next_vector_id = snapshot
            .max_vector_id()
            .map_or(Some(1), |id| id.checked_add(1))
            .ok_or(DurableError::VectorIdOverflow)?;
        snapshot.recount();

        Ok(Self {
            paths,
            snapshot,
            index,
            next_vector_id,
            checkpoint_epoch,
            recovery_source,
            frames_applied,
        })
    }

    /// Open a store using WDBX's default base name in `directory`.
    pub fn open_directory(directory: impl Into<std::path::PathBuf>) -> Result<Self> {
        Self::open(StorePaths::new(directory))
    }

    /// Store paths owned by this session.
    #[must_use]
    pub fn paths(&self) -> &StorePaths {
        &self.paths
    }

    /// Recovered and subsequently mutated state.
    #[must_use]
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Recovery source selected when this session opened.
    #[must_use]
    pub fn recovery_source(&self) -> RecoverySource {
        self.recovery_source
    }

    /// Checkpoint epoch underlying the current WAL.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> u64 {
        self.checkpoint_epoch
    }

    /// WAL frames replayed when this session opened.
    #[must_use]
    pub fn frames_applied(&self) -> usize {
        self.frames_applied
    }

    /// Next absolute vector id.
    #[must_use]
    pub fn next_vector_id(&self) -> u64 {
        self.next_vector_id
    }

    /// Current snapshot counters.
    #[must_use]
    pub fn stats(&self) -> SnapshotStats {
        self.snapshot.stats
    }

    /// Store one key/value mutation durably.
    pub fn put(&mut self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() {
            return Err(DurableError::InvalidKey);
        }
        append_kv(wal_path(&self.paths), key, value)?;
        self.snapshot.kv.insert(key.to_string(), value.to_string());
        self.snapshot.recount();
        Ok(())
    }

    /// Borrow one key/value entry.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.snapshot.kv.get(key).map(String::as_str)
    }

    /// Insert one vector, WAL-first after a rollback-capable graph insert.
    pub fn put_vector(&mut self, values: &[f32]) -> Result<u64> {
        let id = self.next_vector_id;
        let next_id = id.checked_add(1).ok_or(DurableError::VectorIdOverflow)?;

        if self.index.is_none() {
            self.index = Some(HnswIndex::new(values.len())?);
        }
        let index = self.index.as_mut().expect("index was initialized");
        index.insert(id, values)?;
        if let Err(error) = append_vector(wal_path(&self.paths), id, values) {
            let rolled_back = index.rollback_last_insert(id);
            debug_assert!(rolled_back, "newest HNSW insert must be rollback-capable");
            return Err(error.into());
        }

        self.snapshot.vectors.insert(id, values.to_vec());
        index.commit_last_insert(id);
        self.next_vector_id = next_id;
        self.snapshot.recount();
        Ok(id)
    }

    /// Borrow one stored vector.
    #[must_use]
    pub fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.index.as_ref().and_then(|index| index.get(id))
    }

    /// Search the HNSW graph.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult<'_>>> {
        match &self.index {
            Some(index) => Ok(index.search(query, limit)?),
            None if query.is_empty() || query.len() > crate::wal::MAX_VECTOR_DIMENSIONS => {
                Err(HnswError::InvalidDimensions {
                    dimensions: query.len(),
                }
                .into())
            }
            None => Ok(Vec::new()),
        }
    }

    /// Append one deterministic audit-chain block.
    pub fn add_block(
        &mut self,
        profile: &str,
        query_id: u64,
        response_id: u64,
        metadata: &str,
        timestamp_ms: i64,
    ) -> Result<crate::BlockRecord> {
        let block = build_block(
            &self.snapshot,
            profile,
            query_id,
            response_id,
            metadata,
            timestamp_ms,
        )?;
        append_block(
            wal_path(&self.paths),
            profile,
            query_id,
            response_id,
            metadata,
            timestamp_ms,
        )?;
        self.snapshot.blocks.push(block.clone());
        self.snapshot.recount();
        Ok(block)
    }

    /// Upsert one spatial record.
    pub fn put_spatial(&mut self, record: SpatialRecord) -> Result<()> {
        append_spatial(wal_path(&self.paths), &record)?;
        self.snapshot.spatial.insert(record.id, record);
        self.snapshot.recount();
        Ok(())
    }

    /// Upsert one temporal node by id.
    pub fn add_temporal_node(&mut self, id: u64, timestamp_ms: i64) -> Result<()> {
        append_temporal_node(wal_path(&self.paths), id, timestamp_ms)?;
        let mut fields = serde_json::Map::new();
        fields.insert("id".to_string(), serde_json::json!(id));
        fields.insert("timestamp_ms".to_string(), serde_json::json!(timestamp_ms));
        let record = TemporalRecord {
            kind: TemporalKind::Node,
            fields,
        };
        if let Some(existing) = self.snapshot.temporal.iter_mut().find(|entry| {
            entry.kind == TemporalKind::Node
                && entry.fields.get("id").and_then(serde_json::Value::as_u64) == Some(id)
        }) {
            *existing = record;
        } else {
            self.snapshot.temporal.push(record);
        }
        self.snapshot.recount();
        Ok(())
    }

    /// Append one temporal causal edge.
    pub fn add_temporal_edge(&mut self, cause: u64, effect: u64) -> Result<()> {
        append_temporal_edge(wal_path(&self.paths), cause, effect)?;
        let mut fields = serde_json::Map::new();
        fields.insert("cause".to_string(), serde_json::json!(cause));
        fields.insert("effect".to_string(), serde_json::json!(effect));
        self.snapshot.temporal.push(TemporalRecord {
            kind: TemporalKind::Edge,
            fields,
        });
        self.snapshot.recount();
        Ok(())
    }

    /// Publish a complete checkpoint and reset the WAL to its new epoch.
    pub fn checkpoint(&mut self) -> Result<u64> {
        let epoch = checkpoint(&self.paths, &self.snapshot)?;
        self.checkpoint_epoch = epoch;
        self.recovery_source = RecoverySource::Segment;
        self.frames_applied = 0;
        Ok(epoch)
    }

    /// Retain only the newest checkpoint epochs.
    ///
    /// The checkpoint that anchors the current WAL is always retained.
    pub fn compact(&self, keep_latest: usize) -> Result<CompactionResult> {
        Ok(crate::segments::compact_retain_latest(
            &self.paths,
            keep_latest,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    struct Fixture {
        dir: std::path::PathBuf,
        paths: StorePaths,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = abi_foundation::temp_path::temp_file_path(name, "store");
            std::fs::create_dir_all(&dir).expect("fixture directory");
            Self {
                paths: StorePaths {
                    dir: dir.clone(),
                    base: "durable".to_string(),
                },
                dir,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn open_empty_creates_an_epoch_zero_wal() {
        let fixture = Fixture::new("abi_durable_empty");
        let store = DurableStore::open(fixture.paths.clone()).expect("open");
        assert_eq!(store.recovery_source(), RecoverySource::Empty);
        assert_eq!(store.checkpoint_epoch(), 0);
        assert!(wal_path(&fixture.paths).is_file());
    }

    #[test]
    fn kv_and_vector_mutations_survive_reopen() {
        let fixture = Fixture::new("abi_durable_reopen");
        {
            let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
            store.put("persona", "abbey").expect("put");
            assert_eq!(store.put_vector(&[1.0, 0.0]).expect("vector"), 1);
            assert_eq!(store.put_vector(&[0.0, 1.0]).expect("vector"), 2);
        }

        let store = DurableStore::open(fixture.paths.clone()).expect("reopen");
        assert_eq!(store.get("persona"), Some("abbey"));
        assert_eq!(store.frames_applied(), 3);
        assert_eq!(store.next_vector_id(), 3);
        let results = store.search(&[1.0, 0.0], 1).expect("search");
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn wal_failure_rolls_back_the_graph_without_burning_the_id() {
        let fixture = Fixture::new("abi_durable_rollback");
        let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
        let wal = wal_path(&fixture.paths);
        std::fs::remove_file(&wal).expect("remove WAL");
        std::fs::create_dir(&wal).expect("replace WAL with directory");

        assert!(store.put_vector(&[1.0, 0.0]).is_err());
        assert_eq!(store.next_vector_id(), 1);
        assert!(store.get_vector(1).is_none());
        assert_eq!(store.stats().vectors, 0);

        std::fs::remove_dir(&wal).expect("remove blocking directory");
        create_with_epoch(&wal, 0).expect("restore WAL");
        assert_eq!(store.put_vector(&[1.0, 0.0]).expect("retry"), 1);
    }

    #[test]
    fn checkpoint_resets_wal_and_preserves_id_continuity() {
        let fixture = Fixture::new("abi_durable_checkpoint");
        let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
        store.put("k", "v").expect("put");
        store.put_vector(&[1.0, 0.0]).expect("vector");
        assert_eq!(store.checkpoint().expect("checkpoint"), 0);
        assert_eq!(
            crate::wal::read_base_epoch(wal_path(&fixture.paths)).expect("base epoch"),
            0
        );
        drop(store);

        let mut reopened = DurableStore::open(fixture.paths.clone()).expect("reopen");
        assert_eq!(reopened.recovery_source(), RecoverySource::Merged);
        assert_eq!(reopened.frames_applied(), 0);
        assert_eq!(reopened.put_vector(&[0.0, 1.0]).expect("next"), 2);
    }

    #[test]
    fn every_mutation_family_survives_recovery() {
        let fixture = Fixture::new("abi_durable_families");
        {
            let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
            store.put_vector(&[1.0, 0.0]).expect("query vector");
            store.put_vector(&[0.0, 1.0]).expect("response vector");
            store
                .add_block("abbey", 1, 2, "metadata", 123)
                .expect("block");
            store
                .put_spatial(SpatialRecord {
                    id: 8,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                    payload: "point".to_string(),
                })
                .expect("spatial");
            store.add_temporal_node(3, 99).expect("node");
            store.add_temporal_node(3, 100).expect("node upsert");
            store.add_temporal_edge(3, 4).expect("edge");
        }

        let store = DurableStore::open(fixture.paths.clone()).expect("reopen");
        assert_eq!(store.stats().vectors, 2);
        assert_eq!(store.stats().blocks, 1);
        assert_eq!(store.stats().spatial_records, 1);
        assert_eq!(store.stats().temporal_nodes, 1);
        assert_eq!(store.stats().temporal_edges, 1);
        store.snapshot().verify_chain().expect("block chain");
    }

    #[test]
    fn torn_wal_is_checkpointed_before_future_appends() {
        let fixture = Fixture::new("abi_durable_torn");
        {
            let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
            store.put("before", "tear").expect("put");
        }
        let wal = wal_path(&fixture.paths);
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal)
            .expect("open WAL");
        write!(file, "deadbeef-incomplete").expect("tear");
        file.sync_all().expect("sync tear");

        let mut recovered = DurableStore::open(fixture.paths.clone()).expect("recover");
        assert_eq!(recovered.get("before"), Some("tear"));
        assert_eq!(crate::wal::verify(&wal).expect("repaired WAL"), 0);
        recovered.put("after", "repair").expect("append");
        drop(recovered);
        let reopened = DurableStore::open(fixture.paths.clone()).expect("reopen");
        assert_eq!(reopened.get("before"), Some("tear"));
        assert_eq!(reopened.get("after"), Some("repair"));
    }

    #[test]
    fn compaction_preserves_latest_checkpoint_plus_wal_recovery() {
        let fixture = Fixture::new("abi_durable_compact");
        let mut store = DurableStore::open(fixture.paths.clone()).expect("open");
        for value in ["zero", "one", "two"] {
            store.put("checkpoint", value).expect("put");
            store.checkpoint().expect("checkpoint");
        }
        store.put("wal-delta", "preserved").expect("delta");

        let result = store.compact(1).expect("compact");
        assert_eq!(result.before, 3);
        assert_eq!(result.after, 1);
        assert_eq!(result.latest_epoch, Some(2));
        drop(store);

        let reopened = DurableStore::open(fixture.paths.clone()).expect("reopen");
        assert_eq!(reopened.get("checkpoint"), Some("two"));
        assert_eq!(reopened.get("wal-delta"), Some("preserved"));
        assert_eq!(reopened.checkpoint_epoch(), 2);
    }
}
