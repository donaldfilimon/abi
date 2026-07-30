//! Loading a store snapshot from its segments.
//!
//! Ported from the load half of `src/features/wdbx/segments.zig` and `store.zig`.
//!
//! ## Segments are checkpoints, not deltas
//!
//! `SegmentStore.flush` writes the **entire store** into a new epoch, and
//! `loadLatest` reads **only the highest active epoch**. Older epochs are retained
//! checkpoints, reclaimed by compaction. So [`load`] opens one segment, not all of
//! them.
//!
//! This is worth stating plainly because getting it wrong fails *asymmetrically*
//! and so hides itself. An earlier version of this module layered every active
//! epoch. `kv` and `vector` records are keyed, so re-applying 301 copies of the
//! same checkpoint collapsed back to the right values and looked correct. `block`
//! records are an unkeyed ordered chain, so they appended: the real store's ~166
//! blocks became 50,148, and chain verification would have failed on repeated
//! sequence numbers. Only the block count revealed it.
//!
//! [`load_merging_all_epochs`] keeps the layered behaviour for the recovery case
//! where the latest checkpoint is unreadable, under a name that cannot be mistaken
//! for the normal path.

use crate::format::{
    BlockRecord, FormatError, Manifest, Record, Result, SpatialRecord, StorePaths, TemporalRecord,
    read_segment,
};
use std::collections::BTreeMap;

/// Counters describing a snapshot.
///
/// **Not the `wdbx_stats` wire shape.** This is an internal type. The frozen MCP
/// tool emits a flat line
///
/// ```text
/// kv=256 vectors=654 blocks=327 spatial=0 dims=32 backend=metal source=mcp-store
/// ```
///
/// and `abi wdbx db verify` emits a different one again. The store's JSON form
/// (`src/features/wdbx/store.zig:351`) carries ten keys in this order:
/// `kv_entries`, `vectors`, `blocks`, `spatial_records`, `temporal_nodes`,
/// `temporal_edges`, `vector_dimensions`, `next_vector_id`, `backend`, `mode`.
///
/// The first six line up with the fields below; the last four are not counters
/// and are supplied by the MCP layer at step 9, which owns the wire format.
/// `epochs_loaded` and `truncated_segments` are additions with no wire
/// counterpart. Captured goldens: `tests/golden/mcp-tool-calls.jsonl` and
/// `tests/golden/wdbx-db-verify.txt`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapshotStats {
    /// Distinct key/value entries after shadowing.
    pub kv_entries: usize,
    /// Distinct vectors after shadowing.
    pub vectors: usize,
    /// Audit-chain blocks.
    pub blocks: usize,
    /// Spatial records.
    pub spatial_records: usize,
    /// Temporal graph nodes.
    pub temporal_nodes: usize,
    /// Temporal graph edges.
    pub temporal_edges: usize,
    /// Epochs replayed.
    pub epochs_loaded: usize,
    /// Segments whose final line was torn by an interrupted write.
    pub truncated_segments: usize,
}

/// The state reconstructed from a store's active segments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    /// Key/value entries, keyed by key. Later epochs have already won.
    pub kv: BTreeMap<String, String>,
    /// Vectors, keyed by id. Later epochs have already won.
    pub vectors: BTreeMap<u64, Vec<f32>>,
    /// Audit-chain blocks in replay order.
    pub blocks: Vec<BlockRecord>,
    /// Spatial records, keyed by id.
    pub spatial: BTreeMap<u64, SpatialRecord>,
    /// Temporal records in replay order.
    pub temporal: Vec<TemporalRecord>,
    /// Counters.
    pub stats: SnapshotStats,
}

impl Snapshot {
    /// An empty snapshot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one record, letting it shadow any earlier value for the same key.
    pub fn apply(&mut self, record: Record) {
        match record {
            Record::Kv(kv) => {
                self.kv.insert(kv.key, kv.value);
            }
            Record::Vector(vector) => {
                self.vectors.insert(vector.id, vector.values);
            }
            Record::Block(block) => self.blocks.push(block),
            Record::Spatial(spatial) => {
                self.spatial.insert(spatial.id, spatial);
            }
            Record::Temporal(temporal) => self.temporal.push(temporal),
        }
    }

    /// Recompute [`Snapshot::stats`] from the current contents.
    pub fn recount(&mut self) {
        self.stats.kv_entries = self.kv.len();
        self.stats.vectors = self.vectors.len();
        self.stats.blocks = self.blocks.len();
        self.stats.spatial_records = self.spatial.len();
        self.stats.temporal_nodes = self
            .temporal
            .iter()
            .filter(|record| record.kind == crate::format::TemporalKind::Node)
            .count();
        self.stats.temporal_edges = self
            .temporal
            .iter()
            .filter(|record| record.kind == crate::format::TemporalKind::Edge)
            .count();
    }

    /// The dimensionality of the stored vectors, if they agree.
    ///
    /// `None` when there are no vectors, or when they disagree — which is a
    /// corrupt store rather than a valid heterogeneous one, so the caller decides
    /// what to do rather than having a value invented for it.
    #[must_use]
    pub fn vector_dimensions(&self) -> Option<usize> {
        let mut dims = None;
        for values in self.vectors.values() {
            match dims {
                None => dims = Some(values.len()),
                Some(existing) if existing == values.len() => {}
                Some(_) => return None,
            }
        }
        dims
    }

    /// The highest vector id seen, or `None` when empty.
    #[must_use]
    pub fn max_vector_id(&self) -> Option<u64> {
        self.vectors.keys().copied().next_back()
    }

    /// Whether the audit chain links correctly from genesis.
    ///
    /// Each block's `prev_hash` must equal its predecessor's `hash`, and the first
    /// must be the genesis sentinel. Reports the first break rather than a bare
    /// bool, so recovery can say *where* the chain diverged.
    ///
    /// Blocks are checked in replay order, which is how they were written.
    pub fn verify_chain(&self) -> std::result::Result<(), ChainBreak> {
        let mut expected = crate::format::Hash::GENESIS;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.prev_hash != expected {
                return Err(ChainBreak {
                    index,
                    sequence: block.sequence,
                    expected_prev: expected,
                    found_prev: block.prev_hash,
                });
            }
            expected = block.hash;
        }
        Ok(())
    }

    /// Verify predecessor links and recompute every block's content hash.
    ///
    /// [`Self::verify_chain`] intentionally remains the historical link-only
    /// compatibility check used by the snapshot loader. Local security surfaces
    /// such as REST `/verify` use this stricter invariant.
    pub fn verify_chain_strict(&self) -> std::result::Result<(), ChainIntegrityBreak> {
        let mut expected_prev = crate::format::Hash::GENESIS;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.prev_hash != expected_prev {
                return Err(ChainIntegrityBreak {
                    kind: ChainIntegrityKind::PreviousHash,
                    index,
                    sequence: block.sequence,
                    expected: expected_prev,
                    found: block.prev_hash,
                });
            }
            let expected_hash = crate::wal::compute_block_hash(
                block.prev_hash,
                block.timestamp_ms,
                block.sequence,
                &block.profile,
                &block.metadata,
            );
            if block.hash != expected_hash {
                return Err(ChainIntegrityBreak {
                    kind: ChainIntegrityKind::BlockHash,
                    index,
                    sequence: block.sequence,
                    expected: expected_hash,
                    found: block.hash,
                });
            }
            expected_prev = block.hash;
        }
        Ok(())
    }
}

/// Where an audit chain failed to link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainBreak {
    /// Position in the replayed block list.
    pub index: usize,
    /// The block's own sequence number.
    pub sequence: u64,
    /// The `prev_hash` the chain required.
    pub expected_prev: crate::format::Hash,
    /// The `prev_hash` the block carried.
    pub found_prev: crate::format::Hash,
}

impl std::fmt::Display for ChainBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "audit chain breaks at block {} (sequence {}): prev_hash is {} but the previous block hashes to {}",
            self.index,
            self.sequence,
            self.found_prev.to_hex(),
            self.expected_prev.to_hex()
        )
    }
}

impl std::error::Error for ChainBreak {}

/// Which strict audit-chain invariant failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainIntegrityKind {
    /// A block does not point to the preceding block.
    PreviousHash,
    /// A block hash does not authenticate its stored contents.
    BlockHash,
}

/// Where strict chain authentication failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainIntegrityBreak {
    /// Broken invariant.
    pub kind: ChainIntegrityKind,
    /// Position in the replayed block list.
    pub index: usize,
    /// The block's own sequence number.
    pub sequence: u64,
    /// Hash required by the invariant.
    pub expected: crate::format::Hash,
    /// Hash carried by the block.
    pub found: crate::format::Hash,
}

impl std::fmt::Display for ChainIntegrityBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ChainIntegrityKind::PreviousHash => write!(
                f,
                "audit chain breaks at block {} (sequence {}): prev_hash is {} but the previous block hashes to {}",
                self.index,
                self.sequence,
                self.found.to_hex(),
                self.expected.to_hex()
            ),
            ChainIntegrityKind::BlockHash => write!(
                f,
                "audit block {} (sequence {}) hashes to {} but carries {}",
                self.index,
                self.sequence,
                self.expected.to_hex(),
                self.found.to_hex()
            ),
        }
    }
}

impl std::error::Error for ChainIntegrityBreak {}

/// Load a store from its latest checkpoint.
///
/// Reads the single highest active epoch, matching Zig's `loadLatest`. Only epochs
/// listed as `active` are considered: a segment file whose epoch is absent has been
/// retired by compaction, and reading it would resurrect deleted data.
///
/// An empty store, or a manifest whose latest segment is missing from disk, yields
/// an empty snapshot rather than an error — the latter is what a crash between the
/// manifest write and the segment write leaves behind.
pub fn load(paths: &StorePaths) -> Result<Snapshot> {
    let manifest = paths.read_manifest()?;
    load_latest(paths, &manifest)
}

/// Load the highest active epoch named by `manifest`.
pub fn load_latest(paths: &StorePaths, manifest: &Manifest) -> Result<Snapshot> {
    let mut snapshot = Snapshot::new();

    let Some(latest) = manifest.active.iter().copied().max() else {
        return Ok(snapshot);
    };
    let path = paths.segment(latest);
    if !path.is_file() {
        // Manifest written, segment not yet flushed.
        return Ok(snapshot);
    }

    let segment = read_segment(latest, &path)?;
    if segment.truncated_tail {
        snapshot.stats.truncated_segments += 1;
    }
    for record in segment.records {
        snapshot.apply(record);
    }
    snapshot.stats.epochs_loaded = 1;
    snapshot.recount();
    Ok(snapshot)
}

/// Recover the newest readable active checkpoint without merging epochs.
///
/// Active epochs are tried from newest to oldest. Missing files and corrupt
/// checkpoints are skipped until one complete checkpoint loads. Because every
/// epoch is a full snapshot, returning exactly one preserves deletions and avoids
/// duplicating the audit chain.
///
/// If active segment files exist but all are corrupt, the newest corruption is
/// returned rather than silently inventing an empty store.
pub fn load_newest_valid(paths: &StorePaths, manifest: &Manifest) -> Result<Snapshot> {
    load_newest_valid_with_epoch(paths, manifest).map(|(snapshot, _)| snapshot)
}

/// Recover the newest readable checkpoint and report the epoch selected.
///
/// `None` means no manifest-listed segment file exists. This is the form used
/// by epoch-gated WAL recovery.
pub fn load_newest_valid_with_epoch(
    paths: &StorePaths,
    manifest: &Manifest,
) -> Result<(Snapshot, Option<u64>)> {
    let mut epochs = manifest.active.clone();
    epochs.sort_unstable_by(|left, right| right.cmp(left));

    let mut newest_error = None;
    for epoch in epochs {
        let path = paths.segment(epoch);
        if !path.is_file() {
            continue;
        }

        match read_segment(epoch, &path) {
            Ok(segment) => {
                let mut snapshot = Snapshot::new();
                if segment.truncated_tail {
                    snapshot.stats.truncated_segments = 1;
                }
                for record in segment.records {
                    snapshot.apply(record);
                }
                snapshot.stats.epochs_loaded = 1;
                snapshot.recount();
                return Ok((snapshot, Some(epoch)));
            }
            Err(error) => {
                if newest_error.is_none() {
                    newest_error = Some(error);
                }
            }
        }
    }

    newest_error.map_or_else(|| Ok((Snapshot::new(), None)), Err)
}

/// Which checkpoint layout supplied a recovered snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointSource {
    /// No checkpoint exists.
    Empty,
    /// The pre-manifest single-file Zig checkpoint.
    Legacy {
        /// Epoch recorded by a Rust mirror sidecar, or 0 for Zig snapshots.
        epoch: u64,
    },
    /// A manifest-listed immutable segment.
    Segment {
        /// Published segment epoch.
        epoch: u64,
    },
}

impl CheckpointSource {
    /// Epoch used to gate WAL replay.
    #[must_use]
    pub const fn epoch(self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Legacy { epoch } | Self::Segment { epoch } => epoch,
        }
    }
}

/// Load the authoritative checkpoint, including the legacy Zig single-file
/// layout only when no active manifest epochs exist.
pub fn load_checkpoint_with_source(paths: &StorePaths) -> Result<(Snapshot, CheckpointSource)> {
    let manifest = paths.read_manifest()?;
    let (mut snapshot, epoch) = load_newest_valid_with_epoch(paths, &manifest)?;
    if let Some(epoch) = epoch {
        return Ok((snapshot, CheckpointSource::Segment { epoch }));
    }
    let legacy = paths.index();
    if !manifest.active.is_empty() || !legacy.is_file() {
        return Ok((snapshot, CheckpointSource::Empty));
    }

    let segment = read_segment(0, &legacy)?;
    if segment.truncated_tail {
        snapshot.stats.truncated_segments = 1;
    }
    for record in segment.records {
        snapshot.apply(record);
    }
    snapshot.stats.epochs_loaded = 1;
    snapshot.recount();
    Ok((
        snapshot,
        CheckpointSource::Legacy {
            epoch: paths.read_mirror_epoch()?,
        },
    ))
}

/// Layer **every** active segment into one snapshot.
///
/// **Not the normal load path** — see the module docs. Segments are full
/// checkpoints, so this re-applies the same data once per epoch: keyed records
/// collapse harmlessly but the block chain accumulates duplicates. Use it only to
/// recover when the latest checkpoint is unreadable and older ones must be salvaged.
pub fn load_merging_all_epochs(paths: &StorePaths, manifest: &Manifest) -> Result<Snapshot> {
    let mut snapshot = Snapshot::new();

    // Ascending epoch order is what makes later writes shadow earlier ones.
    // Manifest::parse already sorts, but relying on that implicitly would make
    // this correct only by accident.
    let mut epochs = manifest.active.clone();
    epochs.sort_unstable();

    for epoch in epochs {
        let path = paths.segment(epoch);
        if !path.is_file() {
            // Crash between manifest and segment write; skip rather than fail.
            continue;
        }
        let segment = read_segment(epoch, &path)?;
        if segment.truncated_tail {
            snapshot.stats.truncated_segments += 1;
        }
        for record in segment.records {
            snapshot.apply(record);
        }
        snapshot.stats.epochs_loaded += 1;
    }

    snapshot.recount();
    Ok(snapshot)
}

/// Discover segments on disk without consulting the manifest.
///
/// For recovery only, when the manifest is lost. This **will** resurrect segments
/// that compaction had already retired, so it is not the normal path — hence the
/// separate name rather than a flag on [`load`]. Loads the highest epoch found.
pub fn load_ignoring_manifest(paths: &StorePaths) -> Result<Snapshot> {
    let prefix = format!("{}.seg.", paths.base);
    let mut epochs = Vec::new();

    let entries = std::fs::read_dir(&paths.dir).map_err(|e| FormatError::Io {
        path: paths.dir.clone(),
        message: e.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| FormatError::Io {
            path: paths.dir.clone(),
            message: e.to_string(),
        })?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(number) = rest.strip_suffix(".jsonl") else {
            continue;
        };
        if let Ok(epoch) = number.parse::<u64>() {
            epochs.push(epoch);
        }
    }

    // Numeric, not lexicographic: directory order would put seg.10 before seg.2
    // and pick the wrong "latest".
    epochs.sort_unstable();
    load_newest_valid(
        paths,
        &Manifest {
            next_epoch: epochs.last().map_or(0, |last| last + 1),
            active: epochs,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Hash, KvRecord, SEGMENT_HEADER, VectorRecord};
    use std::path::PathBuf;

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = abi_foundation::temp_path::temp_file_path(name, "store");
            std::fs::create_dir_all(&dir).expect("create store dir");
            Self { dir }
        }

        fn paths(&self) -> StorePaths {
            StorePaths::new(&self.dir)
        }

        fn write_segment(&self, epoch: u64, records: &[&str]) {
            let mut content = String::from(SEGMENT_HEADER);
            content.push('\n');
            for record in records {
                content.push_str(record);
                content.push('\n');
            }
            std::fs::write(self.paths().segment(epoch), content).expect("write segment");
        }

        fn write_manifest(&self, next_epoch: u64, active: &[u64]) {
            let manifest = Manifest {
                next_epoch,
                active: active.to_vec(),
            };
            std::fs::write(self.paths().manifest(), manifest.render()).expect("write manifest");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    fn kv(key: &str, value: &str) -> String {
        serde_json::json!({"type":"kv","key":key,"value":value}).to_string()
    }

    #[test]
    fn an_empty_store_loads_as_an_empty_snapshot() {
        let fixture = Fixture::new("abi_wdbx_empty");
        let snapshot = load(&fixture.paths()).expect("loads");
        assert!(snapshot.kv.is_empty());
        assert_eq!(snapshot.stats, SnapshotStats::default());
    }

    #[test]
    fn only_the_latest_checkpoint_is_loaded() {
        // Segments are full checkpoints, so the latest epoch alone IS the store.
        let fixture = Fixture::new("abi_wdbx_latest");
        fixture.write_segment(0, &[&kv("completion:1", "old"), &kv("gone", "dropped")]);
        fixture.write_segment(1, &[&kv("completion:1", "new")]);
        fixture.write_manifest(2, &[0, 1]);

        let snapshot = load(&fixture.paths()).expect("loads");
        assert_eq!(
            snapshot.kv.get("completion:1").map(String::as_str),
            Some("new")
        );
        assert!(
            !snapshot.kv.contains_key("gone"),
            "a key absent from the latest checkpoint was deleted, and must stay gone"
        );
        assert_eq!(snapshot.stats.kv_entries, 1);
        assert_eq!(snapshot.stats.epochs_loaded, 1);
    }

    #[test]
    fn the_latest_epoch_is_chosen_numerically_not_lexicographically() {
        // Lexicographic order puts seg.10 before seg.2, picking epoch 2 as latest.
        let fixture = Fixture::new("abi_wdbx_order");
        fixture.write_segment(2, &[&kv("k", "from-2")]);
        fixture.write_segment(10, &[&kv("k", "from-10")]);
        fixture.write_manifest(11, &[10, 2]);

        let snapshot = load(&fixture.paths()).expect("loads");
        assert_eq!(
            snapshot.kv.get("k").map(String::as_str),
            Some("from-10"),
            "epoch 10 is the latest, not epoch 2"
        );
    }

    #[test]
    fn loading_does_not_duplicate_the_block_chain_across_epochs() {
        // The regression that a layered loader hid: keyed records collapse, so
        // only the unkeyed block chain reveals the duplication. Three identical
        // checkpoints must still yield one block, not three.
        let fixture = Fixture::new("abi_wdbx_no_dup_blocks");
        let timestamp_ms = 1_753_000_000_000_i64;
        let hash = crate::wal::compute_block_hash(Hash::GENESIS, timestamp_ms, 0, "abi", "");
        let genesis = serde_json::json!({
            "type": "block",
            "hash": hash.bytes(),
            "prev_hash": vec![0u8; 32],
            "timestamp_ms": timestamp_ms,
            "sequence": 0,
            "profile": "abi",
            "query_id": 1,
            "response_id": 2,
            "metadata": ""
        })
        .to_string();
        for epoch in 0..3 {
            fixture.write_segment(epoch, &[&kv("k", "v"), &genesis]);
        }
        fixture.write_manifest(3, &[0, 1, 2]);

        let snapshot = load(&fixture.paths()).expect("loads");
        assert_eq!(
            snapshot.stats.blocks, 1,
            "the block chain must not accumulate one copy per checkpoint"
        );
        snapshot
            .verify_chain()
            .expect("a non-duplicated chain verifies");
    }

    #[test]
    fn merging_all_epochs_is_available_for_recovery_and_does_duplicate() {
        // Documents the behaviour of the recovery path, so its cost is explicit.
        let fixture = Fixture::new("abi_wdbx_merge_all");
        fixture.write_segment(0, &[&kv("a", "1")]);
        fixture.write_segment(1, &[&kv("b", "2")]);
        fixture.write_manifest(2, &[0, 1]);

        let manifest = fixture.paths().read_manifest().expect("manifest");
        let snapshot = load_merging_all_epochs(&fixture.paths(), &manifest).expect("recovery load");
        assert_eq!(snapshot.stats.epochs_loaded, 2);
        // Salvages keys from a retired checkpoint that `load` would not see.
        assert!(snapshot.kv.contains_key("a"));
        assert!(snapshot.kv.contains_key("b"));
    }

    #[test]
    fn a_segment_not_listed_as_active_is_not_read() {
        // Reading a retired segment would resurrect deleted data.
        let fixture = Fixture::new("abi_wdbx_retired");
        fixture.write_segment(0, &[&kv("kept", "yes")]);
        fixture.write_segment(1, &[&kv("retired", "must-not-load")]);
        fixture.write_manifest(2, &[0]);

        let snapshot = load(&fixture.paths()).expect("loads");
        assert!(snapshot.kv.contains_key("kept"));
        assert!(
            !snapshot.kv.contains_key("retired"),
            "a collected segment must not be replayed"
        );
    }

    #[test]
    fn recovery_mode_sees_a_segment_the_manifest_omits() {
        // With a lost or stale manifest, the newest checkpoint on disk may not be
        // listed. Recovery discovers it; the normal path deliberately does not.
        let fixture = Fixture::new("abi_wdbx_recovery");
        fixture.write_segment(0, &[&kv("a", "1")]);
        fixture.write_segment(1, &[&kv("b", "2")]);
        fixture.write_manifest(2, &[0]);

        let normal = load(&fixture.paths()).expect("loads");
        assert!(normal.kv.contains_key("a"));
        assert!(
            !normal.kv.contains_key("b"),
            "an unlisted segment must not load on the normal path"
        );

        let recovered = load_ignoring_manifest(&fixture.paths()).expect("recovery load");
        assert!(
            recovered.kv.contains_key("b"),
            "recovery must find the newest checkpoint on disk"
        );
    }

    #[test]
    fn a_missing_active_segment_is_skipped_not_fatal() {
        // A crash between the manifest write and the segment write leaves this.
        // Failing here would lose every other segment too.
        let fixture = Fixture::new("abi_wdbx_missing_seg");
        fixture.write_segment(0, &[&kv("present", "yes")]);
        fixture.write_manifest(3, &[0, 1, 2]);

        // The latest active epoch (2) has no file, so the load yields empty
        // rather than failing.
        let snapshot = load(&fixture.paths()).expect("loads despite gaps");
        assert_eq!(snapshot.stats.epochs_loaded, 0);
        assert!(snapshot.kv.is_empty());

        // Recovery can still salvage the checkpoint that does exist.
        let manifest = fixture.paths().read_manifest().expect("manifest");
        let salvaged = load_merging_all_epochs(&fixture.paths(), &manifest).expect("recovery load");
        assert!(salvaged.kv.contains_key("present"));
    }

    #[test]
    fn newest_valid_recovery_falls_back_without_merging_checkpoints() {
        let fixture = Fixture::new("abi_wdbx_newest_valid");
        fixture.write_segment(0, &[&kv("old", "preserved")]);
        fixture.write_segment(1, &[&kv("new", "must-not-be-merged")]);
        std::fs::write(fixture.paths().segment(2), "corrupt\n").expect("write corrupt segment");
        fixture.write_manifest(3, &[0, 1, 2]);

        let manifest = fixture.paths().read_manifest().expect("manifest");
        let recovered = load_newest_valid(&fixture.paths(), &manifest).expect("fallback loads");
        assert_eq!(recovered.stats.epochs_loaded, 1);
        assert_eq!(
            recovered.kv.get("new").map(String::as_str),
            Some("must-not-be-merged")
        );
        assert!(
            !recovered.kv.contains_key("old"),
            "full checkpoints must not be merged during fallback"
        );
    }

    #[test]
    fn newest_valid_recovery_reports_corruption_when_no_checkpoint_survives() {
        let fixture = Fixture::new("abi_wdbx_no_valid");
        std::fs::write(fixture.paths().segment(0), "old-corrupt\n").expect("write");
        std::fs::write(fixture.paths().segment(1), "new-corrupt\n").expect("write");
        fixture.write_manifest(2, &[0, 1]);

        let manifest = fixture.paths().read_manifest().expect("manifest");
        let error = load_newest_valid(&fixture.paths(), &manifest).expect_err("must fail");
        match error {
            FormatError::InvalidHeader { path, .. } => {
                assert_eq!(path, fixture.paths().segment(1));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn a_torn_segment_tail_is_counted() {
        let fixture = Fixture::new("abi_wdbx_torn");
        let mut content = String::from(SEGMENT_HEADER);
        content.push('\n');
        content.push_str(&kv("good", "1"));
        content.push('\n');
        content.push_str("{\"type\":\"kv\",\"key\":\"tor");
        std::fs::write(fixture.paths().segment(0), content).expect("write");
        fixture.write_manifest(1, &[0]);

        let snapshot = load(&fixture.paths()).expect("loads");
        assert_eq!(snapshot.stats.kv_entries, 1);
        assert_eq!(snapshot.stats.truncated_segments, 1);
    }

    #[test]
    fn vectors_shadow_by_id_and_report_dimensions() {
        let mut snapshot = Snapshot::new();
        snapshot.apply(Record::Vector(VectorRecord {
            id: 1,
            values: vec![1.0, 2.0, 3.0],
        }));
        snapshot.apply(Record::Vector(VectorRecord {
            id: 1,
            values: vec![9.0, 9.0, 9.0],
        }));
        snapshot.apply(Record::Vector(VectorRecord {
            id: 2,
            values: vec![0.0, 0.0, 0.0],
        }));
        snapshot.recount();

        assert_eq!(snapshot.stats.vectors, 2);
        assert_eq!(snapshot.vectors[&1], vec![9.0, 9.0, 9.0]);
        assert_eq!(snapshot.vector_dimensions(), Some(3));
        assert_eq!(snapshot.max_vector_id(), Some(2));
    }

    #[test]
    fn disagreeing_vector_dimensions_report_none_rather_than_guessing() {
        let mut snapshot = Snapshot::new();
        snapshot.apply(Record::Vector(VectorRecord {
            id: 1,
            values: vec![1.0, 2.0],
        }));
        snapshot.apply(Record::Vector(VectorRecord {
            id: 2,
            values: vec![1.0, 2.0, 3.0],
        }));
        assert_eq!(snapshot.vector_dimensions(), None);
    }

    #[test]
    fn no_vectors_means_no_dimensions_and_no_max_id() {
        let snapshot = Snapshot::new();
        assert_eq!(snapshot.vector_dimensions(), None);
        assert_eq!(snapshot.max_vector_id(), None);
    }

    fn block(sequence: u64, _hash: u8, prev: Hash) -> BlockRecord {
        let timestamp_ms = 1_753_000_000_000 + i64::try_from(sequence).unwrap_or(0);
        let profile = "abi".to_string();
        let metadata = String::new();
        BlockRecord {
            hash: crate::wal::compute_block_hash(prev, timestamp_ms, sequence, &profile, &metadata),
            prev_hash: prev,
            timestamp_ms,
            sequence,
            profile,
            query_id: sequence * 2,
            response_id: sequence * 2 + 1,
            metadata,
        }
    }

    #[test]
    fn a_well_formed_chain_verifies_from_genesis() {
        let mut snapshot = Snapshot::new();
        let first = block(0, 1, Hash::GENESIS);
        let second = block(1, 2, first.hash);
        let third = block(2, 3, second.hash);
        for b in [first, second, third] {
            snapshot.apply(Record::Block(b));
        }
        assert!(snapshot.verify_chain().is_ok());
    }

    #[test]
    fn an_empty_chain_verifies() {
        assert!(Snapshot::new().verify_chain().is_ok());
    }

    #[test]
    fn a_chain_not_starting_at_genesis_is_a_break() {
        let mut snapshot = Snapshot::new();
        snapshot.apply(Record::Block(block(0, 1, Hash([7; 32]))));
        let err = snapshot.verify_chain().expect_err("must not verify");
        assert_eq!(err.index, 0);
        assert_eq!(err.expected_prev, Hash::GENESIS);
    }

    #[test]
    fn a_break_reports_where_it_happened() {
        let mut snapshot = Snapshot::new();
        let first = block(0, 1, Hash::GENESIS);
        let second = block(1, 2, first.hash);
        // Third points at the wrong predecessor.
        let third = block(2, 3, Hash([99; 32]));
        for b in [first, second.clone(), third] {
            snapshot.apply(Record::Block(b));
        }
        let err = snapshot.verify_chain().expect_err("must not verify");
        assert_eq!(err.index, 2);
        assert_eq!(err.sequence, 2);
        assert_eq!(err.expected_prev, second.hash);
        assert!(err.to_string().contains("sequence 2"));
    }

    #[test]
    fn a_tampered_block_hash_is_rejected_even_when_links_match() {
        let mut snapshot = Snapshot::new();
        let mut first = block(0, 1, Hash::GENESIS);
        first.hash = Hash([42; 32]);
        snapshot.apply(Record::Block(first));
        assert!(snapshot.verify_chain().is_ok(), "links alone still verify");
        let err = snapshot
            .verify_chain_strict()
            .expect_err("strict verification must fail");
        assert_eq!(err.kind, ChainIntegrityKind::BlockHash);
        assert_eq!(err.index, 0);
    }

    #[test]
    fn stats_count_each_record_kind() {
        let mut snapshot = Snapshot::new();
        snapshot.apply(Record::Kv(KvRecord {
            key: "k".to_string(),
            value: "v".to_string(),
        }));
        snapshot.apply(Record::Vector(VectorRecord {
            id: 1,
            values: vec![0.0],
        }));
        snapshot.apply(Record::Block(block(0, 1, Hash::GENESIS)));
        snapshot.apply(Record::Spatial(SpatialRecord {
            id: 1,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            payload: String::new(),
        }));
        snapshot.apply(Record::Temporal(TemporalRecord {
            kind: crate::format::TemporalKind::Node,
            fields: serde_json::Map::new(),
        }));
        snapshot.apply(Record::Temporal(TemporalRecord {
            kind: crate::format::TemporalKind::Edge,
            fields: serde_json::Map::new(),
        }));
        snapshot.recount();

        assert_eq!(snapshot.stats.kv_entries, 1);
        assert_eq!(snapshot.stats.vectors, 1);
        assert_eq!(snapshot.stats.blocks, 1);
        assert_eq!(snapshot.stats.spatial_records, 1);
        assert_eq!(snapshot.stats.temporal_nodes, 1);
        assert_eq!(snapshot.stats.temporal_edges, 1);
    }
}
