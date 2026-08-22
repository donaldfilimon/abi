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
    assert_eq!(snapshot.kv.len(), 0);
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
    assert_eq!(snapshot.kv.len(), 0);

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
