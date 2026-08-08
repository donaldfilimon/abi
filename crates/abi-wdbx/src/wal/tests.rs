use super::*;
use std::collections::BTreeMap;

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = abi_foundation::temp_path::temp_file_path(name, "wal");
        std::fs::create_dir_all(&dir).expect("create fixture directory");
        Self { dir }
    }

    fn paths(&self) -> StorePaths {
        StorePaths::new(&self.dir)
    }

    fn wal(&self) -> PathBuf {
        wal_path(&self.paths())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

#[test]
fn append_verify_and_replay_all_mutation_families() {
    let fixture = Fixture::new("abi_wal_replay");
    create_with_epoch(fixture.wal(), 0).expect("create");
    append_kv(fixture.wal(), "agent:abbey", "trained").expect("kv");
    append_vector(fixture.wal(), 1, &[1.0, 0.0, 0.0, 0.0]).expect("vector");
    append_block(fixture.wal(), "abbey", 1, 2, "turn", 1000).expect("block");
    append_spatial(
        fixture.wal(),
        &SpatialRecord {
            id: 7,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            payload: "point".to_string(),
        },
    )
    .expect("spatial");
    append_temporal_node(fixture.wal(), 1, 1000).expect("node");
    append_temporal_edge(fixture.wal(), 1, 2).expect("edge");

    let wal = Wal::read(fixture.wal()).expect("read");
    assert_eq!(wal.base_epoch, 0);
    assert_eq!(wal.len(), 6);
    let mut snapshot = Snapshot::new();
    assert_eq!(wal.replay_onto(&mut snapshot).expect("replay"), 6);
    assert_eq!(snapshot.kv["agent:abbey"], "trained");
    assert_eq!(snapshot.vectors.len(), 1);
    assert_eq!(snapshot.blocks.len(), 1);
    assert_eq!(snapshot.spatial.len(), 1);
    assert_eq!(snapshot.stats.temporal_nodes, 1);
    assert_eq!(snapshot.stats.temporal_edges, 1);
    snapshot.verify_chain().expect("linked chain");
}

#[test]
fn crc_matches_the_frozen_zig_algorithm() {
    assert_eq!(
        crc32_hex(br#"{"type":"kv","key":"k","value":"v"}"#),
        "be3abb8d"
    );
}

#[test]
fn flipped_payload_byte_is_corruption_even_on_the_last_frame() {
    let fixture = Fixture::new("abi_wal_crc");
    append_kv(fixture.wal(), "k1", "v1").expect("append");
    let mut content = std::fs::read_to_string(fixture.wal()).expect("read");
    content = content.replacen("k1", "X1", 1);
    std::fs::write(fixture.wal(), content).expect("tamper");
    assert!(matches!(
        Wal::read(fixture.wal()),
        Err(WalError::Corruption { .. })
    ));
}

#[test]
fn incomplete_last_frame_is_a_soft_torn_tail() {
    let fixture = Fixture::new("abi_wal_torn");
    append_kv(fixture.wal(), "keep", "yes").expect("append");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(fixture.wal())
        .expect("open");
    write!(file, "deadbeef-incomplete").expect("tear");

    let wal = Wal::read(fixture.wal()).expect("verified prefix");
    assert_eq!(wal.len(), 1);
    assert!(wal.torn_tail);
}

#[test]
fn incomplete_mid_log_frame_is_corruption() {
    let content = format!(
        "{WAL_HEADER_PREFIX}\nno-separator\n{} {}\n",
        crc32_hex(br#"{"type":"kv","key":"k","value":"v"}"#),
        r#"{"type":"kv","key":"k","value":"v"}"#
    );
    assert!(matches!(
        Wal::parse(&content),
        Err(WalError::Corruption { line: Some(2), .. })
    ));
}

#[test]
fn create_with_epoch_never_truncates_an_existing_delta() {
    let fixture = Fixture::new("abi_wal_epoch");
    create_with_epoch(fixture.wal(), 42).expect("create");
    append_kv(fixture.wal(), "k", "v").expect("append");
    create_with_epoch(fixture.wal(), 99).expect("must preserve");
    let wal = Wal::read(fixture.wal()).expect("read");
    assert_eq!(wal.base_epoch, 42);
    assert_eq!(wal.len(), 1);
}

#[test]
fn vector_ids_must_continue_the_checkpoint() {
    let fixture = Fixture::new("abi_wal_vector_id");
    create_with_epoch(fixture.wal(), 0).expect("create");
    append_vector(fixture.wal(), 3, &[0.0, 0.0]).expect("append");
    let wal = Wal::read(fixture.wal()).expect("read");
    let mut snapshot = Snapshot {
        vectors: BTreeMap::from([(1, vec![1.0, 0.0])]),
        ..Snapshot::new()
    };
    assert!(matches!(
        wal.replay_onto(&mut snapshot),
        Err(WalError::CorruptVectorId {
            expected: 2,
            found: 3
        })
    ));
}

#[test]
fn legacy_snapshot_without_wal_reports_snapshot_source() {
    let fixture = Fixture::new("abi_wal_legacy_snapshot_source");
    let mut legacy = Snapshot::new();
    legacy
        .kv
        .insert("legacy".to_string(), "readable".to_string());
    crate::persistence::write_snapshot(fixture.paths().index(), &legacy)
        .expect("legacy checkpoint");

    let recovered = recover(&fixture.paths()).expect("recover legacy checkpoint");
    assert_eq!(recovered.source, RecoverySource::Snapshot);
    assert_eq!(recovered.checkpoint_epoch, 0);
    assert_eq!(recovered.frames_applied, 0);
    assert_eq!(
        recovered.snapshot.kv.get("legacy").map(String::as_str),
        Some("readable")
    );
}

#[test]
fn legacy_checkpoint_is_recovered_then_refreshed_on_segment_cutover() {
    let fixture = Fixture::new("abi_wal_legacy_checkpoint");
    let mut legacy = Snapshot::new();
    legacy.kv.insert(
        "stale:multiway:experiment".to_string(),
        "old-export".to_string(),
    );
    crate::persistence::write_snapshot(fixture.paths().index(), &legacy)
        .expect("legacy checkpoint");
    create_with_epoch(fixture.wal(), 0).expect("legacy wal");

    let recovered = recover(&fixture.paths()).expect("recover legacy checkpoint");
    assert_eq!(recovered.source, RecoverySource::Merged);
    assert_eq!(recovered.checkpoint_epoch, 0);
    assert_eq!(recovered.frames_applied, 0);
    assert_eq!(
        recovered
            .snapshot
            .kv
            .get("stale:multiway:experiment")
            .map(String::as_str),
        Some("old-export")
    );

    let mut current = recovered.snapshot;
    current.kv.remove("stale:multiway:experiment");
    current.kv.insert(
        "multiway:experiment:latest".to_string(),
        "current-export".to_string(),
    );
    checkpoint(&fixture.paths(), &current).expect("cut over to segments");
    assert!(
        fixture.paths().index().exists(),
        "the compatibility mirror must be refreshed after publication"
    );
    std::fs::remove_file(fixture.paths().manifest()).expect("simulate manifest loss");
    let after_manifest_loss = recover(&fixture.paths()).expect("recover without manifest");
    assert_eq!(after_manifest_loss.source, RecoverySource::Merged);
    assert_eq!(after_manifest_loss.frames_applied, 0);
    assert_eq!(
        after_manifest_loss
            .snapshot
            .kv
            .get("multiway:experiment:latest")
            .map(String::as_str),
        Some("current-export"),
        "manifest loss must preserve the newly published checkpoint"
    );
    assert!(
        !after_manifest_loss
            .snapshot
            .kv
            .contains_key("stale:multiway:experiment"),
        "manifest loss must not resurrect stale mirror data"
    );
}

#[test]
fn mirror_epoch_preserves_a_newer_wal_delta_without_a_manifest() {
    let fixture = Fixture::new("abi_wal_mirror_epoch");
    let mut first = Snapshot::new();
    first.kv.insert("base".to_string(), "epoch-0".to_string());
    assert_eq!(
        checkpoint(&fixture.paths(), &first).expect("epoch 0 checkpoint"),
        0
    );

    let mut second = Snapshot::new();
    second.kv.insert("base".to_string(), "epoch-1".to_string());
    assert_eq!(
        checkpoint(&fixture.paths(), &second).expect("epoch 1 checkpoint"),
        1
    );
    append_kv(fixture.wal(), "delta", "after-epoch-1").expect("WAL delta");
    std::fs::remove_file(fixture.paths().manifest()).expect("simulate manifest loss");

    let recovered = recover(&fixture.paths()).expect("recover mirror plus matching WAL");
    assert_eq!(recovered.source, RecoverySource::Merged);
    assert_eq!(recovered.checkpoint_epoch, 1);
    assert_eq!(recovered.frames_applied, 1);
    assert_eq!(recovered.snapshot.kv["base"], "epoch-1");
    assert_eq!(recovered.snapshot.kv["delta"], "after-epoch-1");
}

#[test]
fn matching_epoch_wal_merges_onto_checkpoint() {
    let fixture = Fixture::new("abi_wal_recover");
    let mut checkpoint_snapshot = Snapshot::new();
    checkpoint_snapshot
        .kv
        .insert("checkpoint".to_string(), "yes".to_string());
    crate::persistence::flush(&fixture.paths(), &checkpoint_snapshot).expect("checkpoint");
    create_with_epoch(fixture.wal(), 0).expect("wal");
    append_kv(fixture.wal(), "delta", "yes").expect("delta");

    let recovered = recover(&fixture.paths()).expect("recover");
    assert_eq!(recovered.source, RecoverySource::Merged);
    assert_eq!(recovered.checkpoint_epoch, 0);
    assert_eq!(recovered.frames_applied, 1);
    assert_eq!(recovered.snapshot.kv["checkpoint"], "yes");
    assert_eq!(recovered.snapshot.kv["delta"], "yes");
}

#[test]
fn stale_wal_is_removed_without_double_applying() {
    let fixture = Fixture::new("abi_wal_stale");
    let mut first = Snapshot::new();
    first.kv.insert("old".to_string(), "1".to_string());
    crate::persistence::flush(&fixture.paths(), &first).expect("epoch 0");
    create_with_epoch(fixture.wal(), 0).expect("old wal");
    append_kv(fixture.wal(), "must-not-return", "stale").expect("old delta");

    let mut second = Snapshot::new();
    second.kv.insert("new".to_string(), "2".to_string());
    crate::persistence::flush(&fixture.paths(), &second).expect("epoch 1");

    let recovered = recover(&fixture.paths()).expect("recover");
    assert_eq!(recovered.source, RecoverySource::Segment);
    assert_eq!(recovered.checkpoint_epoch, 1);
    assert_eq!(recovered.snapshot.kv, second.kv);
    assert!(!fixture.wal().exists());
}

#[test]
fn checkpoint_resets_wal_to_the_new_epoch() {
    let fixture = Fixture::new("abi_wal_checkpoint");
    create_with_epoch(fixture.wal(), 0).expect("wal");
    append_kv(fixture.wal(), "delta", "value").expect("append");
    let mut snapshot = Snapshot::new();
    snapshot.kv.insert("folded".to_string(), "yes".to_string());

    let epoch = checkpoint(&fixture.paths(), &snapshot).expect("checkpoint");
    assert_eq!(epoch, 0);
    let wal = Wal::read(fixture.wal()).expect("fresh wal");
    assert_eq!(wal.base_epoch, 0);
    assert_eq!(wal.len(), 0);
}
