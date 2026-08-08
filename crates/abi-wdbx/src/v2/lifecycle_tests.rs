use super::*;
use crate::{DurableStore, SpatialRecord};
use std::io::Write as _;

fn scratch(name: &str) -> StorePaths {
    StorePaths::new(abi_foundation::temp_path::temp_file_path(name, "store"))
}

#[test]
fn read_only_v1_open_does_not_remove_a_stale_wal() {
    let paths = scratch("wdbx-v2-read-only");
    let mut first = Snapshot::new();
    first.kv.insert("old".into(), "one".into());
    crate::persistence::flush(&paths, &first).unwrap();
    crate::wal::create_with_epoch(wal_path(&paths), 0).unwrap();
    crate::wal::append_kv(wal_path(&paths), "stale", "delta").unwrap();
    let mut second = Snapshot::new();
    second.kv.insert("new".into(), "two".into());
    crate::persistence::flush(&paths, &second).unwrap();
    let before = std::fs::read(wal_path(&paths)).unwrap();

    let VersionedSnapshot::V1(snapshot) = open_versioned_read_only(&paths).unwrap() else {
        panic!("expected v1");
    };
    assert_eq!(snapshot.kv, second.kv);
    assert_eq!(std::fs::read(wal_path(&paths)).unwrap(), before);
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn writable_open_migrates_every_v1_family_and_keeps_a_byte_exact_backup() {
    let paths = scratch("wdbx-v2-migration");
    let mut store = DurableStore::open(paths.clone()).unwrap();
    store.put("key", "value").unwrap();
    let query = store.put_vector(&[1.0, 0.0]).unwrap();
    let response = store.put_vector(&[0.0, 1.0]).unwrap();
    store
        .add_block("abbey", query, response, "meta", 42)
        .unwrap();
    store
        .put_spatial(SpatialRecord {
            id: 9,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            payload: "point".into(),
        })
        .unwrap();
    store.add_temporal_node(9, 42).unwrap();
    store.add_temporal_edge(9, 10).unwrap();
    store.checkpoint().unwrap();
    drop(store);
    let originals = v1_objects(&paths).unwrap();
    let original_bytes: BTreeMap<_, _> = originals
        .iter()
        .map(|path| {
            (
                path.file_name().unwrap().to_owned(),
                std::fs::read(path).unwrap(),
            )
        })
        .collect();

    let (store, report) = open_versioned_writable(&paths).unwrap();
    assert!(report.migrated);
    let backup = report.backup.as_ref().unwrap();
    for (name, bytes) in original_bytes {
        assert_eq!(std::fs::read(backup.join(&name)).unwrap(), bytes);
        assert_eq!(std::fs::read(paths.dir.join(name)).unwrap(), bytes);
    }
    let snapshot = store.snapshot();
    assert_eq!(snapshot.get("key").unwrap().preferred.value, "value");
    assert_eq!(
        snapshot
            .get_vector(RecordId::Legacy(query))
            .unwrap()
            .preferred
            .value,
        [1.0, 0.0]
    );
    assert_eq!(snapshot.audit_blocks().count(), 1);
    assert!(snapshot.get_spatial(RecordId::Legacy(9)).is_some());
    assert_eq!(snapshot.temporal.len(), 2);
    assert!(matches!(
        migration_status(&paths).unwrap(),
        MigrationStatus::V2 { .. }
    ));
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn rekey_refuses_active_writers_then_atomically_activates_verified_generation() {
    let paths = scratch("wdbx-v2-rekey");
    let (mut source, source_report) = open_versioned_writable(&paths).unwrap();
    source
        .commit(vec![V2Mutation::PutKv {
            key: "rekey-secret".into(),
            value: "preserved".into(),
        }])
        .unwrap();
    let old_journal = std::fs::read_dir(source_report.generation.join("journals"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let old_bytes = std::fs::read(&old_journal).unwrap();
    let (encryption, signing, verifying) = super::super::generate_key_material();
    let replacement =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let Err(error) = rekey_versioned(&paths, replacement.clone()) else {
        panic!("an active writer lease must block rekey");
    };
    assert!(error.to_string().contains("active writer lease"));
    assert!(!source_report.generation.join("MAINTENANCE").exists());
    drop(source);

    let (rekeyed, report) = rekey_versioned(&paths, replacement).unwrap();
    assert_eq!(report.previous_generation, source_report.generation);
    assert!(report.previous_generation.is_dir());
    assert!(report.generation.is_dir());
    assert_ne!(report.previous_generation, report.generation);
    assert_eq!(std::fs::read(old_journal).unwrap(), old_bytes);
    assert_eq!(
        rekeyed
            .snapshot()
            .get("rekey-secret")
            .unwrap()
            .preferred
            .value,
        "preserved"
    );
    let segment = std::fs::read_dir(report.generation.join("segments"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let encoded = std::fs::read(segment).unwrap();
    assert!(
        !encoded
            .windows(b"preserved".len())
            .any(|window| window == b"preserved")
    );
    assert!(matches!(
        migration_status(&paths).unwrap(),
        MigrationStatus::V2 { generation, .. } if generation == report.generation
    ));
    drop(rekeyed);
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn confirmed_gc_requires_no_live_writer_and_keeps_one_covering_segment() {
    let paths = scratch("wdbx-v2-gc");
    let (mut store, opened) = open_versioned_writable(&paths).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "gc-key".into(),
            value: "retained".into(),
        }])
        .unwrap();
    store.compact().unwrap();
    assert!(gc_versioned(&paths, false).is_err());
    let Err(error) = gc_versioned(&paths, true) else {
        panic!("a live writer must prevent garbage collection");
    };
    assert!(error.to_string().contains("active writer lease"));
    drop(store);

    let report = gc_versioned(&paths, true).unwrap();
    assert_eq!(report.generation, opened.generation);
    assert!(report.removed_objects >= 3);
    assert!(report.removed_bytes > 0);
    assert_eq!(
        std::fs::read_dir(report.generation.join("journals"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        std::fs::read_dir(report.generation.join("heads"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        std::fs::read_dir(report.generation.join("segments"))
            .unwrap()
            .count(),
        1
    );
    let reopened = V2Store::open(&report.generation).unwrap();
    assert_eq!(
        reopened.snapshot().get("gc-key").unwrap().preferred.value,
        "retained"
    );
    drop(reopened);
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn wal_ahead_and_torn_tail_migrate_only_committed_v1_frames() {
    let paths = scratch("wdbx-v2-migration-wal-ahead");
    let mut store = DurableStore::open(paths.clone()).unwrap();
    store.put("checkpoint-free", "wal-value").unwrap();
    drop(store);
    let wal = wal_path(&paths);
    let mut file = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    file.write_all(b"dead").unwrap();
    file.sync_all().unwrap();
    let before = std::fs::read(&wal).unwrap();

    let (store, report) = open_versioned_writable(&paths).unwrap();
    assert!(report.migrated);
    assert_eq!(
        store
            .snapshot()
            .get("checkpoint-free")
            .unwrap()
            .preferred
            .value,
        "wal-value"
    );
    assert_eq!(std::fs::read(&wal).unwrap(), before);
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn legacy_monolithic_snapshot_migrates_without_a_manifest() {
    let paths = scratch("wdbx-v2-migration-legacy-snapshot");
    std::fs::create_dir_all(&paths.dir).unwrap();
    let mut snapshot = Snapshot::new();
    snapshot.kv.insert("legacy".into(), "snapshot".into());
    snapshot.vectors.insert(41, vec![0.5, 0.25]);
    snapshot.recount();
    crate::persistence::write_snapshot(paths.index(), &snapshot).unwrap();
    assert!(!paths.manifest().exists());

    let (store, report) = open_versioned_writable(&paths).unwrap();
    assert!(report.migrated);
    assert_eq!(
        store.snapshot().get("legacy").unwrap().preferred.value,
        "snapshot"
    );
    assert!(store.snapshot().get_vector(RecordId::Legacy(41)).is_some());
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn retry_discards_only_unpublished_v2_temporary_generations() {
    let paths = scratch("wdbx-v2-migration-restart");
    std::fs::create_dir_all(&paths.dir).unwrap();
    let stale_staging = paths.dir.join(format!(".{}.migration-stale", paths.base));
    let stale_generation = paths.dir.join(format!("{}.v2-stale", paths.base));
    let retained_backup = paths
        .dir
        .join(format!("{}.v1-backup-retainedZ", paths.base));
    std::fs::create_dir_all(&stale_staging).unwrap();
    std::fs::create_dir_all(&stale_generation).unwrap();
    std::fs::create_dir_all(&retained_backup).unwrap();

    let (_store, _report) = open_versioned_writable(&paths).unwrap();
    assert!(!stale_staging.exists());
    assert!(!stale_generation.exists());
    assert!(retained_backup.exists());
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn activation_rejects_path_traversal_and_bad_v2_version_markers() {
    let paths = scratch("wdbx-v2-activation-validation");
    std::fs::create_dir_all(&paths.dir).unwrap();
    let activation = format!(
        "{ACTIVE_HEADER}{{\"version\":2,\"generation\":\"../escape\",\"backup\":null,\"digest\":\"x\"}}\n"
    );
    std::fs::write(paths_active(&paths), activation).unwrap();
    assert!(migration_status(&paths).is_err());
    std::fs::remove_file(paths_active(&paths)).unwrap();

    let root = paths.dir.join("direct-v2");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("VERSION"), b"ABI-WDBX 3\n").unwrap();
    assert!(matches!(
        V2Store::open(&root),
        Err(V2Error::UnsupportedVersion { .. })
    ));
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn failed_verification_never_publishes_or_changes_v1() {
    let paths = scratch("wdbx-v2-migration-corrupt");
    let mut store = DurableStore::open(paths.clone()).unwrap();
    let query = store.put_vector(&[1.0]).unwrap();
    let response = store.put_vector(&[2.0]).unwrap();
    store
        .add_block("abbey", query, response, "meta", 42)
        .unwrap();
    store.checkpoint().unwrap();
    drop(store);
    let segment = paths.segment(0);
    let before = std::fs::read(&segment).unwrap();
    let mut content = String::from_utf8(before.clone()).unwrap();
    content = content.replacen("\"profile\":\"abbey\"", "\"profile\":\"aviva\"", 1);
    std::fs::write(&segment, content).unwrap();
    let corrupted = std::fs::read(&segment).unwrap();

    assert!(open_versioned_writable(&paths).is_err());
    assert!(!paths_active(&paths).exists());
    assert_eq!(std::fs::read(&segment).unwrap(), corrupted);
    std::fs::remove_dir_all(paths.dir).unwrap();
}

#[test]
fn new_store_uses_v2_and_read_only_empty_open_is_non_mutating() {
    let paths = scratch("wdbx-v2-new");
    assert!(matches!(
        open_versioned_read_only(&paths).unwrap(),
        VersionedSnapshot::Empty
    ));
    assert!(!paths.dir.exists());
    let (_store, report) = open_versioned_writable(&paths).unwrap();
    assert!(!report.migrated);
    assert!(matches!(
        migration_status(&paths).unwrap(),
        MigrationStatus::V2 { .. }
    ));
    std::fs::remove_dir_all(paths.dir).unwrap();
}
