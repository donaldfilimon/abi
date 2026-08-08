use super::*;
use abi_foundation::temp_path::temp_file_path;
use std::sync::{Arc, Barrier};

fn scratch() -> PathBuf {
    temp_file_path("wdbx-v2", "store")
}

#[test]
fn record_ids_keep_legacy_numbers_and_v2_strings() {
    assert_eq!(serde_json::to_string(&RecordId::Legacy(7)).unwrap(), "7");
    let id = RecordId::new_v2();
    let json = serde_json::to_string(&id).unwrap();
    assert!(json.starts_with('"'));
    assert_eq!(serde_json::from_str::<RecordId>(&json).unwrap(), id);
}

#[test]
fn mutations_in_one_transaction_are_causal_peers() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    let first = RecordId::new_v2();
    let second = RecordId::new_v2();
    store
        .commit(vec![
            V2Mutation::PutVector {
                id: first,
                values: vec![1.0, 0.0],
            },
            V2Mutation::PutVector {
                id: second,
                values: vec![0.0, 1.0],
            },
            V2Mutation::PutKv {
                key: "peer-key".into(),
                value: "first".into(),
            },
            V2Mutation::PutKv {
                key: "peer-key".into(),
                value: "second".into(),
            },
        ])
        .unwrap();

    let snapshot = store.snapshot();
    assert_eq!(snapshot.vector_count(), 2);
    assert!(snapshot.get_vector(first).is_some());
    assert!(snapshot.get_vector(second).is_some());
    assert!(matches!(
        snapshot.causal_focus_vector_id(),
        Some(id) if id == first || id == second
    ));
    let values = snapshot.get("peer-key").unwrap();
    assert_eq!(values.conflicts.len(), 1);
    let mut visible = vec![values.preferred.value, values.conflicts[0].value.clone()];
    visible.sort();
    assert_eq!(visible, ["first", "second"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_writers_surface_conflicts_and_resolution_dominates_them() {
    let root = scratch();
    let mut left = V2Store::open(&root).unwrap();
    let mut right = V2Store::open(&root).unwrap();
    left.commit(vec![V2Mutation::PutKv {
        key: "k".into(),
        value: "left".into(),
    }])
    .unwrap();
    // Force right's observation back to its open-time empty frontier to model
    // a genuinely concurrent transaction written without refreshing peers.
    let right_begin = BeginFrame {
        writer_id: right.writer_id,
        transaction_id: Uuid::new_v4(),
        sequence: 1,
        observed_heads: CausalHeads::new(),
    };
    let mutations = vec![V2Mutation::PutKv {
        key: "k".into(),
        value: "right".into(),
    }];
    let hash = transaction_hash(&right_begin, &mutations).unwrap();
    append_frames(
        &right.journal_path(),
        &[
            JournalFrame::Begin(right_begin.clone()),
            JournalFrame::Mutation {
                transaction_id: right_begin.transaction_id,
                mutation: mutations[0].clone(),
            },
            JournalFrame::Commit {
                transaction_id: right_begin.transaction_id,
                transaction_hash: hash,
            },
        ],
    )
    .unwrap();
    atomic_write(&right.head_path(1), b"committed\n").unwrap();
    right.refresh().unwrap();
    let set = right.snapshot().get("k").unwrap();
    assert_eq!(set.conflicts.len(), 1);
    let ids = [set.preferred.version_id, set.conflicts[0].version_id];
    right.resolve("k", &ids, "resolved".into()).unwrap();
    let resolved = right.snapshot().get("k").unwrap();
    assert_eq!(resolved.preferred.value, "resolved");
    assert_eq!(resolved.conflicts.len(), 0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn incomplete_transaction_is_ignored_and_old_views_survive_mutation() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "k".into(),
            value: "old".into(),
        }])
        .unwrap();
    let old = store.snapshot();
    let begin = JournalFrame::Begin(BeginFrame {
        writer_id: store.writer_id,
        transaction_id: Uuid::new_v4(),
        sequence: 2,
        observed_heads: old.heads.clone(),
    });
    append_frames(&store.journal_path(), &[begin]).unwrap();
    store.refresh().unwrap();
    assert_eq!(store.snapshot().get("k").unwrap().preferred.value, "old");
    store
        .commit(vec![V2Mutation::PutKv {
            key: "k".into(),
            value: "new".into(),
        }])
        .unwrap();
    assert_eq!(old.get("k").unwrap().preferred.value, "old");
    assert_eq!(store.snapshot().get("k").unwrap().preferred.value, "new");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn v2_accepts_4096_dimensions_but_rejects_bad_vectors() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    let id = RecordId::new_v2();
    store
        .commit(vec![V2Mutation::PutVector {
            id,
            values: vec![0.25; 4096],
        }])
        .unwrap();
    assert_eq!(
        store
            .snapshot()
            .get_vector(id)
            .unwrap()
            .preferred
            .value
            .len(),
        4096
    );
    assert!(
        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![]
            }])
            .is_err()
    );
    assert!(
        store
            .commit(vec![V2Mutation::PutVector {
                id: RecordId::new_v2(),
                values: vec![f32::NAN]
            }])
            .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_tampered_committed_transaction_fails_closed() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "k".into(),
            value: "value".into(),
        }])
        .unwrap();
    let journal = store.object_journal_path();
    drop(store);
    let mut content = std::fs::read(&journal).unwrap();
    let marker = b"\"transaction_hash\":\"";
    let hash_start = content
        .windows(marker.len())
        .position(|window| window == marker)
        .unwrap()
        + marker.len();
    content[hash_start] = if content[hash_start] == b'0' {
        b'1'
    } else {
        b'0'
    };
    std::fs::write(&journal, content).unwrap();
    let Err(error) = V2Store::open(&root) else {
        panic!("tampered commit must not open");
    };
    assert!(error.to_string().contains("authentication failed"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_published_head_without_its_transaction_fails_closed() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "missing".into(),
            value: "journal".into(),
        }])
        .unwrap();
    let journal = store.object_journal_path();
    drop(store);
    std::fs::remove_file(journal).unwrap();
    let Err(error) = V2Store::open(&root) else {
        panic!("a published head cannot outlive all recoverable state");
    };
    assert!(error.to_string().contains("has no recoverable transaction"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn encrypted_signed_journals_recover_and_hide_plaintext() {
    let root = scratch();
    let (encryption, signing, verifying) = generate_key_material();
    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let mut store = V2Store::open_with_security(&root, security).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "classified-key".into(),
            value: "classified-value".into(),
        }])
        .unwrap();
    let journal = std::fs::read(store.object_journal_path()).unwrap();
    assert!(
        !journal
            .windows(b"classified-value".len())
            .any(|window| window == b"classified-value")
    );
    drop(store);

    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let reopened = V2Store::open_with_security(&root, security).unwrap();
    assert_eq!(
        reopened
            .snapshot()
            .get("classified-key")
            .unwrap()
            .preferred
            .value,
        "classified-value"
    );
    drop(reopened);

    let error = V2Store::open_with_security(&root, ObjectSecurity::plaintext())
        .err()
        .expect("encrypted signed journals require configured keys");
    assert!(error.to_string().contains(ABI_WDBX_VERIFY_KEY_FILE));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_torn_authenticated_object_tail_is_ignored() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "durable".into(),
            value: "visible".into(),
        }])
        .unwrap();
    let journal = store.object_journal_path();
    drop(store);
    let mut file = OpenOptions::new().append(true).open(&journal).unwrap();
    file.write_all(&128_u64.to_be_bytes()).unwrap();
    file.write_all(b"crash-tail").unwrap();
    file.sync_all().unwrap();
    drop(file);

    let reopened = V2Store::open(&root).unwrap();
    assert_eq!(
        reopened.snapshot().get("durable").unwrap().preferred.value,
        "visible"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn compaction_preserves_covered_state_and_replays_uncovered_journals() {
    let root = scratch();
    let mut first = V2Store::open(&root).unwrap();
    first
        .commit(vec![V2Mutation::PutKv {
            key: "covered".into(),
            value: "segment".into(),
        }])
        .unwrap();
    let report = first.compact().unwrap();
    assert!(!report.encrypted);
    assert!(!report.signed);
    assert!(report.path.is_file());

    let mut concurrent = V2Store::open(&root).unwrap();
    concurrent
        .commit(vec![V2Mutation::PutKv {
            key: "uncovered".into(),
            value: "journal".into(),
        }])
        .unwrap();
    drop((first, concurrent));

    let reopened = V2Store::open(&root).unwrap();
    assert_eq!(
        reopened.snapshot().get("covered").unwrap().preferred.value,
        "segment"
    );
    assert_eq!(
        reopened
            .snapshot()
            .get("uncovered")
            .unwrap()
            .preferred
            .value,
        "journal"
    );
    assert_eq!(reopened.snapshot().committed_transactions(), 2);
    assert!(std::fs::read_dir(root.join("journals")).unwrap().count() >= 2);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn encrypted_signed_segments_hide_plaintext_and_fail_closed_on_tampering() {
    let root = scratch();
    let (encryption, signing, verifying) = generate_key_material();
    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let mut store = V2Store::open_with_security(&root, security).unwrap();
    store
        .commit(vec![V2Mutation::PutKv {
            key: "segment-secret".into(),
            value: "hidden-in-segment".into(),
        }])
        .unwrap();
    let report = store.compact().unwrap();
    assert!(report.encrypted);
    assert!(report.signed);
    let mut bytes = std::fs::read(&report.path).unwrap();
    assert!(
        !bytes
            .windows(b"hidden-in-segment".len())
            .any(|window| window == b"hidden-in-segment")
    );
    let midpoint = bytes.len() / 2;
    bytes[midpoint] ^= 1;
    std::fs::write(&report.path, bytes).unwrap();
    drop(store);

    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let Err(error) = V2Store::open_with_security(&root, security) else {
        panic!("tampered immutable segment must fail closed");
    };
    assert!(error.to_string().contains("authentication failed"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn required_signature_policy_rejects_plaintext_and_accepts_signed_generations() {
    let plain_root = scratch();
    let mut plain = V2Store::open(&plain_root).unwrap();
    plain
        .commit(vec![V2Mutation::PutKv {
            key: "plain".into(),
            value: "unsigned".into(),
        }])
        .unwrap();
    drop(plain);
    let error = recover_with_policy(&plain_root, &ObjectSecurity::plaintext(), true).unwrap_err();
    assert!(error.to_string().contains("signature is required"));

    let secure_root = scratch();
    let (encryption, signing, verifying) = generate_key_material();
    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let mut secure = V2Store::open_with_security(&secure_root, security).unwrap();
    secure
        .commit(vec![V2Mutation::PutKv {
            key: "signed".into(),
            value: "verified".into(),
        }])
        .unwrap();
    secure.compact().unwrap();
    drop(secure);
    let security =
        ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying)).unwrap();
    let recovered = recover_with_policy(&secure_root, &security, true).unwrap();
    assert_eq!(recovered.get("signed").unwrap().preferred.value, "verified");
    std::fs::remove_dir_all(plain_root).unwrap();
    std::fs::remove_dir_all(secure_root).unwrap();
}

#[test]
fn a_synced_commit_is_invisible_until_its_writer_head_is_published() {
    let root = scratch();
    let mut store = V2Store::open(&root).unwrap();
    let begin = BeginFrame {
        writer_id: store.writer_id,
        transaction_id: Uuid::new_v4(),
        sequence: 1,
        observed_heads: CausalHeads::new(),
    };
    let mutations = vec![V2Mutation::PutKv {
        key: "unpublished".into(),
        value: "hidden".into(),
    }];
    let hash = transaction_hash(&begin, &mutations).unwrap();
    append_frames(
        &store.journal_path(),
        &[
            JournalFrame::Begin(begin.clone()),
            JournalFrame::Mutation {
                transaction_id: begin.transaction_id,
                mutation: mutations[0].clone(),
            },
            JournalFrame::Commit {
                transaction_id: begin.transaction_id,
                transaction_hash: hash,
            },
        ],
    )
    .unwrap();
    store.refresh().unwrap();
    assert!(store.snapshot().get("unpublished").is_none());
    atomic_write(&store.head_path(1), b"committed\n").unwrap();
    store.refresh().unwrap();
    assert_eq!(
        store.snapshot().get("unpublished").unwrap().preferred.value,
        "hidden"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn fifty_independent_writers_recover_every_commit_without_a_global_lock() {
    let root = scratch();
    let barrier = Arc::new(Barrier::new(50));
    let mut workers = Vec::new();
    for index in 0_u16..50 {
        let root = root.clone();
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let mut store = V2Store::open(root).unwrap();
            barrier.wait();
            store
                .commit(vec![
                    V2Mutation::PutKv {
                        key: format!("writer-{index}"),
                        value: index.to_string(),
                    },
                    V2Mutation::PutVector {
                        id: RecordId::new_v2(),
                        values: vec![f32::from(index), 1.0],
                    },
                ])
                .unwrap();
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let recovered = V2Store::open(&root).unwrap().snapshot();
    assert_eq!(recovered.committed_transactions(), 50);
    assert_eq!(recovered.vector_ids().count(), 50);
    for index in 0_u16..50 {
        assert_eq!(
            recovered
                .get(&format!("writer-{index}"))
                .unwrap()
                .preferred
                .value,
            index.to_string()
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}
