use super::*;
use crate::{DeliveryPolicy, JobBinding, RelayKey, StreamFrame, Submission, seal};
use ed25519_dalek::SigningKey;
fn directory() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    d
}
fn fixture() -> (Dispatch, SignedDeliveryPolicy, SigningKey) {
    let key = SigningKey::from_bytes(&[7; 32]);
    let reply = RelayKey::generate().unwrap();
    let binding = JobBinding {
        protocol: "abi.relay-job.v1".into(),
        audience: "http://localhost".into(),
        job_id: uuid::Uuid::new_v4().to_string(),
        account_id: uuid::Uuid::new_v4().to_string(),
        tenant_id: uuid::Uuid::new_v4().to_string(),
        node_id: uuid::Uuid::new_v4().to_string(),
        capability: "model.infer".into(),
        expires_at: 300,
        submitted_at: 1,
        max_result_bytes: 65536,
        reply_key: reply.public_key(),
    };
    let submission = Submission {
        payload: seal(
            &reply.public_key(),
            &binding.aad("request").unwrap(),
            b"private-outbox-canary",
        )
        .unwrap(),
        binding,
    };
    let policy = SignedDeliveryPolicy::sign(
        DeliveryPolicy::for_submission(&submission, 86401).unwrap(),
        &key,
    )
    .unwrap();
    (
        Dispatch {
            submission,
            signature: vec![1; 64],
        },
        policy,
        key,
    )
}
#[test]
fn exclusive_create_reopen_retains_exact_packets_and_expiry_tombstone() {
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    let (dispatch, policy, key) = fixture();
    let id = dispatch.submission.binding.job_id.clone();
    let mut out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
    assert!(FileRelayOutbox::create(&path, OutboxLimits::default()).is_err());
    assert!(FileRelayOutbox::open(&path, OutboxLimits::default()).is_err());
    out.prepare(&dispatch, &policy).unwrap();
    out.prepare(&dispatch, &policy).unwrap();
    let packet = OutboxPacket::Delta {
        frame: StreamFrame::seal(&dispatch.submission.binding, 0, b"delta-canary", &key).unwrap(),
    };
    out.append(&id, &packet).unwrap();
    out.append(&id, &packet).unwrap();
    let before = encode(&packet).unwrap();
    drop(out);
    let mut out = FileRelayOutbox::open(&path, OutboxLimits::default()).unwrap();
    assert_eq!(
        encode(
            &out.packet(&id, packet.id(&dispatch.submission.binding).unwrap())
                .unwrap()
                .unwrap()
        )
        .unwrap(),
        before
    );
    out.expire(86401).unwrap();
    assert!(out.entries(86401).unwrap().is_empty());
    assert!(out.entry(&id).unwrap().unwrap().packets.is_empty());
    assert!(out.prepare(&dispatch, &policy).is_err());
    drop(out);
    assert!(FileRelayOutbox::open(&path, OutboxLimits::default()).is_ok());
    assert!(
        !std::fs::read(&path)
            .unwrap()
            .windows(b"delta-canary".len())
            .any(|w| w == b"delta-canary")
    );
}
#[test]
fn frame_order_conflict_reservation_and_missing_file_refuse() {
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    assert!(FileRelayOutbox::open(&path, OutboxLimits::default()).is_err());
    assert!(!path.exists());
    let (dispatch, policy, key) = fixture();
    let id = &dispatch.submission.binding.job_id;
    let mut out = FileRelayOutbox::create(
        &path,
        OutboxLimits {
            retained_jobs: 1,
            ..Default::default()
        },
    )
    .unwrap();
    out.prepare(&dispatch, &policy).unwrap();
    let (other, other_policy, _) = fixture();
    assert!(out.prepare(&other, &other_policy).is_err());
    let frame = |n, bytes: &[u8]| OutboxPacket::Delta {
        frame: StreamFrame::seal(&dispatch.submission.binding, n, bytes, &key).unwrap(),
    };
    assert!(out.append(id, &frame(1, b"gap")).is_err());
    out.append(id, &frame(0, b"first")).unwrap();
    assert!(out.append(id, &frame(0, b"changed")).is_err());
    for i in 1..16 {
        out.append(id, &frame(i, b"next")).unwrap();
    }
    assert_eq!(out.entry(id).unwrap().unwrap().packets.len(), 16);
}
#[cfg(unix)]
#[test]
fn private_permissions_symlink_and_replacement_refuse() {
    use std::os::unix::fs::{PermissionsExt as _, symlink};
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    let out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o077,
        0
    );
    let link = d.path().join("link");
    symlink(&path, &link).unwrap();
    assert!(FileRelayOutbox::open(&link, OutboxLimits::default()).is_err());
    std::fs::rename(&path, d.path().join("old")).unwrap();
    std::fs::write(&path, b"replacement").unwrap();
    assert!(out.entries(0).is_err());
}
#[test]
fn actual_sqlite_full_partial_write_and_sync_failures_poison() {
    for fault in [
        faults::Fault::Full,
        faults::Fault::PartialWrite,
        faults::Fault::Sync,
    ] {
        let d = directory();
        let path = d.path().join("outbox.sqlite");
        let guard = faults::Guard::new(&path);
        let (dispatch, policy, key) = fixture();
        let id = &dispatch.submission.binding.job_id;
        let mut out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
        out.prepare(&dispatch, &policy).unwrap();
        let packet = OutboxPacket::Delta {
            frame: StreamFrame::seal(&dispatch.submission.binding, 0, b"committed-prefix", &key)
                .unwrap(),
        };
        guard.arm(fault);
        assert!(out.append(id, &packet).is_err());
        assert!(out.poisoned());
        assert!(out.entries(1).is_err());
        assert!(out.append(id, &packet).is_err());
        drop(out);
        if let Ok(out) = FileRelayOutbox::open(&path, OutboxLimits::default()) {
            let entry = out.entry(id).unwrap().unwrap();
            assert!(
                entry.packets.is_empty()
                    || encode(&entry.packets[0]).unwrap() == encode(&packet).unwrap()
            );
        }
    }
}

#[test]
fn cancellation_writer_is_monotonic_and_ack_keeps_ciphertext() {
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    let (dispatch, policy, key) = fixture();
    let id = &dispatch.submission.binding.job_id;
    let mut out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
    out.prepare(&dispatch, &policy).unwrap();
    let mut recorder = out.cancellation_recorder(id).unwrap();
    recorder.observe().unwrap();
    let packet = OutboxPacket::Delta {
        frame: StreamFrame::seal(&dispatch.submission.binding, 0, b"retained", &key).unwrap(),
    };
    out.append(id, &packet).unwrap();
    assert!(out.entry(id).unwrap().unwrap().cancelled);
    out.record_ack(&crate::RecoveryReply {
        status: crate::RecoveryStatus::Cached,
        epoch: "epoch".into(),
        committed: vec![packet.id(&dispatch.submission.binding).unwrap()],
        cached: vec![],
        grant: None,
        cancelled: true,
    })
    .unwrap();
    drop(recorder);
    drop(out);
    let out = FileRelayOutbox::open(&path, OutboxLimits::default()).unwrap();
    let entry = out.entry(id).unwrap().unwrap();
    assert!(entry.cancelled);
    assert_eq!(entry.packets.len(), 1);
}

#[test]
fn terminal_closes_packets_and_encoded_job_reservation_is_enforced() {
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    let (dispatch, policy, key) = fixture();
    let id = &dispatch.submission.binding.job_id;
    let mut out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
    out.prepare(&dispatch, &policy).unwrap();
    let terminal = OutboxPacket::Terminal {
        packet: crate::TerminalPacket::seal(&dispatch.submission.binding, b"sealed-result", &key)
            .unwrap(),
    };
    out.append(id, &terminal).unwrap();
    out.append(id, &terminal).unwrap();
    let delta = OutboxPacket::Delta {
        frame: StreamFrame::seal(&dispatch.submission.binding, 0, b"late", &key).unwrap(),
    };
    assert!(out.append(id, &delta).is_err());
    let other = d.path().join("small.sqlite");
    let mut small = FileRelayOutbox::create(
        &other,
        OutboxLimits {
            job_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(small.prepare(&dispatch, &policy).is_err());
}

#[test]
fn modified_packet_history_refuses_instead_of_resetting() {
    let d = directory();
    let path = d.path().join("outbox.sqlite");
    let (dispatch, policy, _) = fixture();
    let mut out = FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap();
    out.prepare(&dispatch, &policy).unwrap();
    drop(out);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "UPDATE entries SET body=replace(CAST(body AS TEXT),'model.infer','model.other')",
        [],
    )
    .unwrap();
    drop(db);
    assert!(FileRelayOutbox::open(&path, OutboxLimits::default()).is_err());
    assert!(path.exists());
}

#[cfg(unix)]
#[test]
fn writable_ancestor_and_symlink_journal_refuse() {
    use std::os::unix::fs::{PermissionsExt as _, symlink};
    let d = directory();
    let shared = d.path().join("shared");
    std::fs::create_dir(&shared).unwrap();
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o777)).unwrap();
    let private = shared.join("private");
    std::fs::create_dir(&private).unwrap();
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(FileRelayOutbox::create(&private.join("bad.sqlite"), OutboxLimits::default()).is_err());
    let path = d.path().join("outbox.sqlite");
    drop(FileRelayOutbox::create(&path, OutboxLimits::default()).unwrap());
    let other = d.path().join("other");
    std::fs::write(&other, b"private").unwrap();
    symlink(&other, d.path().join("outbox.sqlite-journal")).unwrap();
    assert!(FileRelayOutbox::open(&path, OutboxLimits::default()).is_err());
}
