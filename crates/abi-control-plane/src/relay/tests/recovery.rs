use super::*;
use crate::{
    DeliveryPolicy, DurableSubmission, OutboxPacket, RecordedOutcome, RecoveryAction,
    RecoveryRequest, RecoveryStatus, TerminalManifest, TerminalPacket,
};
fn durable(f: &mut Fixture) -> Submission {
    let s = submission(f, &f.alice, &f.node_a);
    let delivery = DeliveryPolicy::for_submission(&s, 100 + 86400).unwrap();
    f.relay
        .submit_durable(
            &mut f.authority,
            &f.alice,
            DurableSubmission {
                submission: s.clone(),
                delivery,
            },
            101,
        )
        .unwrap();
    poll(f, true, 102);
    permit(f, true, &s.binding.job_id, 102);
    s
}
fn manifest(
    f: &Fixture,
    s: &Submission,
    frames: &[crate::StreamFrame],
    terminal: Option<&TerminalPacket>,
) -> TerminalManifest {
    TerminalManifest {
        binding: s.binding.clone(),
        submission_digest: crate::authority::hash(&crate::enrollment::json(s).unwrap()),
        frame_digests: frames
            .iter()
            .map(|frame| {
                OutboxPacket::Delta {
                    frame: frame.clone(),
                }
                .id(&s.binding)
                .unwrap()
                .digest
            })
            .collect(),
        terminal_digest: terminal.map(|packet| {
            OutboxPacket::Terminal {
                packet: packet.clone(),
            }
            .id(&s.binding)
            .unwrap()
            .digest
        }),
        outcome: if terminal.is_some() {
            RecordedOutcome::Completed
        } else {
            RecordedOutcome::Unresolved
        },
        completed_at: terminal.map(|_| 104),
        signature: vec![],
    }
    .sign(&f.key_a)
    .unwrap()
}
fn action(f: &mut Fixture, action: RecoveryAction, now: i64) -> crate::RecoveryReply {
    let message = NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Recovery {
            request: RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action,
            },
        },
        now,
    )
    .unwrap();
    f.relay
        .node(&mut f.authority, &message, now)
        .unwrap()
        .recovery
        .unwrap()
}
#[test]
fn restart_replays_identical_packets_after_execution_deadline_and_prior_ack() {
    let mut f = fixture();
    let s = durable(&mut f);
    let id = s.binding.job_id.clone();
    let frame =
        crate::StreamFrame::seal(&s.binding, 0, b"private-recovery-canary", &f.key_a).unwrap();
    let terminal = TerminalPacket::seal(&s.binding, b"result", &f.key_a).unwrap();
    let m = manifest(&f, &s, std::slice::from_ref(&frame), Some(&terminal));
    let inspect = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m.clone(),
        },
        105,
    );
    assert_eq!(inspect.status, RecoveryStatus::Missing);
    let grant = inspect.grant.unwrap();
    let packets = [
        OutboxPacket::Delta { frame },
        OutboxPacket::Terminal { packet: terminal },
    ];
    for packet in &packets {
        assert_eq!(
            action(
                &mut f,
                RecoveryAction::Publish {
                    grant: grant.clone(),
                    packet: packet.clone()
                },
                106
            )
            .status,
            RecoveryStatus::Recorded
        );
    }
    let ack = crate::DeliveryAck {
        manifest_digest: m.digest().unwrap(),
        contiguous: 1,
        terminal_digest: m.terminal_digest.clone(),
    };
    f.relay
        .acknowledge_delivery(&mut f.authority, &f.alice, &id, &ack, 107)
        .unwrap();
    let page = f
        .relay
        .delivery(&mut f.authority, &f.alice, &id, 0, 107)
        .unwrap();
    assert_eq!(page.packets.len(), 2);
    drop(f.relay);
    f.relay = Relay::open(&mut f.authority, RelayLimits::default(), 210).unwrap();
    assert!(
        f.relay
            .delivery(&mut f.authority, &f.alice, &id, 0, 210)
            .unwrap()
            .packets
            .is_empty()
    );
    let inspect = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m,
        },
        211,
    );
    assert_eq!(inspect.cached.len(), 0);
    assert_eq!(inspect.committed.len(), 2);
    let new_grant = inspect.grant.unwrap();
    assert_ne!(grant.epoch, new_grant.epoch);
    let stale = NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Recovery {
            request: RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish {
                    grant,
                    packet: packets[0].clone(),
                },
            },
        },
        212,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &stale, 212).is_err());
    for packet in &packets {
        action(
            &mut f,
            RecoveryAction::Publish {
                grant: new_grant.clone(),
                packet: packet.clone(),
            },
            212,
        );
    }
    assert_restored(&mut f, &id, &packets);
}
#[test]
fn cancellation_survives_restart_and_blocks_success_replay() {
    let mut f = fixture();
    let s = durable(&mut f);
    let id = s.binding.job_id.clone();
    let terminal = TerminalPacket::seal(&s.binding, b"result", &f.key_a).unwrap();
    let m = manifest(&f, &s, &[], Some(&terminal));
    let grant = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m.clone(),
        },
        105,
    )
    .grant
    .unwrap();
    f.relay
        .cancel(&mut f.authority, &f.alice, &id, 106)
        .unwrap();
    let message = NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Recovery {
            request: RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish {
                    grant,
                    packet: OutboxPacket::Terminal { packet: terminal },
                },
            },
        },
        107,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &message, 107).is_err());
    drop(f.relay);
    f.relay = Relay::open(&mut f.authority, RelayLimits::default(), 108).unwrap();
    let inspected = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m,
        },
        109,
    );
    assert!(inspected.cancelled);
    assert_eq!(inspected.grant.unwrap().packets.len(), 0);
    assert_eq!(
        f.relay
            .delivery(&mut f.authority, &f.alice, &id, 0, 109)
            .unwrap()
            .receipt
            .state,
        "cancelled"
    );
}
#[test]
fn schema_migration_retains_v1_ephemeral_and_refuses_future_versions() {
    let mut f = fixture();
    assert_eq!(
        f.authority
            .db
            .query_row("SELECT version FROM relay_meta", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    let s = submission(&f, &f.alice, &f.node_a);
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 101).unwrap();
    assert!(
        Relay::delivery_policy(&f.authority.db, &id)
            .unwrap()
            .is_none()
    );
    drop(f.relay);
    f.authority
        .db
        .execute("UPDATE relay_meta SET version=3", [])
        .unwrap();
    assert!(Relay::open(&mut f.authority, RelayLimits::default(), 102).is_err());
}
#[test]
fn replay_capacity_refusal_preserves_commitments_and_closed_manifest() {
    let mut f = fixture();
    let s = durable(&mut f);
    let id = s.binding.job_id.clone();
    let frame = crate::StreamFrame::seal(&s.binding, 0, &vec![3; 16000], &f.key_a).unwrap();
    let m = manifest(&f, &s, std::slice::from_ref(&frame), None);
    let grant = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m.clone(),
        },
        103,
    )
    .grant
    .unwrap();
    f.relay.limits.memory_bytes = 100;
    let reply = action(
        &mut f,
        RecoveryAction::Publish {
            grant: grant.clone(),
            packet: OutboxPacket::Delta {
                frame: frame.clone(),
            },
        },
        104,
    );
    assert_eq!(reply.status, RecoveryStatus::CapacityFull);
    assert!(!f.relay.frames.contains_key(&id));
    f.relay.limits.memory_bytes = 1024 * 1024;
    action(
        &mut f,
        RecoveryAction::Publish {
            grant,
            packet: OutboxPacket::Delta { frame },
        },
        105,
    );
    let terminal = TerminalPacket::seal(&s.binding, b"done", &f.key_a).unwrap();
    let final_manifest = TerminalManifest {
        terminal_digest: Some(
            OutboxPacket::Terminal { packet: terminal }
                .id(&s.binding)
                .unwrap()
                .digest,
        ),
        outcome: RecordedOutcome::Completed,
        completed_at: Some(104),
        signature: vec![],
        ..m.clone()
    }
    .sign(&f.key_a)
    .unwrap();
    action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: final_manifest,
        },
        106,
    );
    assert_eq!(
        action(
            &mut f,
            RecoveryAction::Inspect {
                job_id: id,
                manifest: m
            },
            107
        )
        .status,
        RecoveryStatus::Contradictory
    );
}
#[test]
fn policy_and_packet_identity_signatures_refuse_mutation() {
    let mut f = fixture();
    let s = durable(&mut f);
    let id = &s.binding.job_id;
    let signed = Relay::delivery_policy(&f.authority.db, id)
        .unwrap()
        .unwrap();
    signed
        .verify(
            &f.authority.public_key(),
            &s.binding,
            &signed.policy.submission_digest,
        )
        .unwrap();
    let mut altered = signed.clone();
    altered.policy.until_unix_s += 1;
    assert!(
        altered
            .verify(
                &f.authority.public_key(),
                &s.binding,
                &signed.policy.submission_digest
            )
            .is_err()
    );
    let terminal = TerminalPacket::seal(&s.binding, b"test", &f.key_a).unwrap();
    terminal
        .verify(&s.binding, &f.key_a.verifying_key().to_bytes())
        .unwrap();
    let mut wrong = s.binding.clone();
    wrong.job_id = uuid::Uuid::new_v4().to_string();
    assert!(
        terminal
            .verify(&wrong, &f.key_a.verifying_key().to_bytes())
            .is_err()
    );
    let m = manifest(&f, &s, &[], Some(&terminal));
    let grant = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m,
        },
        104,
    )
    .grant
    .unwrap();
    assert!(grant.verify(&f.authority.public_key(), 134).is_err());
    let mut changed = grant;
    changed.cancelled = true;
    assert!(changed.verify(&f.authority.public_key(), 105).is_err());
}
#[test]
fn delivery_pages_enforce_actual_json_ceiling_and_contiguous_cursor() {
    let mut f = fixture();
    let mut s = submission(&f, &f.alice, &f.node_a);
    s.binding.max_result_bytes = 65536;
    let id = s.binding.job_id.clone();
    let delivery = DeliveryPolicy::for_submission(&s, 86500).unwrap();
    f.relay
        .submit_durable(
            &mut f.authority,
            &f.alice,
            DurableSubmission {
                submission: s.clone(),
                delivery,
            },
            101,
        )
        .unwrap();
    poll(&mut f, true, 102);
    permit(&mut f, true, &id, 102);
    let bytes: Vec<u8> = (0..65536).map(|i| u8::try_from(i % 256).unwrap()).collect();
    let frames = vec![
        crate::StreamFrame::seal(&s.binding, 0, &bytes[..32768], &f.key_a).unwrap(),
        crate::StreamFrame::seal(&s.binding, 1, &bytes[32768..], &f.key_a).unwrap(),
    ];
    let terminal = TerminalPacket::seal(&s.binding, &bytes, &f.key_a).unwrap();
    let m = manifest(&f, &s, &frames, Some(&terminal));
    let grant = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m,
        },
        104,
    )
    .grant
    .unwrap();
    for frame in frames {
        action(
            &mut f,
            RecoveryAction::Publish {
                grant: grant.clone(),
                packet: OutboxPacket::Delta { frame },
            },
            105,
        );
    }
    action(
        &mut f,
        RecoveryAction::Publish {
            grant,
            packet: OutboxPacket::Terminal { packet: terminal },
        },
        105,
    );
    let first = f
        .relay
        .delivery(&mut f.authority, &f.alice, &id, 0, 106)
        .unwrap();
    assert!(serde_json::to_vec(&first).unwrap().len() <= 384 * 1024);
    assert_eq!(first.next, 2);
    assert_eq!(first.packets.len(), 2);
    let last = f
        .relay
        .delivery(&mut f.authority, &f.alice, &id, first.next, 106)
        .unwrap();
    assert_eq!(last.packets.len(), 1);
    assert!(serde_json::to_vec(&last).unwrap().len() <= 384 * 1024);
}
#[test]
fn populated_version_one_migrates_without_retroactive_durability() {
    let mut f = fixture();
    let s = submission(&f, &f.alice, &f.node_a);
    let id = s.binding.job_id.clone();
    drop(f.relay);
    f.authority
        .db
        .execute_batch("DROP TABLE relay_frames; DROP TABLE relay_jobs; DROP TABLE relay_meta;")
        .unwrap();
    f.authority
        .db
        .execute_batch(include_str!("../../relay_schema.sql"))
        .unwrap();
    f.authority.db.execute("INSERT INTO relay_jobs(id,account,node,binding,digest,state,deadline) VALUES(?1,?2,?3,?4,?5,'dispatched',200)",params![id,f.alice.account_id,f.node_a.node_id,crate::enrollment::json(&s.binding).unwrap(),crate::authority::hash(&crate::enrollment::json(&s).unwrap())]).unwrap();
    f.relay = Relay::open(&mut f.authority, RelayLimits::default(), 105).unwrap();
    assert_eq!(
        f.relay
            .view(&mut f.authority, &f.alice, &id, 105)
            .unwrap()
            .receipt
            .state,
        "unresolved"
    );
    assert!(
        Relay::delivery_policy(&f.authority.db, &id)
            .unwrap()
            .is_none()
    );
    let m = manifest(&f, &s, &[], None);
    assert_eq!(
        action(
            &mut f,
            RecoveryAction::Inspect {
                job_id: id,
                manifest: m
            },
            106
        )
        .status,
        RecoveryStatus::Unavailable
    );
}

fn assert_restored(f: &mut Fixture, id: &str, packets: &[OutboxPacket]) {
    let page = f
        .relay
        .delivery(&mut f.authority, &f.alice, id, 0, 213)
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&page.packets).unwrap(),
        serde_json::to_vec(packets).unwrap()
    );
    assert_eq!(page.receipt.state, "completed");
    assert!(
        f.relay
            .delivery(&mut f.authority, &f.bob, id, 0, 213)
            .is_err()
    );
    let db = std::fs::read(f.dir.path().join("authority.sqlite")).unwrap();
    for packet in packets {
        let bytes = match packet {
            OutboxPacket::Delta { frame } => &frame.payload.ciphertext,
            OutboxPacket::Terminal { packet } => &packet.payload.ciphertext,
        };
        assert!(!db.windows(bytes.len()).any(|w| w == bytes));
    }
    assert!(!db.windows(23).any(|w| w == b"private-recovery-canary"));
    assert_eq!(
        f.authority
            .db
            .query_row(
                "SELECT count(*) FROM relay_frames WHERE job=?1",
                [id],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}
#[test]
fn failed_audit_rolls_back_new_commitments_and_exposes_no_transient_packet() {
    let mut f = fixture();
    let s = durable(&mut f);
    let id = s.binding.job_id.clone();
    let frame = crate::StreamFrame::seal(&s.binding, 0, b"saved prefix", &f.key_a).unwrap();
    let terminal = TerminalPacket::seal(&s.binding, b"saved terminal", &f.key_a).unwrap();
    let m = manifest(&f, &s, std::slice::from_ref(&frame), Some(&terminal));
    let grant = action(
        &mut f,
        RecoveryAction::Inspect {
            job_id: id.clone(),
            manifest: m,
        },
        104,
    )
    .grant
    .unwrap();
    for packet in [
        OutboxPacket::Delta { frame },
        OutboxPacket::Terminal { packet: terminal },
    ] {
        let before: i64 = f
            .authority
            .db
            .query_row("SELECT count(*) FROM audit WHERE resource=?1", [&id], |r| {
                r.get(0)
            })
            .unwrap();
        f.authority.db.execute_batch("CREATE TRIGGER fail_packet_audit BEFORE INSERT ON audit WHEN NEW.kind IN ('job.frame','job.finished') BEGIN SELECT RAISE(ABORT,'injected audit failure'); END;").unwrap();
        let message = publish_message(&f, &grant, &packet, 105);
        assert!(f.relay.node(&mut f.authority, &message, 105).is_err());
        let count: i64 = f
            .authority
            .db
            .query_row("SELECT count(*) FROM audit WHERE resource=?1", [&id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, before);
        match &packet {
            OutboxPacket::Delta { .. } => {
                assert!(!f.relay.frames.contains_key(&id));
                assert_eq!(
                    f.authority
                        .db
                        .query_row(
                            "SELECT count(*) FROM relay_frames WHERE job=?1",
                            [&id],
                            |r| r.get::<_, i64>(0)
                        )
                        .unwrap(),
                    0
                );
            }
            OutboxPacket::Terminal { .. } => {
                assert!(!f.relay.terminals.contains_key(&id));
                assert_eq!(
                    f.authority
                        .db
                        .query_row(
                            "SELECT result_digest FROM relay_jobs WHERE id=?1",
                            [&id],
                            |r| r.get::<_, Option<String>>(0)
                        )
                        .unwrap(),
                    None
                );
            }
        }
        f.authority
            .db
            .execute_batch("DROP TRIGGER fail_packet_audit;")
            .unwrap();
        for _ in 0..2 {
            action(
                &mut f,
                RecoveryAction::Publish {
                    grant: grant.clone(),
                    packet: packet.clone(),
                },
                106,
            );
        }
        let count: i64 = f
            .authority
            .db
            .query_row("SELECT count(*) FROM audit WHERE resource=?1", [&id], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, before + 1);
    }
}
#[test]
fn legacy_view_cannot_bypass_durable_revocation_expiry_or_cancellation() {
    for revoke in [false, true] {
        let mut f = fixture();
        let s = durable(&mut f);
        let id = s.binding.job_id.clone();
        let terminal = TerminalPacket::seal(&s.binding, b"result", &f.key_a).unwrap();
        let m = manifest(&f, &s, &[], Some(&terminal));
        let grant = action(
            &mut f,
            RecoveryAction::Inspect {
                job_id: id.clone(),
                manifest: m,
            },
            104,
        )
        .grant
        .unwrap();
        action(
            &mut f,
            RecoveryAction::Publish {
                grant,
                packet: OutboxPacket::Terminal { packet: terminal },
            },
            105,
        );
        assert!(
            f.relay
                .view(&mut f.authority, &f.alice, &id, 106)
                .unwrap()
                .result
                .is_some()
        );
        assert!(
            f.relay
                .view(&mut f.authority, &f.alice, &id, f.node_a.expires_at)
                .is_err()
        );
        if revoke {
            f.authority
                .revoke(&f.alice, &f.node_a.node_id, 107)
                .unwrap();
            assert!(f.relay.view(&mut f.authority, &f.alice, &id, 108).is_err());
        } else {
            f.relay
                .cancel(&mut f.authority, &f.alice, &id, 107)
                .unwrap();
            let view = f.relay.view(&mut f.authority, &f.alice, &id, 108).unwrap();
            assert_eq!(view.receipt.state, "cancelled");
            assert!(view.result.is_none());
        }
    }
}

fn publish_message(
    f: &Fixture,
    grant: &crate::ReplayGrant,
    packet: &OutboxPacket,
    now: i64,
) -> NodeMessage {
    NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Recovery {
            request: RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish {
                    grant: grant.clone(),
                    packet: packet.clone(),
                },
            },
        },
        now,
    )
    .unwrap()
}
