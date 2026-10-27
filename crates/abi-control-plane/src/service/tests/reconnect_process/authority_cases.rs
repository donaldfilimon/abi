//! Process-level current-authority and private-journal refusals.
use super::*;

#[test]
fn reconnect_process_current_authority_refuses_revoked_deleted_and_expired_nodes() {
    for fault in ["revoked", "deleted", "expired", "capacity"] {
        let (directory, mut config, submission, _) = seeded();
        let worker = Process::start(&config);
        assert_eq!(worker.ready(), "synced");
        drop(worker);
        let outbox = FileRelayOutbox::open(
            &directory.path().join("outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap();
        let entry = outbox.entry(&submission.binding.job_id).unwrap().unwrap();
        let manifest = entry.manifest.unwrap();
        let saved = serde_json::to_vec(&entry.packets).unwrap();
        config.role = "authority".into();
        config.fault = fault.into();
        if fault == "expired" {
            let authority = Authority::open(
                &directory.path().join("authority.sqlite"),
                false,
                &config.audience,
            )
            .unwrap();
            config.certificate.expires_at = now().unwrap() - 1;
            config.certificate.signature = authority
                .key
                .sign(&config.certificate.signing_bytes().unwrap())
                .to_bytes()
                .to_vec();
        }
        let service = Process::start(&config);
        let address = service.ready();
        let key = SigningKey::from_bytes(&[41; 32]);
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Inspect {
                job_id: submission.binding.job_id.clone(),
                manifest,
            },
        };
        let message = NodeMessage::sign(
            &config.certificate,
            &key,
            NodeAction::Recovery { request },
            now().unwrap(),
        )
        .unwrap();
        let (status, body) = post_status(&address, &message);
        if fault == "capacity" {
            assert_eq!(status, 200);
            let response: NodeReply = serde_json::from_slice(&body).unwrap();
            let grant = response.recovery.unwrap().grant.unwrap();
            let packet = entry.packets.first().unwrap().clone();
            let request = RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish { grant, packet },
            };
            let response = post_node(
                &address,
                &NodeMessage::sign(
                    &config.certificate,
                    &key,
                    NodeAction::Recovery { request },
                    now().unwrap(),
                )
                .unwrap(),
            );
            let recovery = response.recovery.unwrap();
            assert_eq!(recovery.status, RecoveryStatus::CapacityFull);
            assert_eq!(recovery.cached, Vec::<PacketId>::new());
        } else {
            assert_eq!(status, 401, "current authority must refuse {fault}");
        }
        drop(service);
        assert_eq!(
            serde_json::to_vec(
                &outbox
                    .entry(&submission.binding.job_id)
                    .unwrap()
                    .unwrap()
                    .packets
            )
            .unwrap(),
            saved
        );
        assert_eq!(
            std::fs::read(directory.path().join("invocations")).unwrap(),
            b"1"
        );
    }
}

#[test]
fn reconnect_process_corruption_lock_and_reservation_failure_refuse_before_inference() {
    for fault in ["corruption", "lock", "outbox_capacity"] {
        let (directory, mut config, submission, _) = seeded();
        if fault == "outbox_capacity" {
            config.fault = fault.into();
            let mut worker = Process::start(&config);
            worker.refused();
            assert!(!directory.path().join("invocations").exists());
            continue;
        }
        let worker = Process::start(&config);
        assert_eq!(worker.ready(), "synced");
        drop(worker);
        config.initialize = false;
        let path = directory.path().join("outbox.sqlite");
        let guard = if fault == "lock" {
            Some(FileRelayOutbox::open(&path, OutboxLimits::default()).unwrap())
        } else {
            None
        };
        if fault == "corruption" {
            let db = rusqlite::Connection::open(&path).unwrap();
            db.execute(
                "UPDATE entries SET checksum='corrupt-fixture' WHERE job=?1",
                [&submission.binding.job_id],
            )
            .unwrap();
        }
        let mut worker = Process::start(&config);
        worker.refused();
        assert_eq!(
            std::fs::read(directory.path().join("invocations")).unwrap(),
            b"1"
        );
        drop(guard);
    }
}

fn second_tenant(config: &ChildConfig) -> ChildConfig {
    let mut authority = Authority::open(
        &config.directory.join("authority.sqlite"),
        false,
        &config.audience,
    )
    .unwrap();
    let time = now().unwrap();
    let account = authority
        .account(
            &crate::types::VerifiedIdentity {
                issuer: "https://fixture.example".into(),
                subject: "bob".into(),
            },
            time,
        )
        .unwrap();
    let key = SigningKey::from_bytes(&[42; 32]);
    let encryption = RelayKey::from_seed(&[102; 32]).unwrap();
    let ticket = authority
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(encryption.public_key()),
                capabilities: vec!["model.infer".into()],
            },
            time,
        )
        .unwrap();
    let claim = authority
        .claim_pairing(&account, &ticket.user_code, time)
        .unwrap();
    let certificate = authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            time,
        )
        .unwrap();
    let mut binding = config.dispatch.submission.binding.clone();
    binding.job_id = uuid::Uuid::new_v4().to_string();
    binding.account_id.clone_from(&account.account_id);
    binding.tenant_id.clone_from(&account.tenant_id);
    binding.node_id.clone_from(&certificate.node_id);
    binding.reply_key = RelayKey::from_seed(&[104; 32]).unwrap().public_key();
    binding.submitted_at = time;
    binding.expires_at = time + 120;
    let payload = seal(
        &encryption.public_key(),
        &binding.aad("request").unwrap(),
        br#"{"prompt":"BOB_PRIVATE_CANARY","tokens":10,"model":"local-echo-v1"}"#,
    )
    .unwrap();
    let submission = Submission { binding, payload };
    let mut relay = Relay::open(&mut authority, RelayLimits::default(), time).unwrap();
    relay
        .submit_durable(
            &mut authority,
            &account,
            DurableSubmission {
                delivery: DeliveryPolicy::for_submission(&submission, time + 3600).unwrap(),
                submission: submission.clone(),
            },
            time,
        )
        .unwrap();
    let offered = relay
        .node(
            &mut authority,
            &NodeMessage::sign(&certificate, &key, NodeAction::Poll, time).unwrap(),
            time,
        )
        .unwrap();
    let permit = relay
        .node(
            &mut authority,
            &NodeMessage::sign(
                &certificate,
                &key,
                NodeAction::Inspect {
                    job_id: submission.binding.job_id,
                },
                time,
            )
            .unwrap(),
            time,
        )
        .unwrap()
        .permit
        .unwrap();
    let directory = config.directory.join("worker-b");
    std::fs::create_dir(&directory).unwrap();
    std::fs::set_permissions(
        &directory,
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    ChildConfig {
        role: "worker".into(),
        directory,
        audience: config.audience.clone(),
        certificate,
        authority_key: authority.public_key(),
        dispatch: offered.dispatch.unwrap(),
        policy: offered.delivery.unwrap(),
        permit,
        time,
        initialize: true,
        fault: String::new(),
        key_seed: 42,
        encryption_seed: 102,
        reply_seed: 104,
    }
}
#[test]
fn reconnect_process_concurrent_tenant_workers_replay_only_their_own_ciphertext() {
    let (_directory, mut alice, _, _) = seeded();
    let mut bob = second_tenant(&alice);
    let a = Process::start(&alice);
    let b = Process::start(&bob);
    assert_eq!(a.ready(), "synced");
    assert_eq!(b.ready(), "synced");
    drop(a);
    drop(b);
    for c in [&mut alice, &mut bob] {
        c.initialize = false;
        c.time += 121;
        let worker = Process::start(c);
        assert_eq!(worker.ready(), "synced");
        drop(worker);
    }
    let stores = [
        FileRelayOutbox::open(
            &alice.directory.join("outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap(),
        FileRelayOutbox::open(
            &bob.directory.join("outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap(),
    ];
    let entries = [
        stores[0]
            .entry(&alice.dispatch.submission.binding.job_id)
            .unwrap()
            .unwrap(),
        stores[1]
            .entry(&bob.dispatch.submission.binding.job_id)
            .unwrap()
            .unwrap(),
    ];
    alice.role = "authority".into();
    let service = Process::start(&alice);
    let address = service.ready();
    for (index, c) in [&alice, &bob].into_iter().enumerate() {
        let key = SigningKey::from_bytes(&[c.key_seed; 32]);
        let other = if index == 0 { &bob } else { &alice };
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Inspect {
                job_id: c.dispatch.submission.binding.job_id.clone(),
                manifest: entries[index].manifest.clone().unwrap(),
            },
        };
        let forbidden = NodeMessage::sign(
            &other.certificate,
            &SigningKey::from_bytes(&[other.key_seed; 32]),
            NodeAction::Recovery {
                request: request.clone(),
            },
            now().unwrap(),
        )
        .unwrap();
        assert_eq!(post_status(&address, &forbidden).0, 401);
        let grant = post_node(
            &address,
            &NodeMessage::sign(
                &c.certificate,
                &key,
                NodeAction::Recovery { request },
                now().unwrap(),
            )
            .unwrap(),
        )
        .recovery
        .unwrap()
        .grant
        .unwrap();
        for packet in &entries[index].packets {
            if let OutboxPacket::Terminal { packet } = packet {
                assert!(
                    packet
                        .open(
                            &c.dispatch.submission.binding,
                            &key.verifying_key().to_bytes(),
                            &RelayKey::from_seed(&[c.reply_seed; 32]).unwrap()
                        )
                        .is_ok()
                );
                assert!(
                    packet
                        .open(
                            &c.dispatch.submission.binding,
                            &key.verifying_key().to_bytes(),
                            &RelayKey::from_seed(&[other.reply_seed; 32]).unwrap()
                        )
                        .is_err()
                );
            }
            let request = RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish {
                    grant: grant.clone(),
                    packet: packet.clone(),
                },
            };
            assert_eq!(
                post_node(
                    &address,
                    &NodeMessage::sign(
                        &c.certificate,
                        &key,
                        NodeAction::Recovery { request },
                        now().unwrap()
                    )
                    .unwrap()
                )
                .recovery
                .unwrap()
                .status,
                RecoveryStatus::Recorded
            );
        }
        assert_eq!(
            std::fs::read(c.directory.join("invocations")).unwrap(),
            b"1"
        );
    }
}
#[test]
fn reconnect_process_cancellation_during_disconnect_allows_prefix_but_refuses_success() {
    let (directory, mut config, submission, _) = seeded();
    let worker = Process::start(&config);
    assert_eq!(worker.ready(), "synced");
    drop(worker);
    let outbox = FileRelayOutbox::open(
        &directory.path().join("outbox.sqlite"),
        OutboxLimits::default(),
    )
    .unwrap();
    let entry = outbox.entry(&submission.binding.job_id).unwrap().unwrap();
    config.role = "authority".into();
    config.fault = "cancelled".into();
    let service = Process::start(&config);
    let address = service.ready();
    let key = SigningKey::from_bytes(&[41; 32]);
    let request = RecoveryRequest {
        protocol: "abi.relay-recovery.v1".into(),
        action: RecoveryAction::Inspect {
            job_id: submission.binding.job_id,
            manifest: entry.manifest.unwrap(),
        },
    };
    let response = post_node(
        &address,
        &NodeMessage::sign(
            &config.certificate,
            &key,
            NodeAction::Recovery { request },
            now().unwrap(),
        )
        .unwrap(),
    )
    .recovery
    .unwrap();
    assert!(response.cancelled);
    let grant = response.grant.unwrap();
    for packet in entry.packets {
        let terminal = matches!(packet, OutboxPacket::Terminal { .. });
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Publish {
                grant: grant.clone(),
                packet,
            },
        };
        let message = NodeMessage::sign(
            &config.certificate,
            &key,
            NodeAction::Recovery { request },
            now().unwrap(),
        )
        .unwrap();
        let (status, body) = post_status(&address, &message);
        if terminal {
            assert_eq!(status, 401);
        } else {
            assert_eq!(status, 200);
            let reply: NodeReply = serde_json::from_slice(&body).unwrap();
            assert!(reply.recovery.unwrap().cancelled);
        }
    }
    assert_eq!(
        std::fs::read(directory.path().join("invocations")).unwrap(),
        b"1"
    );
}

#[test]
fn reconnect_process_http_replays_after_execution_expiry_with_current_credentials() {
    // Historical clock fixture avoids timing races; the real HTTP router checks current time.
    let (directory, mut config, submission, _) = seeded_at(now().unwrap() - 121);
    let worker = Process::start(&config);
    assert_eq!(worker.ready(), "synced");
    drop(worker);
    assert!(submission.binding.expires_at < now().unwrap());
    let outbox = FileRelayOutbox::open(
        &directory.path().join("outbox.sqlite"),
        OutboxLimits::default(),
    )
    .unwrap();
    let entry = outbox.entry(&submission.binding.job_id).unwrap().unwrap();
    config.role = "authority".into();
    let service = Process::start(&config);
    let address = service.ready();
    let key = SigningKey::from_bytes(&[41; 32]);
    let request = RecoveryRequest {
        protocol: "abi.relay-recovery.v1".into(),
        action: RecoveryAction::Inspect {
            job_id: submission.binding.job_id,
            manifest: entry.manifest.unwrap(),
        },
    };
    let response = post_node(
        &address,
        &NodeMessage::sign(
            &config.certificate,
            &key,
            NodeAction::Recovery { request },
            now().unwrap(),
        )
        .unwrap(),
    );
    assert!(response.permit.is_none());
    let grant = response.recovery.unwrap().grant.unwrap();
    for packet in entry.packets {
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Publish {
                grant: grant.clone(),
                packet,
            },
        };
        let reply = post_node(
            &address,
            &NodeMessage::sign(
                &config.certificate,
                &key,
                NodeAction::Recovery { request },
                now().unwrap(),
            )
            .unwrap(),
        );
        assert!(reply.permit.is_none());
        assert_eq!(reply.recovery.unwrap().status, RecoveryStatus::Recorded);
    }
    assert_eq!(
        std::fs::read(directory.path().join("invocations")).unwrap(),
        b"1"
    );
}

#[test]
fn reconnect_process_saved_completion_refuses_missing_or_admitted_control_history() {
    let (directory, config, _, _) = seeded();
    let worker = Process::start(&config);
    assert_eq!(worker.ready(), "synced");
    drop(worker);
    for admitted in [false, true] {
        let name = if admitted {
            "contradictory-admitted.log"
        } else {
            "missing-history.log"
        };
        let control = abi_worker::FileControlStore::create(
            &directory.path().join(name),
            abi_worker::FileControlStoreLimits::default(),
        )
        .unwrap();
        let mut runtime = RelayRuntime::new(
            config.certificate.clone(),
            SigningKey::from_bytes(&[41; 32]),
            RelayKey::from_seed(&[101; 32]).unwrap(),
            config.authority_key,
            "local-echo-v1",
            control,
            config.time,
        )
        .unwrap();
        if admitted {
            runtime
                .admit_recovery_fixture(&config.dispatch, config.time)
                .unwrap();
        }
        let outbox = FileRelayOutbox::open(
            &directory.path().join("outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap();
        match DurableRelayRuntime::new(runtime, outbox) {
            Ok(mut runtime) => assert!(runtime.reconcile_local(config.time).is_err()),
            Err(error) => assert!(matches!(error, Error::Denied)),
        }
        assert_eq!(
            std::fs::read(directory.path().join("invocations")).unwrap(),
            b"1"
        );
    }
}
