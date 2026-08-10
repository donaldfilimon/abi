//! Public authenticated worker-plane contract tests.

mod support;

use abi_wdbx_gateway::{Limits, TlsFiles};
use abi_worker::{
    Admission, CancellationRequest, ControlEvent, CoordinatorTrustStore, JobControlStore,
    JobEnvelope, JobRequest, JobStatus, MemoryControlStore, ResultChunk, ResultDigest, WorkerError,
    WorkerLimits, WorkerManager, WorkerTransportSecurity,
};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use support::{
    COORDINATOR, KEY_ID, NAMESPACE, NOW, PRINCIPAL, WORKER, envelope, envelope_for, identity,
    manager, manager_with, request, signing_key, trust,
};
use uuid::Uuid;

fn leased(admission: Admission) -> abi_worker::JobLease {
    match admission {
        Admission::Leased(lease) => lease,
        other => panic!("expected new lease, got {other:?}"),
    }
}

fn chunks() -> Vec<ResultChunk> {
    vec![
        ResultChunk {
            sequence: 0,
            bytes: b"hello ".to_vec(),
        },
        ResultChunk {
            sequence: 1,
            bytes: b"world".to_vec(),
        },
    ]
}

#[test]
fn signed_job_admits_once_and_exact_retry_reuses_the_lease() {
    let mut manager = manager();
    let envelope = envelope();
    let lease = leased(manager.submit(&envelope, NOW).expect("admitted"));
    assert_eq!(lease.job_id, envelope.job_id);
    assert_eq!(lease.worker_id, WORKER);
    assert!(lease.expires_at_unix_ms <= envelope.deadline_unix_ms);
    assert!(matches!(
        manager.submit(&envelope, NOW + 1).expect("retry"),
        Admission::ExistingLease(ref existing) if *existing == lease
    ));
    assert_eq!(manager.control_store().events().len(), 1);
}

#[test]
fn same_job_identity_with_changed_signed_fields_is_not_an_exact_retry() {
    let mut manager = manager();
    let original = envelope();
    manager.submit(&original, NOW).expect("original admitted");
    let changed_deadline = JobEnvelope::sign(
        original.job_id,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW,
        NOW + 20_000,
        original.nonce,
        request(),
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    assert!(matches!(
        manager.submit(&changed_deadline, NOW + 1),
        Err(WorkerError::Replay(_))
    ));
}

#[test]
fn request_input_complexity_is_bounded_before_canonicalization() {
    let mut input = json!(null);
    for _ in 0..66 {
        input = json!([input]);
    }
    assert!(matches!(
        JobRequest::new("model.infer", "revision-1", "infer", input, 8, 1),
        Err(WorkerError::Invalid(_))
    ));
}

#[test]
fn digest_deserialization_rejects_noncanonical_values() {
    assert!(serde_json::from_str::<ResultDigest>("\"abcd\"").is_err());
    assert!(serde_json::from_str::<ResultDigest>(&format!("\"{}\"", "AB".repeat(32))).is_err());
    assert!(serde_json::from_str::<ResultDigest>(&format!("\"{}\"", "ab".repeat(32))).is_ok());
}

#[test]
fn signature_and_signed_payload_tampering_fail_before_admission() {
    let mut manager = manager();
    let mut bad_signature = envelope();
    bad_signature.signature[0] ^= 1;
    assert!(matches!(
        manager.submit(&bad_signature, NOW),
        Err(WorkerError::Authentication(_))
    ));

    let mut changed_payload = envelope();
    changed_payload.request.input = json!({"prompt": "tampered"});
    assert!(matches!(
        manager.submit(&changed_payload, NOW),
        Err(WorkerError::Authentication(_))
    ));
    assert_eq!(manager.control_store().events(), []);
}

#[test]
fn audience_namespace_and_coordinator_identity_fail_closed() {
    for (worker, namespace, coordinator) in [
        ("worker-2", NAMESPACE, COORDINATOR),
        (WORKER, "personal-production", COORDINATOR),
        (WORKER, NAMESPACE, "other-coordinator"),
    ] {
        let request = JobEnvelope::sign(
            Uuid::new_v4(),
            PRINCIPAL,
            worker,
            namespace,
            coordinator,
            NOW,
            NOW + 1_000,
            Uuid::new_v4(),
            request(),
            KEY_ID,
            &signing_key(),
        )
        .unwrap();
        assert!(matches!(
            manager().submit(&request, NOW),
            Err(WorkerError::Authentication(_))
        ));
    }
}

#[test]
fn coordinator_key_identity_is_bound_to_its_configured_audience() {
    let key = signing_key();
    let mut wrong_binding = CoordinatorTrustStore::new();
    wrong_binding
        .insert(KEY_ID, "other-coordinator", key.verifying_key().as_bytes())
        .unwrap();
    let mut manager = WorkerManager::new(
        identity(),
        [support::capability()],
        wrong_binding,
        WorkerLimits::default(),
        MemoryControlStore::new(),
    )
    .unwrap();
    assert!(matches!(
        manager.submit(&envelope(), NOW),
        Err(WorkerError::Authentication(_))
    ));
}

#[test]
fn unknown_and_revision_incompatible_capabilities_are_refused() {
    for request in [
        JobRequest::new("unknown", "revision-1", "infer", json!({}), 8, 1).unwrap(),
        JobRequest::new("model.infer", "revision-2", "infer", json!({}), 8, 1).unwrap(),
    ] {
        assert!(matches!(
            manager().submit(&envelope_for(Uuid::new_v4(), Uuid::new_v4(), request), NOW),
            Err(WorkerError::Capability(_))
        ));
    }
}

#[test]
fn canonical_request_digest_is_derived_inside_the_manager() {
    let job_id = Uuid::new_v4();
    let nonce = Uuid::new_v4();
    let first = JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"z": 1, "nested": {"b": 2, "a": 1}}),
        256,
        4,
    )
    .unwrap();
    let second = JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"nested": {"a": 1, "b": 2}, "z": 1}),
        256,
        4,
    )
    .unwrap();
    let mut manager = manager();
    let lease = leased(
        manager
            .submit(&envelope_for(job_id, nonce, first), NOW)
            .unwrap(),
    );
    let result = manager
        .complete(&lease, JobStatus::Succeeded, chunks(), NOW + 10)
        .unwrap();
    let cached = manager
        .submit(
            &envelope_for(Uuid::new_v4(), Uuid::new_v4(), second),
            NOW + 20,
        )
        .unwrap();
    assert!(matches!(cached, Admission::Cached(ref found) if *found == result));
    assert_eq!(manager.control_store().events().len(), 2);
}

#[test]
fn idempotency_never_reuses_results_across_principals() {
    let first = envelope();
    let mut manager = manager();
    let lease = leased(manager.submit(&first, NOW).unwrap());
    manager
        .complete(&lease, JobStatus::Succeeded, chunks(), NOW + 1)
        .unwrap();
    let second = JobEnvelope::sign(
        Uuid::new_v4(),
        "service-account-2",
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 2,
        NOW + 10_000,
        Uuid::new_v4(),
        request(),
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    assert!(matches!(
        manager.submit(&second, NOW + 2).unwrap(),
        Admission::Leased(_)
    ));
}

#[test]
fn job_and_nonce_replay_conflicts_are_rejected() {
    let accepted = envelope();
    let mut manager = manager();
    manager.submit(&accepted, NOW).unwrap();

    let conflicting_job = envelope_for(accepted.job_id, Uuid::new_v4(), request());
    assert!(matches!(
        manager.submit(&conflicting_job, NOW),
        Err(WorkerError::Replay(_))
    ));
    let conflicting_nonce = envelope_for(Uuid::new_v4(), accepted.nonce, request());
    assert!(matches!(
        manager.submit(&conflicting_nonce, NOW),
        Err(WorkerError::Replay(_))
    ));
}

#[test]
fn deadlines_future_skew_and_ambiguous_expired_leases_fail_closed() {
    let expired = JobEnvelope::sign(
        Uuid::new_v4(),
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW - 2_000,
        NOW - 1_000,
        Uuid::new_v4(),
        request(),
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    assert!(matches!(
        manager().submit(&expired, NOW),
        Err(WorkerError::Deadline(_))
    ));

    let future = JobEnvelope::sign(
        Uuid::new_v4(),
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 31_000,
        NOW + 40_000,
        Uuid::new_v4(),
        request(),
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    assert!(matches!(
        manager().submit(&future, NOW),
        Err(WorkerError::Deadline(_))
    ));

    let accepted = envelope();
    let limits = WorkerLimits {
        lease_ms: 1,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(limits, MemoryControlStore::new());
    manager.submit(&accepted, NOW).unwrap();
    assert!(matches!(
        manager.submit(&accepted, NOW + 2),
        Err(WorkerError::State(_))
    ));
}

#[test]
fn active_principal_capability_and_output_quotas_are_enforced() {
    let limits = WorkerLimits {
        active_jobs: 1,
        active_jobs_per_principal: 1,
        result_bytes: 256,
        stream_chunks: 4,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(limits, MemoryControlStore::new());
    manager.submit(&envelope(), NOW).unwrap();
    let distinct = JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"prompt": "different"}),
        256,
        4,
    )
    .unwrap();
    assert!(matches!(
        manager.submit(&envelope_for(Uuid::new_v4(), Uuid::new_v4(), distinct), NOW),
        Err(WorkerError::Quota(_))
    ));

    let oversized =
        JobRequest::new("model.infer", "revision-1", "infer", json!({}), 257, 4).unwrap();
    assert!(matches!(
        manager_with(limits, MemoryControlStore::new()).submit(
            &envelope_for(Uuid::new_v4(), Uuid::new_v4(), oversized),
            NOW
        ),
        Err(WorkerError::Quota(_))
    ));
}

#[test]
fn completion_requires_exact_live_lease_and_bounded_contiguous_chunks() {
    let envelope = envelope();
    let mut manager = manager();
    let lease = leased(manager.submit(&envelope, NOW).unwrap());
    let mut wrong = lease.clone();
    wrong.lease_id = Uuid::new_v4();
    assert!(matches!(
        manager.complete(&wrong, JobStatus::Succeeded, chunks(), NOW + 1),
        Err(WorkerError::State(_))
    ));
    let unordered = vec![ResultChunk {
        sequence: 1,
        bytes: vec![1],
    }];
    assert!(matches!(
        manager.complete(&lease, JobStatus::Succeeded, unordered, NOW + 1),
        Err(WorkerError::Invalid(_))
    ));
    let oversized = vec![ResultChunk {
        sequence: 0,
        bytes: vec![0; 257],
    }];
    assert!(matches!(
        manager.complete(&lease, JobStatus::Succeeded, oversized, NOW + 1),
        Err(WorkerError::Quota(_))
    ));
    let result = manager
        .complete(&lease, JobStatus::Succeeded, chunks(), NOW + 1)
        .expect("valid completion");
    assert_eq!(result.request_digest, lease.request_digest);
    assert_eq!(result.result_digest.as_str().len(), 64);
    assert!(result.verify_digest());
    let mut tampered_result = result.clone();
    tampered_result.chunks[0].bytes.push(0);
    assert!(!tampered_result.verify_digest());
    assert!(matches!(
        manager.complete(&lease, JobStatus::Succeeded, chunks(), NOW + 2),
        Err(WorkerError::State(_))
    ));
}

#[test]
fn lease_and_job_deadlines_bound_completion() {
    let limits = WorkerLimits {
        lease_ms: 5,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(limits, MemoryControlStore::new());
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    assert!(matches!(
        manager.complete(&lease, JobStatus::Succeeded, chunks(), NOW + 6),
        Err(WorkerError::Deadline(_))
    ));
}

#[test]
fn signed_cancellation_is_exact_durable_and_idempotent() {
    let mut manager = manager();
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    let executor_cancellation = manager
        .cancellation_token(&lease)
        .expect("active lease exposes its cancellation token");
    assert!(!executor_cancellation.is_cancelled());
    let cancellation = CancellationRequest::sign(
        lease.job_id,
        lease.request_digest.clone(),
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 1,
        "operator requested stop",
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    manager.cancel(&cancellation, NOW + 1).unwrap();
    assert!(executor_cancellation.is_cancelled());
    manager.cancel(&cancellation, NOW + 1).unwrap();
    assert!(manager.is_cancelled(lease.job_id, &lease.request_digest));
    assert!(matches!(
        manager.complete(&lease, JobStatus::Succeeded, chunks(), NOW + 2),
        Err(WorkerError::State(_))
    ));
    assert_eq!(manager.health(NOW + 2).cancelled_jobs, 1);
    assert_eq!(manager.control_store().events().len(), 2);
}

#[test]
fn tampered_or_future_cancellation_is_rejected() {
    let mut manager = manager();
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    let mut cancellation = CancellationRequest::sign(
        lease.job_id,
        lease.request_digest,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 1,
        "stop",
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    cancellation.reason = "tampered".into();
    assert!(matches!(
        manager.cancel(&cancellation, NOW + 1),
        Err(WorkerError::Authentication(_))
    ));

    let future = CancellationRequest::sign(
        lease.job_id,
        cancellation.request_digest,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 31_000,
        "future stop",
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    assert!(matches!(
        manager.cancel(&future, NOW),
        Err(WorkerError::Deadline(_))
    ));
}

#[test]
fn completed_history_reopens_and_serves_cached_retry_without_execution() {
    let envelope = envelope();
    let mut manager = manager();
    let lease = leased(manager.submit(&envelope, NOW).unwrap());
    let result = manager
        .complete(&lease, JobStatus::Succeeded, chunks(), NOW + 1)
        .unwrap();
    let store = manager.control_store().clone();
    let mut reopened = manager_with(WorkerLimits::default(), store);
    assert!(matches!(
        reopened.submit(&envelope, NOW + 2).unwrap(),
        Admission::Cached(ref cached) if *cached == result
    ));
    assert_eq!(reopened.health(NOW + 2).completed_jobs, 1);
}

#[test]
fn replay_revalidates_active_concurrency_and_terminal_capacity() {
    let limits = WorkerLimits {
        active_jobs: 2,
        active_jobs_per_principal: 2,
        control_events: 4,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(limits, MemoryControlStore::new());
    manager.submit(&envelope(), NOW).unwrap();
    let distinct = JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"prompt": "different"}),
        256,
        4,
    )
    .unwrap();
    manager
        .submit(&envelope_for(Uuid::new_v4(), Uuid::new_v4(), distinct), NOW)
        .unwrap();
    let store = manager.control_store().clone();
    let narrower = WorkerLimits {
        active_jobs: 1,
        active_jobs_per_principal: 1,
        control_events: 2,
        ..WorkerLimits::default()
    };
    assert!(matches!(
        WorkerManager::new(
            identity(),
            [support::capability()],
            trust(),
            narrower,
            store
        ),
        Err(WorkerError::Quota(_))
    ));
}

#[test]
fn malformed_control_history_fails_reconstruction() {
    let mut store = MemoryControlStore::new();
    let digest = ResultDigest::parse("00".repeat(32)).unwrap();
    store
        .append(&ControlEvent::Cancelled {
            job_id: Uuid::new_v4(),
            request_digest: digest,
            cancelled_at_unix_ms: NOW,
        })
        .unwrap();
    assert!(matches!(
        WorkerManager::new(
            identity(),
            [support::capability()],
            trust(),
            WorkerLimits::default(),
            store
        ),
        Err(WorkerError::ControlStore(_))
    ));
}

#[test]
fn tampered_result_digest_in_control_history_fails_reconstruction() {
    let mut manager = manager();
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    manager
        .complete(&lease, JobStatus::Succeeded, chunks(), NOW + 1)
        .unwrap();
    let mut events = manager.control_store().events().to_vec();
    let ControlEvent::Completed { result } = &mut events[1] else {
        panic!("second event is completion");
    };
    result.result_digest = ResultDigest::parse("00".repeat(32)).unwrap();
    let mut tampered = MemoryControlStore::new();
    for event in events {
        tampered.append(&event).unwrap();
    }
    assert!(matches!(
        WorkerManager::new(
            identity(),
            [support::capability()],
            trust(),
            WorkerLimits::default(),
            tampered
        ),
        Err(WorkerError::ControlStore(_))
    ));
}

#[test]
fn completion_history_after_lease_expiry_fails_even_with_a_valid_digest() {
    #[derive(serde::Serialize)]
    struct CanonicalResult<'a> {
        job_id: Uuid,
        request_digest: &'a str,
        status: JobStatus,
        chunks: &'a [ResultChunk],
        completed_at_unix_ms: i64,
    }

    let limits = WorkerLimits {
        lease_ms: 5,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(limits, MemoryControlStore::new());
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    let after_lease = lease.expires_at_unix_ms + 1;
    let chunks = chunks();
    let canonical = serde_json::to_vec(&CanonicalResult {
        job_id: lease.job_id,
        request_digest: lease.request_digest.as_str(),
        status: JobStatus::Succeeded,
        chunks: &chunks,
        completed_at_unix_ms: after_lease,
    })
    .unwrap();
    let result = abi_worker::JobResult {
        job_id: lease.job_id,
        request_digest: lease.request_digest,
        status: JobStatus::Succeeded,
        chunks,
        result_digest: ResultDigest::parse(format!("{:x}", Sha256::digest(canonical))).unwrap(),
        completed_at_unix_ms: after_lease,
    };
    let mut store = manager.control_store().clone();
    store.append(&ControlEvent::Completed { result }).unwrap();
    assert!(matches!(
        WorkerManager::new(identity(), [support::capability()], trust(), limits, store),
        Err(WorkerError::ControlStore(_))
    ));
}

#[test]
fn admission_reserves_control_capacity_for_a_terminal_event() {
    let no_terminal_room = WorkerLimits {
        control_events: 1,
        ..WorkerLimits::default()
    };
    assert!(matches!(
        manager_with(no_terminal_room, MemoryControlStore::new()).submit(&envelope(), NOW),
        Err(WorkerError::Quota(_))
    ));

    let terminal_room = WorkerLimits {
        control_events: 2,
        ..WorkerLimits::default()
    };
    let mut manager = manager_with(terminal_room, MemoryControlStore::new());
    let lease = leased(manager.submit(&envelope(), NOW).unwrap());
    let cancellation = CancellationRequest::sign(
        lease.job_id,
        lease.request_digest,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 1,
        "stop",
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    manager.cancel(&cancellation, NOW + 1).unwrap();
    assert_eq!(manager.control_store().events().len(), 2);
}

#[test]
fn health_is_stably_ordered_and_counts_each_terminal_class() {
    let mut manager = manager();
    let first = leased(manager.submit(&envelope(), NOW).unwrap());
    manager
        .complete(&first, JobStatus::Failed, chunks(), NOW + 1)
        .unwrap();
    let next_request = JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"prompt": "second"}),
        256,
        4,
    )
    .unwrap();
    let second = leased(
        manager
            .submit(
                &envelope_for(Uuid::new_v4(), Uuid::new_v4(), next_request),
                NOW + 2,
            )
            .unwrap(),
    );
    let cancellation = CancellationRequest::sign(
        second.job_id,
        second.request_digest,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW + 3,
        "stop",
        KEY_ID,
        &signing_key(),
    )
    .unwrap();
    manager.cancel(&cancellation, NOW + 3).unwrap();
    let health = manager.health(NOW + 4);
    assert_eq!(health.worker_id, WORKER);
    assert_eq!(health.capabilities[0].id, "model.infer");
    assert_eq!(health.active_jobs, 0);
    assert_eq!(health.completed_jobs, 1);
    assert_eq!(health.cancelled_jobs, 1);
}

#[cfg(unix)]
#[test]
fn worker_transport_reuses_owner_protected_mandatory_mtls_loader() {
    use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair, KeyUsagePurpose};
    use std::os::unix::fs::PermissionsExt as _;

    let root = abi_foundation::temp_path::temp_file_path("abi_worker_mtls", "scratch");
    std::fs::create_dir_all(&root).unwrap();
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyCertSign,
    ];
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let server_key_pair = KeyPair::generate().unwrap();
    let server = CertificateParams::new(vec!["localhost".into()])
        .unwrap()
        .signed_by(&server_key_pair, &ca, &ca_key)
        .unwrap();
    let certificate = root.join("server.pem");
    let private_key = root.join("server-key.pem");
    let client_ca = root.join("client-ca.pem");
    std::fs::write(&certificate, server.pem()).unwrap();
    std::fs::write(&private_key, server_key_pair.serialize_pem()).unwrap();
    std::fs::write(&client_ca, ca.pem()).unwrap();
    std::fs::set_permissions(&private_key, std::fs::Permissions::from_mode(0o600)).unwrap();

    let security = WorkerTransportSecurity {
        tls: TlsFiles {
            certificate: Some(certificate),
            private_key: Some(private_key),
            client_ca: Some(client_ca),
        },
        transport_limits: Limits::default(),
    };
    security.validate().expect("gateway mTLS loader validates");
    let worker_limits = security.worker_limits().expect("shared bounds derive");
    assert_eq!(
        worker_limits.envelope_bytes,
        Limits::default().message_bytes
    );

    let no_client_identity = WorkerTransportSecurity {
        tls: TlsFiles::default(),
        transport_limits: Limits::default(),
    };
    assert!(matches!(
        no_client_identity.validate(),
        Err(WorkerError::Transport(_))
    ));
    let _ = std::fs::remove_dir_all(root);
}
