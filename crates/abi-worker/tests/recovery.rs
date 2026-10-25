//! Historical evidence never renews execution authority.
#[allow(dead_code)]
mod support;
use abi_worker::{Admission, JobStatus, RecordedExecution, WorkerLimits, WorkerManager};
use support::*;

#[test]
fn completed_lookup_after_deadline_is_read_only() {
    let mut worker = manager();
    let envelope = envelope();
    let Admission::Leased(lease) = worker.submit(&envelope, NOW).unwrap() else {
        panic!()
    };
    let result = worker
        .complete(&lease, JobStatus::Succeeded, vec![], NOW + 1)
        .unwrap();
    let before = worker.control_store().events().len();
    assert!(worker.submit(&envelope, NOW + 100_000).is_err());
    assert_eq!(
        worker.recorded(&envelope).unwrap(),
        RecordedExecution::Completed(result)
    );
    assert_eq!(worker.control_store().events().len(), before);
}

#[test]
fn historical_model_revision_does_not_require_current_model() {
    let mut worker = manager();
    let envelope = envelope();
    worker.submit(&envelope, NOW).unwrap();
    let mut replacement = capability();
    replacement.id = "new.model".into();
    replacement.revision = "revision-2".into();
    let worker = WorkerManager::new(
        identity(),
        [replacement],
        trust(),
        WorkerLimits::default(),
        worker.control_store().clone(),
    )
    .unwrap();
    assert!(matches!(
        worker.recorded(&envelope).unwrap(),
        RecordedExecution::Admitted(_)
    ));
}

#[test]
fn changed_envelope_principal_nonce_or_scope_refuses() {
    let mut worker = manager();
    let envelope = envelope();
    worker.submit(&envelope, NOW).unwrap();
    for field in ["principal", "nonce", "namespace", "deadline_unix_ms"] {
        let mut changed = serde_json::to_value(&envelope).unwrap();
        changed[field] = match field {
            "nonce" => serde_json::json!(uuid::Uuid::new_v4()),
            "deadline_unix_ms" => serde_json::json!(NOW + 20_000),
            _ => serde_json::json!("foreign"),
        };
        let changed: abi_worker::JobEnvelope = serde_json::from_value(changed).unwrap();
        let changed = abi_worker::JobEnvelope::sign(
            changed.job_id,
            changed.principal,
            changed.audience,
            changed.namespace,
            changed.coordinator_audience,
            changed.submitted_at_unix_ms,
            changed.deadline_unix_ms,
            changed.nonce,
            changed.request,
            KEY_ID,
            &signing_key(),
        )
        .unwrap();
        assert!(worker.recorded(&changed).is_err());
    }
}

#[test]
fn replayed_admission_can_be_abandoned_once_without_readmission() {
    let limits = WorkerLimits {
        active_jobs: 1,
        ..WorkerLimits::default()
    };
    let mut worker = manager_with(limits, abi_worker::MemoryControlStore::default());
    let envelope = envelope();
    worker.submit(&envelope, NOW).unwrap();
    let mut recovered = manager_with(limits, worker.control_store().clone());
    assert_eq!(
        recovered
            .recover_interrupted(&envelope, NOW + 100_000)
            .unwrap(),
        RecordedExecution::Unresolved
    );
    assert_eq!(recovered.control_store().events().len(), 2);
    assert_eq!(
        recovered
            .recover_interrupted(&envelope, NOW + 100_001)
            .unwrap(),
        RecordedExecution::Unresolved
    );
    assert_eq!(recovered.control_store().events().len(), 2);
    assert_eq!(recovered.health(NOW).unresolved_jobs, 1);
    assert_eq!(recovered.health(NOW).active_jobs, 0);
    assert!(recovered.submit(&envelope, NOW).is_err());
    let duplicate = envelope_for(
        uuid::Uuid::new_v4(),
        uuid::Uuid::new_v4(),
        envelope.request.clone(),
    );
    assert!(recovered.submit(&duplicate, NOW).is_err());
    let mut fresh = request();
    fresh.input = serde_json::json!({"prompt":"different"});
    recovered
        .submit(
            &envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), fresh),
            NOW,
        )
        .unwrap();
}

#[test]
fn current_process_admission_cannot_be_abandoned_as_replayed() {
    let mut worker = manager();
    let envelope = envelope();
    worker.submit(&envelope, NOW).unwrap();
    assert!(
        worker
            .recover_interrupted(&envelope, NOW + 100_000)
            .is_err()
    );
    assert_eq!(worker.control_store().events().len(), 1);
}

#[cfg(unix)]
#[test]
fn file_reopen_preserves_completed_and_unresolved_evidence() {
    use abi_worker::{FileControlStore, FileControlStoreLimits};
    let path = abi_foundation::temp_path::temp_file_path("abi-worker-recovery", "log");
    let open_manager = |store| {
        WorkerManager::new(
            identity(),
            [capability()],
            trust(),
            WorkerLimits::default(),
            store,
        )
        .unwrap()
    };
    let mut worker =
        open_manager(FileControlStore::create(&path, FileControlStoreLimits::default()).unwrap());
    let completed = envelope();
    let Admission::Leased(lease) = worker.submit(&completed, NOW).unwrap() else {
        panic!()
    };
    let result = worker
        .complete(&lease, JobStatus::Succeeded, vec![], NOW + 1)
        .unwrap();
    let mut other_request = request();
    other_request.input = serde_json::json!({"prompt":"interrupted"});
    let interrupted = envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), other_request);
    worker.submit(&interrupted, NOW).unwrap();
    drop(worker);
    let mut worker =
        open_manager(FileControlStore::open(&path, FileControlStoreLimits::default()).unwrap());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        worker.recorded(&completed).unwrap(),
        RecordedExecution::Completed(result)
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    worker
        .recover_interrupted(&interrupted, NOW + 100_000)
        .unwrap();
    drop(worker);
    let worker =
        open_manager(FileControlStore::open(&path, FileControlStoreLimits::default()).unwrap());
    assert_eq!(
        worker.recorded(&interrupted).unwrap(),
        RecordedExecution::Unresolved
    );
    assert_eq!(worker.health(NOW).completed_jobs, 1);
    assert_eq!(worker.health(NOW).unresolved_jobs, 1);
    drop(worker);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unresolved_history_rejects_contradictory_terminal_events() {
    use abi_worker::{ControlEvent, JobControlStore};
    let mut worker = manager();
    let envelope = envelope();
    let Admission::Leased(lease) = worker.submit(&envelope, NOW).unwrap() else {
        panic!()
    };
    let mut worker = manager_with(WorkerLimits::default(), worker.control_store().clone());
    worker.recover_interrupted(&envelope, NOW + 1).unwrap();
    let mut store = worker.control_store().clone();
    store
        .append(&ControlEvent::Cancelled {
            job_id: lease.job_id,
            request_digest: lease.request_digest,
            cancelled_at_unix_ms: NOW + 2,
        })
        .unwrap();
    assert!(
        WorkerManager::new(
            identity(),
            [capability()],
            trust(),
            WorkerLimits::default(),
            store
        )
        .is_err()
    );
}
