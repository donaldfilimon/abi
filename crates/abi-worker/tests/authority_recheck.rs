//! A refused or cancelling authority callback never reaches the provider.

#[allow(dead_code)]
mod support;

use abi_agent_runtime::{EchoProvider, ModelProvider, ModelRequest, RunContext, RuntimeError};
use abi_worker::{CancellationToken, InferenceOutcome, JobStatus, RecordedExecution, WorkerError};
use std::sync::atomic::{AtomicUsize, Ordering};
use support::{NOW, envelope, manager};

struct CountingProvider(AtomicUsize);
impl ModelProvider for CountingProvider {
    fn id(&self) -> &'static str {
        "counted-echo"
    }
    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        EchoProvider::new().run(request, run)
    }
}

#[test]
fn authority_refusal_leaves_admission_unresolved_and_never_dispatches() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    assert!(matches!(
        worker.infer_with_authority(
            &request,
            &provider,
            || NOW,
            |_| Err(WorkerError::Authentication("revoked".into())),
        ),
        Err(WorkerError::Authentication(_))
    ));
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    assert!(matches!(
        worker.recorded(&request),
        Ok(RecordedExecution::Admitted(_))
    ));
    assert!(matches!(
        worker.infer_with_authority(
            &request,
            &provider,
            || NOW,
            |_| panic!("existing admission must not request fresh authority"),
        ),
        Ok(InferenceOutcome::InProgress(_))
    ));
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_during_authority_callback_never_dispatches_and_records_failure() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    let cancel = CancellationToken::new();
    let InferenceOutcome::Completed(result) = worker
        .infer_observed(
            &request,
            &provider,
            || NOW,
            |_| {
                cancel.cancel();
                Ok(())
            },
            &cancel,
            |_| Ok(()),
        )
        .unwrap()
    else {
        panic!("cancellation must commit a terminal failure");
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks.len(), 0);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    assert_eq!(
        worker.infer(&request, &provider, || NOW).unwrap(),
        InferenceOutcome::Cached(result)
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}
