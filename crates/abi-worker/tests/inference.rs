//! Authenticated inference executes once and retains bounded durable results.

#[allow(dead_code)]
mod support;

use abi_agent_runtime::{
    EchoProvider, ModelProvider, ModelRequest, RunContext, RuntimeError, ScriptedProvider, ToolCall,
};
use abi_worker::{Admission, InferenceOutcome, JobStatus};
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
fn fresh_inference_executes_once_and_exact_retry_is_cached() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    let InferenceOutcome::Completed(result) = worker.infer(&request, &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Succeeded);
    assert_eq!(result.chunks[0].bytes, b"hello");
    assert!(result.verify_digest());
    assert_eq!(
        worker.infer(&request, &provider, || NOW + 1).unwrap(),
        InferenceOutcome::Cached(result)
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}

#[test]
fn previously_admitted_inference_never_reexecutes() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    let Admission::Leased(lease) = worker.submit(&request, NOW).unwrap() else {
        panic!("leased")
    };
    assert_eq!(
        worker.infer(&request, &provider, || NOW + 1).unwrap(),
        InferenceOutcome::InProgress(lease)
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_signature_never_invokes_provider() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let mut request = envelope();
    request.principal = "other-principal".into();
    assert!(manager().infer(&request, &provider, || NOW).is_err());
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

#[test]
fn elapsed_lease_before_dispatch_never_invokes_provider() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let mut clock_calls = 0;
    assert!(
        manager()
            .infer(&envelope(), &provider, || {
                clock_calls += 1;
                if clock_calls == 1 { NOW } else { NOW + 20_000 }
            })
            .is_err()
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

#[test]
fn authority_callback_consuming_lease_never_dispatches_or_readmits() {
    for after_guard in [NOW - 1, NOW + 10_000, NOW + 20_000] {
        let now = std::cell::Cell::new(NOW);
        let provider = CountingProvider(AtomicUsize::new(0));
        let request = envelope();
        let mut worker = manager();
        assert!(matches!(
            worker.infer_with_authority(
                &request,
                &provider,
                || now.get(),
                |_| {
                    now.set(after_guard);
                    Ok(())
                },
            ),
            Err(abi_worker::WorkerError::Deadline(_))
        ));
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);
        assert!(matches!(
            worker
                .infer_with_authority(
                    &request,
                    &provider,
                    || NOW,
                    |_| panic!("existing admission must not request fresh authority"),
                )
                .unwrap(),
            InferenceOutcome::InProgress(_)
        ));
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn authority_callback_with_remaining_lease_executes_once() {
    let now = std::cell::Cell::new(NOW);
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    let InferenceOutcome::Completed(result) = worker
        .infer_with_authority(
            &request,
            &provider,
            || now.get(),
            |_| {
                now.set(NOW + 1);
                Ok(())
            },
        )
        .unwrap()
    else {
        panic!("valid authority must complete")
    };
    assert_eq!(result.status, JobStatus::Succeeded);
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}

#[test]
fn signed_output_limit_refuses_oversized_delta_without_success() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let mut input = support::request();
    input.max_result_bytes = 2;
    let request = support::envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), input);
    let InferenceOutcome::Completed(result) = manager().infer(&request, &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks, []);
}

#[test]
fn unsupported_operation_is_never_sent_to_provider() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let mut input = support::request();
    input.operation = "shell.execute".into();
    let request = support::envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), input);
    let InferenceOutcome::Completed(result) = manager().infer(&request, &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}

#[test]
fn provider_tool_call_refuses_the_run_and_never_returns_success() {
    let provider = ScriptedProvider::new("tools")
        .with_tool_call(ToolCall::new("call-1", "shell", "danger"))
        .with_text("must not be retained");
    let InferenceOutcome::Completed(result) =
        manager().infer(&envelope(), &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks, []);
}

#[test]
fn provider_diagnostics_are_not_persisted_as_result_content() {
    let provider = ScriptedProvider::new("failure").with_failure(RuntimeError::Provider {
        provider: "sensitive".into(),
        message: "SECRET_CANARY".into(),
    });
    let InferenceOutcome::Completed(result) =
        manager().infer(&envelope(), &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks, []);
    assert!(
        !serde_json::to_string(&result)
            .unwrap()
            .contains("SECRET_CANARY")
    );
}

#[test]
fn final_reported_token_overshoot_never_persists_success() {
    struct FinalUsage;
    impl ModelProvider for FinalUsage {
        fn id(&self) -> &'static str {
            "final-usage"
        }
        fn run(
            &self,
            _request: &ModelRequest,
            run: &mut RunContext<'_>,
        ) -> Result<(), RuntimeError> {
            let _ = run.emit(&abi_agent_runtime::ModelEvent::text("a"));
            run.record_output_tokens(5);
            Ok(())
        }
    }
    let InferenceOutcome::Completed(result) =
        manager().infer(&envelope(), &FinalUsage, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
}

#[test]
fn chunk_limit_retains_only_admitted_prefix_and_reports_failure() {
    let provider = ScriptedProvider::new("chunks")
        .with_text("a")
        .with_text("b")
        .with_text("c");
    let mut input = support::request();
    input.max_stream_chunks = 1;
    let request = support::envelope_for(uuid::Uuid::new_v4(), uuid::Uuid::new_v4(), input);
    let InferenceOutcome::Completed(result) = manager().infer(&request, &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].bytes, b"a");
}

#[test]
fn completion_after_deadline_is_not_reexecuted() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    assert!(
        worker
            .infer(&request, &provider, || {
                if provider.0.load(Ordering::SeqCst) == 0 {
                    NOW
                } else {
                    NOW + 20_000
                }
            })
            .is_err()
    );
    assert!(matches!(
        worker.infer(&request, &provider, || NOW + 1).unwrap(),
        InferenceOutcome::InProgress(_)
    ));
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}

#[cfg(unix)]
#[test]
fn committed_inference_is_cached_after_real_store_reopen() {
    use abi_foundation::temp_path::temp_file_path;
    use abi_worker::{FileControlStore, FileControlStoreLimits, WorkerLimits, WorkerManager};
    let path = temp_file_path("worker-infer-reopen", "log");
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = WorkerManager::new(
        support::identity(),
        [support::capability()],
        support::trust(),
        WorkerLimits::default(),
        FileControlStore::create(&path, FileControlStoreLimits::default()).unwrap(),
    )
    .unwrap();
    let InferenceOutcome::Completed(result) = worker.infer(&request, &provider, || NOW).unwrap()
    else {
        panic!("completed")
    };
    drop(worker);
    let mut worker = WorkerManager::new(
        support::identity(),
        [support::capability()],
        support::trust(),
        WorkerLimits::default(),
        FileControlStore::open(&path, FileControlStoreLimits::default()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        worker.infer(&request, &provider, || NOW + 1).unwrap(),
        InferenceOutcome::Cached(result)
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    drop(worker);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn observer_backpressure_refusal_is_durable_and_never_replays_output() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let request = envelope();
    let mut worker = manager();
    let token = abi_worker::CancellationToken::new();
    let mut observed = 0;
    let outcome = worker
        .infer_observed(
            &request,
            &provider,
            || NOW,
            |_| Ok(()),
            &token,
            |_| {
                observed += 1;
                Err(abi_worker::WorkerError::State("transport full".into()))
            },
        )
        .unwrap();
    let InferenceOutcome::Completed(result) = outcome else {
        panic!("completed failure");
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(result.chunks, []);
    assert!(token.is_cancelled());
    assert_eq!(observed, 1);
    assert_eq!(
        worker
            .infer_observed(
                &request,
                &provider,
                || NOW + 1,
                |_| Ok(()),
                &abi_worker::CancellationToken::new(),
                |_| panic!("cached replay must not emit frames")
            )
            .unwrap(),
        InferenceOutcome::Cached(result)
    );
    assert_eq!(provider.0.load(Ordering::SeqCst), 1);
}
#[test]
fn pre_cancelled_observed_inference_never_calls_provider() {
    let provider = CountingProvider(AtomicUsize::new(0));
    let token = abi_worker::CancellationToken::new();
    token.cancel();
    let InferenceOutcome::Completed(result) = manager()
        .infer_observed(
            &envelope(),
            &provider,
            || NOW,
            |_| Ok(()),
            &token,
            |_| panic!("no frames"),
        )
        .unwrap()
    else {
        panic!("completed failure");
    };
    assert_eq!(result.status, JobStatus::Failed);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
}
