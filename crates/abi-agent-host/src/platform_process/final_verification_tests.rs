//! Deterministic cancellation/time changes during the final actual OS observation.
use super::*;
use abi_agent_runtime::CancellationToken;
use abi_capability::{
    Principal, PrincipalKind, lifecycle::AuthorizationBinding, source::SourceRegistry,
};
use std::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}
fn execution(compiled: &CompiledSource) -> (Request, PreparedExecution, ToolCall) {
    let principal = Principal::new("actor", PrincipalKind::HumanSubject).unwrap();
    let call = ToolCall::new("call", "process_cwd", "{}");
    let src = compiled.source();
    let id = commitment(&(
        "abi.host.execution.v2",
        "attempt",
        &principal,
        &src.resource_scope,
        &src.id,
        &src.version,
        &call.id,
        &call.name,
    ));
    let request = Request::new(
        "attempt",
        id.to_string().trim_start_matches("sha256:"),
        principal,
        src.resource_scope.clone(),
        &src.id,
        &src.version,
        compiled.package().digest,
        commitment(&json!({})),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let binding = AuthorizationBinding {
        request_digest: request.request_digest,
        principal_digest: commitment(&request.principal),
        scope_digest: commitment(&request.scope),
        package_digest: request.package_digest,
        grant_digest: Digest::from_bytes([6; 32]),
        policy_digest: commitment(&("abi.policy.v1", "p", "g", "s", 0_u64)),
        schema_digest: compiled.schema_digest(),
        precondition_digest: Digest::from_bytes([7; 32]),
        compensation_origin: None,
        required_approval: ApprovalLevel::A1Actor,
    };
    (request, PreparedExecution::after_commit(id, binding), call)
}
#[test]
fn final_verification_cancellation_deadline_or_clock_change_never_publishes_receipt() {
    for fault in ["cancel", "deadline", "rollback", "expired"] {
        let path =
            std::env::temp_dir().join(format!("abi-final-verify-{}-{fault}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let mut plan = ProcessPlan::working_directory(&path, ProcessLimits::default()).unwrap();
        let token = CancellationToken::new();
        let clock = Arc::new(TestClock(AtomicU64::new(100)));
        let calls = Arc::new(AtomicUsize::new(0));
        let deadline = Instant::now() + Duration::from_secs(1);
        let (stop, time, count) = (token.clone(), clock.clone(), calls.clone());
        plan.pinned.after_verify = Some(Arc::new(move || {
            // Bind observation, pre-spawn observation, then final observation.
            if count.fetch_add(1, Ordering::SeqCst) == 2 {
                match fault {
                    "cancel" => stop.cancel(),
                    "deadline" => std::thread::sleep(
                        deadline.saturating_duration_since(Instant::now())
                            + Duration::from_millis(1),
                    ),
                    "rollback" => time.0.store(99, Ordering::SeqCst),
                    "expired" => time.0.store(1100, Ordering::SeqCst),
                    _ => unreachable!(),
                }
            }
        }));
        let mut src: PackageSource =
            serde_json::from_str(include_str!("../../tests/platform_process/source.json")).unwrap();
        src.resource_scope.resource_digest = plan.resource_digest();
        let sources = SourceRegistry::compile([src]).unwrap();
        let compiled = sources.get("local.process.cwd", "1.0.0").unwrap();
        let (mut executor, handle) = plan.bind(compiled, clock).unwrap();
        let (request, prepared, call) = execution(compiled);
        let result = executor.execute(
            &call,
            &request,
            compiled,
            &prepared,
            ToolExecutionContext {
                cancellation: &token,
                deadline,
            },
        );
        std::fs::remove_dir(&path).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "must reach actual final verification"
        );
        assert!(result.is_err(), "{fault} must refuse successful output");
        assert!(
            handle.shared.state.lock().unwrap().receipt.is_none(),
            "{fault} must not publish a receipt"
        );
    }
}
