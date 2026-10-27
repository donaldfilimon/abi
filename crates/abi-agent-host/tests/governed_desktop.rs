//! Governed desktop application activation: exact bundle binding, refusal
//! before activation, single-shot activation and frontmost verification
//! against a fake backend. The macOS native test is `#[ignore]` (run by hand).
use abi_agent_host::{
    GovernedFactProvider, GovernedFacts, GovernedLiveExecutor, ObservationSource,
    PostconditionObservation, Postconditions, ToolExecutionContext,
    governed_desktop::{DesktopBackend, DesktopError, GovernedDesktopExecutor},
};
use abi_agent_runtime::{CancellationToken, ToolCall};
use abi_capability::{
    ApprovalLevel, Clock, Digest, PlatformFacts, Principal, PrincipalKind, Request,
    lifecycle::{AuthorizationBinding, LifecycleError, PreparedExecution},
    source::{
        CompiledSource, DataClass, PackageSource, ResidencyPolicy, RetentionClass, SourceRegistry,
    },
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const FINDER: &str = "com.apple.finder";

fn hash(value: &impl serde::Serialize) -> Digest {
    Digest::from_bytes(Sha256::digest(serde_json::to_vec(value).unwrap()).into())
}
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Shared record of what the fake backend was asked to do.
#[derive(Default)]
struct Log {
    probes: u32,
    activations: Vec<String>,
}
/// Fake backend: `probe` answers the pre-activation read, `after` every later read.
struct FakeBackend {
    log: Arc<Mutex<Log>>,
    probe: Result<&'static str, DesktopError>,
    after: &'static str,
    activated: bool,
}
impl DesktopBackend for FakeBackend {
    fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError> {
        self.log.lock().unwrap().probes += 1;
        if self.activated {
            Ok(self.after.into())
        } else {
            self.probe.map(Into::into)
        }
    }
    fn activate(&mut self, bundle_id: &str) -> Result<(), DesktopError> {
        self.log.lock().unwrap().activations.push(bundle_id.into());
        self.activated = true;
        Ok(())
    }
}

fn input(bundle_id: &str) -> serde_json::Value {
    serde_json::json!({ "bundle_id": bundle_id })
}
fn compile(bundle_id: &str) -> SourceRegistry {
    let mut src: PackageSource =
        serde_json::from_str(include_str!("governed_desktop/source.json")).unwrap();
    src.resource_scope.resource_digest =
        abi_agent_host::governed_desktop::resource_digest(bundle_id);
    SourceRegistry::compile([src]).unwrap()
}
/// A request/preparation pair bound to `parameters` (what was approved).
fn execution(
    compiled: &CompiledSource,
    parameters: &serde_json::Value,
) -> (Request, PreparedExecution, ToolCall) {
    let principal = Principal::new("actor", PrincipalKind::Service).unwrap();
    let call = ToolCall::new("call", "desktop_activate", parameters.to_string());
    let src = compiled.source();
    let id = hash(&(
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
        hash(parameters),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let binding = AuthorizationBinding {
        request_digest: request.request_digest,
        principal_digest: hash(&request.principal),
        scope_digest: hash(&request.scope),
        package_digest: request.package_digest,
        grant_digest: Digest::from_bytes([6; 32]),
        policy_digest: hash(&("abi.policy.v1", "p", "g", "s", 0_u64)),
        schema_digest: compiled.schema_digest(),
        precondition_digest: Digest::from_bytes([7; 32]),
        compensation_origin: None,
        required_approval: ApprovalLevel::A1Actor,
    };
    (request, PreparedExecution::after_commit(id, binding), call)
}
fn executor(
    compiled: &CompiledSource,
    probe: Result<&'static str, DesktopError>,
    after: &'static str,
) -> (GovernedDesktopExecutor<FakeBackend>, Arc<Mutex<Log>>) {
    let log = Arc::new(Mutex::new(Log::default()));
    let exec = GovernedDesktopExecutor::bind(
        compiled,
        FINDER.into(),
        FakeBackend {
            log: Arc::clone(&log),
            probe,
            after,
            activated: false,
        },
        Arc::new(TestClock(AtomicU64::new(100))),
    )
    .unwrap();
    (exec, log)
}
fn run<B: DesktopBackend>(
    exec: &mut GovernedDesktopExecutor<B>,
    compiled: &CompiledSource,
    parts: &(Request, PreparedExecution, ToolCall),
) -> Result<abi_agent_host::ToolOutput, abi_agent_host::ToolExecutionError> {
    let (request, prepared, call) = parts;
    exec.execute(
        call,
        request,
        compiled,
        prepared,
        ToolExecutionContext {
            cancellation: &CancellationToken::new(),
            deadline: Instant::now() + Duration::from_secs(10),
        },
    )
}

#[test]
fn approved_application_activates_and_verifies_frontmost() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, log) = executor(compiled, Ok("com.example.editor"), FINDER);
    let parts = execution(compiled, &input(FINDER));
    let output = run(&mut exec, compiled, &parts).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(output.payload()).unwrap(),
        serde_json::json!({"bundle_id": FINDER, "frontmost": true})
    );
    assert_eq!(log.lock().unwrap().activations, [FINDER]);
    assert_eq!(exec.activations(), 1);
    // The executor is single-shot: a replay refuses without a second activation.
    let again = run(&mut exec, compiled, &parts).unwrap_err();
    assert_eq!(again.message(), "desktop_already_attempted");
    assert_eq!(log.lock().unwrap().activations.len(), 1);
}

#[test]
fn different_bundle_in_call_refuses_before_activation() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, log) = executor(compiled, Ok("com.example.editor"), FINDER);
    let other = input("com.apple.Terminal");
    // Executed input differs from the approval, whether or not the request
    // parameters were re-derived from it.
    for parameters in [input(FINDER), other.clone()] {
        let (request, prepared, mut call) = execution(compiled, &parameters);
        call.input = other.to_string();
        let refused = run(&mut exec, compiled, &(request, prepared, call)).unwrap_err();
        assert_eq!(refused.message(), "desktop_binding_refused");
    }
    assert_eq!(log.lock().unwrap().activations.len(), 0);
    assert_eq!(exec.activations(), 0);
    // A different or malformed approval cannot bind to the approved source.
    for bundle in [
        "com.apple.Terminal",
        "finder",
        "-b.apple.finder",
        "com..finder",
        "com.apple.finder;rm",
        "ab",
    ] {
        let log = Arc::new(Mutex::new(Log::default()));
        let bound = GovernedDesktopExecutor::bind(
            compiled,
            bundle.into(),
            FakeBackend {
                log,
                probe: Ok("x.y"),
                after: FINDER,
                activated: false,
            },
            Arc::new(TestClock(AtomicU64::new(100))),
        );
        assert!(bound.is_err(), "{bundle}");
    }
    // The refusals did not consume the executor: the exact approval activates once.
    let parts = execution(compiled, &input(FINDER));
    assert!(run(&mut exec, compiled, &parts).is_ok());
    assert_eq!(log.lock().unwrap().activations, [FINDER]);
}

#[test]
fn backend_unsupported_or_permission_denied_refuses() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let parts = execution(compiled, &input(FINDER));
    for (failure, code) in [
        (DesktopError::Unsupported, "desktop_unsupported"),
        (DesktopError::PermissionDenied, "desktop_permission_denied"),
    ] {
        let (mut exec, log) = executor(compiled, Err(failure), FINDER);
        let refused = run(&mut exec, compiled, &parts).unwrap_err();
        assert_eq!(refused.message(), code);
        assert_eq!(log.lock().unwrap().activations.len(), 0, "{code}");
        assert_eq!(exec.activations(), 0, "{code}");
    }
}

#[test]
fn frontmost_mismatch_after_activation_is_uncertain_not_retried() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, log) = executor(compiled, Ok("com.example.editor"), "com.example.editor");
    let parts = execution(compiled, &input(FINDER));
    let failed = run(&mut exec, compiled, &parts).unwrap_err();
    assert_eq!(failed.message(), "desktop_frontmost_unverified");
    {
        let log = log.lock().unwrap();
        assert_eq!(log.activations, [FINDER]);
        // One pre-activation probe plus at most 20 verification polls.
        assert!((2..=21).contains(&log.probes), "{}", log.probes);
    }
    let again = run(&mut exec, compiled, &parts).unwrap_err();
    assert_eq!(again.message(), "desktop_already_attempted");
    assert_eq!(log.lock().unwrap().activations.len(), 1);
    assert_eq!(exec.activations(), 1);
}

/// Permissive upstream: claims every postcondition verified with fixed evidence.
struct Permissive;
const UPSTREAM_EVIDENCE: Digest = Digest::from_bytes([7; 32]);
impl GovernedFactProvider for Permissive {
    fn observe(&mut self, _: &Request) -> Result<GovernedFacts, LifecycleError> {
        Ok(GovernedFacts {
            platform: PlatformFacts::new(BTreeSet::from(["desktop.activate"]), 0, "p", "g", "s"),
            grants: vec![],
            used_decision_ids: BTreeSet::new(),
            predicates: BTreeSet::from(["desktop.approved".into(), "desktop.exact".into()]),
            captured_pre_state: None,
            observed_at_ms: 90,
            expires_at_ms: 500,
            permitted_data_classes: BTreeSet::from([DataClass::Operational]),
            residency: ResidencyPolicy::LocalOnly,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        })
    }
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        Ok(required
            .iter()
            .map(|predicate| PostconditionObservation {
                predicate: predicate.clone(),
                request_digest: request.request_digest,
                observed_at_ms: 90,
                expires_at_ms: 500,
                evidence_digest: Some(UPSTREAM_EVIDENCE),
                source: ObservationSource::TrustedLocalAdapter,
                status: Postconditions::Verified,
            })
            .collect())
    }
}
fn frontmost_required() -> BTreeSet<String> {
    BTreeSet::from(["desktop.frontmost_matches".into()])
}

#[test]
fn frontmost_postcondition_evidence_comes_only_from_the_executor_receipt() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, _log) = executor(compiled, Ok("com.example.editor"), FINDER);
    let mut facts = exec.observation_handle().decorate(Permissive);
    let parts = execution(compiled, &input(FINDER));
    let request = &parts.0;
    // Upstream claims are never inherited: no receipt yet, no evidence.
    assert!(
        facts
            .postconditions(request, &frontmost_required())
            .is_err()
    );
    assert!(facts.observe(request).is_ok());
    run(&mut exec, compiled, &parts).unwrap();
    // Only the exact owned predicate set is answered.
    let wider = BTreeSet::from(["desktop.frontmost_matches".into(), "other".into()]);
    assert!(facts.postconditions(request, &wider).is_err());
    // The wrong-set attempt consumed the receipt: evidence is single-use.
    assert!(
        facts
            .postconditions(request, &frontmost_required())
            .is_err()
    );
}

#[test]
fn verified_activation_yields_one_single_use_adapter_observation() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, _log) = executor(compiled, Ok("com.example.editor"), FINDER);
    let mut facts = exec.observation_handle().decorate(Permissive);
    let parts = execution(compiled, &input(FINDER));
    let request = &parts.0;
    run(&mut exec, compiled, &parts).unwrap();
    let observed = facts
        .postconditions(request, &frontmost_required())
        .unwrap();
    assert_eq!(observed.len(), 1);
    let only = &observed[0];
    assert_eq!(only.predicate, "desktop.frontmost_matches");
    assert_eq!(only.request_digest, request.request_digest);
    assert_eq!(only.status, Postconditions::Verified);
    assert_eq!(only.source, ObservationSource::TrustedLocalAdapter);
    assert!(
        only.evidence_digest
            .is_some_and(|d| d != UPSTREAM_EVIDENCE && d != Digest::default())
    );
    assert!((only.observed_at_ms..only.expires_at_ms).contains(&100));
    assert!(
        facts
            .postconditions(request, &frontmost_required())
            .is_err()
    );
    // A request for a different approval is refused outright.
    let other = execution(compiled, &input("com.apple.Terminal"));
    assert!(facts.observe(&other.0).is_err());
}

#[test]
fn lost_focus_has_no_evidence_even_when_upstream_claims_verified() {
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let (mut exec, _log) = executor(compiled, Ok("com.example.editor"), "com.example.editor");
    let mut facts = exec.observation_handle().decorate(Permissive);
    let parts = execution(compiled, &input(FINDER));
    let failed = run(&mut exec, compiled, &parts).unwrap_err();
    assert_eq!(failed.message(), "desktop_frontmost_unverified");
    assert!(
        facts
            .postconditions(&parts.0, &frontmost_required())
            .is_err()
    );
}

/// Activates Finder on this Mac through `MacDesktop`. Changes the frontmost
/// application; run by hand: `-- --ignored native_mac`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "activates com.apple.finder on the local desktop"]
fn native_mac_activation_of_finder() {
    use abi_agent_host::governed_desktop::MacDesktop;
    let registry = compile(FINDER);
    let compiled = registry.get("abi.desktop.activate", "1.0.0").unwrap();
    let mut exec = GovernedDesktopExecutor::bind(
        compiled,
        FINDER.into(),
        MacDesktop::new(),
        Arc::new(TestClock(AtomicU64::new(100))),
    )
    .unwrap();
    let parts = execution(compiled, &input(FINDER));
    let output = run(&mut exec, compiled, &parts).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(output.payload()).unwrap(),
        serde_json::json!({"bundle_id": FINDER, "frontmost": true})
    );
    assert_eq!(exec.activations(), 1);
}
