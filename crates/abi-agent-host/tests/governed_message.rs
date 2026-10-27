//! Governed Discord/Twilio message executor: exact approval binding and
//! single-shot sending against a counting fake sender (no network).
use abi_agent_host::{
    GovernedFactProvider, GovernedFacts, GovernedLiveExecutor, ObservationSource,
    PostconditionObservation, Postconditions, ToolExecutionContext,
    governed_message::{
        ApprovedMessage, GovernedMessageExecutor, MessageChannel, MessageReceipt, MessageSender,
        SendError,
    },
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
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

fn hash(value: &impl serde::Serialize) -> Digest {
    Digest::from_bytes(Sha256::digest(serde_json::to_vec(value).unwrap()).into())
}
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Counting fake sender: records every call and replays one fixed outcome.
struct FakeSender {
    sends: Arc<AtomicU64>,
    outcome: Result<&'static str, SendError>,
}
impl MessageSender for FakeSender {
    fn send(&mut self, _message: &ApprovedMessage) -> Result<MessageReceipt, SendError> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        self.outcome.map(|id| MessageReceipt {
            provider_id: id.into(),
        })
    }
}

fn approved() -> ApprovedMessage {
    ApprovedMessage {
        channel: MessageChannel::DiscordChannel,
        account: "abbey-bot".into(),
        recipient: "123456789012345678".into(),
        body: "deploy finished".into(),
    }
}
fn source(message: &ApprovedMessage) -> PackageSource {
    let mut src: PackageSource =
        serde_json::from_str(include_str!("governed_message/source.json")).unwrap();
    src.resource_scope.resource_digest = message.resource_digest();
    src
}
fn compile(message: &ApprovedMessage) -> SourceRegistry {
    SourceRegistry::compile([source(message)]).unwrap()
}
/// A request/preparation pair bound to `input` (the parameters actually approved).
fn execution(
    compiled: &CompiledSource,
    input: &serde_json::Value,
) -> (Request, PreparedExecution, ToolCall) {
    let principal = Principal::new("actor", PrincipalKind::Service).unwrap();
    let call = ToolCall::new("call", "message_send", input.to_string());
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
        hash(input),
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
    outcome: Result<&'static str, SendError>,
    now: u64,
) -> (GovernedMessageExecutor<FakeSender>, Arc<AtomicU64>) {
    let sends = Arc::new(AtomicU64::new(0));
    let exec = GovernedMessageExecutor::bind(
        compiled,
        approved(),
        FakeSender {
            sends: Arc::clone(&sends),
            outcome,
        },
        Arc::new(TestClock(AtomicU64::new(now))),
    )
    .unwrap();
    (exec, sends)
}
fn run(
    exec: &mut GovernedMessageExecutor<FakeSender>,
    compiled: &CompiledSource,
    parts: &(Request, PreparedExecution, ToolCall),
    cancel: &CancellationToken,
) -> Result<abi_agent_host::ToolOutput, abi_agent_host::ToolExecutionError> {
    let (request, prepared, call) = parts;
    exec.execute(
        call,
        request,
        compiled,
        prepared,
        ToolExecutionContext {
            cancellation: cancel,
            deadline: Instant::now() + Duration::from_secs(5),
        },
    )
}

#[test]
fn approved_message_sends_exactly_once_and_returns_receipt() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    let (mut exec, sends) = executor(compiled, Ok("discord-msg-1"), 100);
    assert_eq!(exec.sends(), 0);
    let parts = execution(compiled, &message.input_value());
    let output = run(&mut exec, compiled, &parts, &CancellationToken::new()).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(output.payload()).unwrap(),
        serde_json::json!({"provider_id":"discord-msg-1"})
    );
    assert_eq!((exec.sends(), sends.load(Ordering::SeqCst)), (1, 1));
    // A replay of the identical, already-sent approval refuses without sending.
    let again = run(&mut exec, compiled, &parts, &CancellationToken::new()).unwrap_err();
    assert_eq!(again.message(), "message_already_attempted");
    assert_eq!((exec.sends(), sends.load(Ordering::SeqCst)), (1, 1));
}

#[test]
fn recipient_or_content_mismatch_refuses_before_send() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    for field in ["recipient", "body", "account", "channel"] {
        let mut changed = message.clone();
        match field {
            "recipient" => changed.recipient = "999999999999999999".into(),
            "body" => changed.body = "deploy finished!".into(),
            "account" => changed.account = "other-bot".into(),
            "channel" => changed.channel = MessageChannel::TwilioSms,
            _ => unreachable!(),
        }
        let (mut exec, sends) = executor(compiled, Ok("never"), 100);
        // Executed input differs from the approval, whether or not the request
        // parameters were re-derived from the changed input.
        for parameters in [message.input_value(), changed.input_value()] {
            let (request, prepared, mut call) = execution(compiled, &parameters);
            call.input = changed.input_value().to_string();
            let refused = run(
                &mut exec,
                compiled,
                &(request, prepared, call),
                &CancellationToken::new(),
            );
            assert!(refused.is_err(), "{field}");
        }
        assert_eq!(sends.load(Ordering::SeqCst), 0, "{field}");
        assert_eq!(exec.sends(), 0, "{field}");
        // A changed approval cannot be bound against the approved source either.
        assert!(
            GovernedMessageExecutor::bind(
                compiled,
                changed,
                FakeSender {
                    sends: Arc::clone(&sends),
                    outcome: Ok("never"),
                },
                Arc::new(TestClock(AtomicU64::new(100))),
            )
            .is_err(),
            "{field}"
        );
        // The refusals did not consume the executor: the exact approval still sends once.
        let parts = execution(compiled, &message.input_value());
        assert!(run(&mut exec, compiled, &parts, &CancellationToken::new()).is_ok());
        assert_eq!(sends.load(Ordering::SeqCst), 1, "{field}");
    }
}

#[test]
fn ambiguous_send_is_not_retried() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    let (mut exec, sends) = executor(
        compiled,
        Err(SendError::Uncertain("message_send_uncertain")),
        100,
    );
    let parts = execution(compiled, &message.input_value());
    let failed = run(&mut exec, compiled, &parts, &CancellationToken::new()).unwrap_err();
    assert_eq!(failed.message(), "message_send_uncertain");
    assert_eq!((exec.sends(), sends.load(Ordering::SeqCst)), (1, 1));
    let again = run(&mut exec, compiled, &parts, &CancellationToken::new()).unwrap_err();
    assert_eq!(again.message(), "message_already_attempted");
    assert_eq!((exec.sends(), sends.load(Ordering::SeqCst)), (1, 1));
}

#[test]
fn expired_source_or_stop_refuses_before_send() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    let parts = execution(compiled, &message.input_value());
    for case in ["expired", "cancelled", "deadline"] {
        // The fixture source expires at 10,000 ms.
        let now = if case == "expired" { 10_000 } else { 100 };
        let (mut exec, sends) = executor(compiled, Ok("never"), now);
        let cancel = CancellationToken::new();
        if case == "cancelled" {
            cancel.cancel();
        }
        let deadline = if case == "deadline" {
            Instant::now()
        } else {
            Instant::now() + Duration::from_secs(5)
        };
        let (request, prepared, call) = &parts;
        let refused = exec.execute(
            call,
            request,
            compiled,
            prepared,
            ToolExecutionContext {
                cancellation: &cancel,
                deadline,
            },
        );
        assert!(refused.is_err(), "{case}");
        assert_eq!(sends.load(Ordering::SeqCst), 0, "{case}");
        assert_eq!(exec.sends(), 0, "{case}");
    }
}

/// Permissive upstream: claims every postcondition verified with fixed evidence.
struct Permissive;
const UPSTREAM_EVIDENCE: Digest = Digest::from_bytes([7; 32]);
impl GovernedFactProvider for Permissive {
    fn observe(&mut self, _: &Request) -> Result<GovernedFacts, LifecycleError> {
        Ok(GovernedFacts {
            platform: PlatformFacts::new(BTreeSet::from(["message.send"]), 0, "p", "g", "s"),
            grants: vec![],
            used_decision_ids: BTreeSet::new(),
            predicates: BTreeSet::from(["message.approved".into(), "message.exact".into()]),
            captured_pre_state: None,
            observed_at_ms: 90,
            expires_at_ms: 500,
            permitted_data_classes: BTreeSet::from([DataClass::Personal]),
            residency: ResidencyPolicy::TenantRegion,
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
fn accepted_required() -> BTreeSet<String> {
    BTreeSet::from(["message.provider_accepted".into()])
}

#[test]
fn provider_receipt_is_the_only_postcondition_evidence() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    let (mut exec, _sends) = executor(compiled, Ok("1100"), 100);
    let mut facts = exec.observation_handle().decorate(Permissive);
    let parts = execution(compiled, &message.input_value());
    let request = &parts.0;
    // Upstream claims are never inherited: no receipt yet, no evidence.
    assert!(facts.postconditions(request, &accepted_required()).is_err());
    assert!(facts.observe(request).is_ok());
    run(&mut exec, compiled, &parts, &CancellationToken::new()).unwrap();
    let observed = facts.postconditions(request, &accepted_required()).unwrap();
    assert_eq!(observed.len(), 1);
    let only = &observed[0];
    assert_eq!(only.predicate, "message.provider_accepted");
    assert_eq!(only.request_digest, request.request_digest);
    assert_eq!(only.status, Postconditions::Verified);
    assert_eq!(only.source, ObservationSource::TrustedLocalAdapter);
    assert!(
        only.evidence_digest
            .is_some_and(|d| d != UPSTREAM_EVIDENCE && d != Digest::default())
    );
    assert!((only.observed_at_ms..only.expires_at_ms).contains(&100));
    // Single use.
    assert!(facts.postconditions(request, &accepted_required()).is_err());
    // A request for a different approval is refused outright.
    let mut other = message.clone();
    other.body = "something else".into();
    assert!(
        facts
            .observe(&execution(compiled, &other.input_value()).0)
            .is_err()
    );
}

#[test]
fn failed_send_has_no_evidence_even_when_upstream_claims_verified() {
    let message = approved();
    let registry = compile(&message);
    let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
    for outcome in [
        Err(SendError::Uncertain("message_send_uncertain")),
        Err(SendError::Refused("message_send_refused")),
        Ok(""),
    ] {
        let (mut exec, _sends) = executor(compiled, outcome, 100);
        let mut facts = exec.observation_handle().decorate(Permissive);
        let parts = execution(compiled, &message.input_value());
        assert!(run(&mut exec, compiled, &parts, &CancellationToken::new()).is_err());
        assert!(
            facts
                .postconditions(&parts.0, &accepted_required())
                .is_err(),
            "{outcome:?}"
        );
    }
}

#[test]
fn source_must_declare_exactly_the_provider_accepted_postcondition() {
    let message = approved();
    for postconditions in [
        BTreeSet::from(["message.delivered".to_string()]),
        BTreeSet::from([
            "message.provider_accepted".to_string(),
            "message.delivered".to_string(),
        ]),
    ] {
        let mut src = source(&message);
        src.postconditions = postconditions.clone();
        let registry = SourceRegistry::compile([src]).unwrap();
        let compiled = registry.get("abi.message.send", "1.0.0").unwrap();
        let bound = GovernedMessageExecutor::bind(
            compiled,
            message.clone(),
            FakeSender {
                sends: Arc::new(AtomicU64::new(0)),
                outcome: Ok("never"),
            },
            Arc::new(TestClock(AtomicU64::new(100))),
        );
        assert!(bound.is_err(), "{postconditions:?}");
    }
}
