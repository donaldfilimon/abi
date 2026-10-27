//! Governed Discord/Twilio message executor: exact approval binding and
//! single-shot sending against a counting fake sender (no network).
use abi_agent_host::{
    GovernedLiveExecutor, ToolExecutionContext,
    governed_message::{
        ApprovedMessage, GovernedMessageExecutor, MessageChannel, MessageReceipt, MessageSender,
        SendError,
    },
};
use abi_agent_runtime::{CancellationToken, ToolCall};
use abi_capability::{
    ApprovalLevel, Clock, Digest, Principal, PrincipalKind, Request,
    lifecycle::{AuthorizationBinding, PreparedExecution},
    source::{CompiledSource, PackageSource, SourceRegistry},
};
use sha2::{Digest as _, Sha256};
use std::{
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
