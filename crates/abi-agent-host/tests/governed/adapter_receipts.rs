//! Desktop and message executors under the governed lifecycle: the terminal
//! outcome rests on the executor's own receipt, never on upstream claims.
use super::*;
use abi_agent_host::{
    governed_desktop::{DesktopBackend, DesktopError, GovernedDesktopExecutor},
    governed_message::{
        ApprovedMessage, GovernedMessageExecutor, MessageChannel, MessageReceipt, MessageSender,
        SendError,
    },
};
use std::sync::Arc;

const FINDER: &str = "com.apple.finder";

struct OneCall(AtomicUsize, &'static str, String);
impl ModelProvider for OneCall {
    fn id(&self) -> &'static str {
        "adapter-receipt-provider"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            run.emit(&ModelEvent::ToolCall(ToolCall::new(
                "call", self.1, &self.2,
            )));
        }
        Ok(())
    }
}
/// Frontmost becomes `after` once activation was requested.
struct Desktop {
    after: &'static str,
    activated: bool,
}
impl DesktopBackend for Desktop {
    fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError> {
        Ok(if self.activated {
            self.after
        } else {
            "com.example.editor"
        }
        .into())
    }
    fn activate(&mut self, _: &str) -> Result<(), DesktopError> {
        self.activated = true;
        Ok(())
    }
}
struct Sender(Result<&'static str, SendError>);
impl MessageSender for Sender {
    fn send(&mut self, _: &ApprovedMessage) -> Result<MessageReceipt, SendError> {
        self.0.map(|id| MessageReceipt {
            provider_id: id.into(),
        })
    }
}
fn message() -> ApprovedMessage {
    ApprovedMessage {
        channel: MessageChannel::DiscordChannel,
        account: "abbey-bot".into(),
        recipient: "123456789012345678".into(),
        body: "deploy finished".into(),
    }
}
fn desktop_sources() -> SourceRegistry {
    let mut src: PackageSource =
        serde_json::from_str(include_str!("../governed_desktop/source.json")).unwrap();
    src.resource_scope.resource_digest = abi_agent_host::governed_desktop::resource_digest(FINDER);
    SourceRegistry::compile([src]).unwrap()
}
fn message_sources() -> SourceRegistry {
    let mut src: PackageSource =
        serde_json::from_str(include_str!("../governed_message/source.json")).unwrap();
    src.resource_scope.resource_digest = message().resource_digest();
    // An irreversible capability is granted only by a guild owner.
    src.resource_scope.guild_ref = Some("guild".into());
    SourceRegistry::compile([src]).unwrap()
}
/// Upstream authority for `compiled`; `post` is what it would claim afterwards.
fn upstream(compiled: &CompiledSource, actor: &Principal, post: Postconditions) -> Facts {
    let src = compiled.source();
    let issuer = if src.resource_scope.guild_ref.is_some() {
        PrincipalKind::GuildOwner
    } else {
        PrincipalKind::OrganizationOwner
    };
    let grant = Grant::active(
        "grant",
        Principal::new("owner", issuer).unwrap(),
        actor.clone(),
        src.resource_scope.clone(),
        compiled.package(),
        0,
        2000,
        ApprovalLevel::A1Actor,
        0,
        "p",
        "g",
        "s",
    )
    .unwrap();
    Facts {
        value: GovernedFacts {
            platform: PlatformFacts::new(
                src.required_platform_permissions
                    .iter()
                    .map(String::as_str)
                    .collect(),
                0,
                "p",
                "g",
                "s",
            ),
            grants: vec![grant],
            used_decision_ids: BTreeSet::new(),
            predicates: src.preconditions.union(&src.invariants).cloned().collect(),
            captured_pre_state: None,
            observed_at_ms: 90,
            expires_at_ms: 2000,
            permitted_data_classes: src.data_classes.clone(),
            residency: src.residency,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        },
        reads: 0,
        change: false,
        post,
    }
}
/// Submit once for approval, approve exactly, then run the approved call.
fn approved_run(
    sources: &SourceRegistry,
    compiled: &CompiledSource,
    tool: &'static str,
    input: &serde_json::Value,
    actor: &Principal,
    facts: &mut dyn GovernedFactProvider,
    executor: &mut dyn GovernedLiveExecutor,
) -> GovernedRunReport {
    let src = compiled.source();
    let template = Request::new(
        "attempt",
        "call",
        actor.clone(),
        src.resource_scope.clone(),
        &src.id,
        &src.version,
        compiled.package().digest,
        Digest::default(),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let clock = FixedClock::new(100);
    let mut ledger = Ledger::default();
    let mut submit = |approval: Option<ExactApproval>,
                      facts: &mut dyn GovernedFactProvider,
                      executor: &mut dyn GovernedLiveExecutor| {
        let provider = OneCall(AtomicUsize::new(0), tool, input.to_string());
        let host = GovernedHost::new(
            &provider,
            sources,
            vec![GovernedRegistration {
                tool_name: tool.into(),
                request: template.clone(),
            }],
        )
        .unwrap();
        host.run_live(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &CancellationToken::new(),
            GovernedOptions {
                clock: &clock,
                host_budget: HostBudget::default(),
                approval,
                idempotency_key: Some(Digest::from_bytes([8; 32])),
                requested_deadline_ms: 1000,
                requested_budgets: src.budgets.clone(),
            },
            facts,
            &mut ledger,
            executor,
        )
        .unwrap()
    };
    let pending = submit(None, facts, executor);
    let GovernedStop::PendingApproval(binding) = pending.governance else {
        panic!("{:?}", pending.governance)
    };
    // An irreversible send needs the guild owner; activation needs the actor.
    let level = binding.required_approval;
    let approver = if level == ApprovalLevel::A1Actor {
        actor.clone()
    } else {
        Principal::new("owner", PrincipalKind::GuildOwner).unwrap()
    };
    let approval = ExactApproval {
        issued_at_ms: 90,
        approval: Approval {
            decision_id: "decision".into(),
            call_id: pending.pending_approval.unwrap().call_reference,
            call_digest: binding.request_digest,
            package_digest: binding.package_digest,
            grant_id: "grant".into(),
            approver,
            coapprover: None,
            level,
            expires_at_ms: 200,
            state: ApprovalState::Approved,
        },
        binding,
    };
    submit(Some(approval), facts, executor)
}
fn actor() -> Principal {
    Principal::new("actor", PrincipalKind::HumanSubject).unwrap()
}

#[test]
fn verified_activation_succeeds_from_the_executor_receipt() {
    let sources = desktop_sources();
    let compiled = sources.get("abi.desktop.activate", "1.0.0").unwrap();
    let actor = actor();
    let mut executor = GovernedDesktopExecutor::bind(
        compiled,
        FINDER.into(),
        Desktop {
            after: FINDER,
            activated: false,
        },
        Arc::new(FixedClock::new(100)),
    )
    .unwrap();
    // The upstream provider cannot verify the frontmost application itself.
    let mut facts = executor.observation_handle().decorate(upstream(
        compiled,
        &actor,
        Postconditions::Unverifiable,
    ));
    let report = approved_run(
        &sources,
        compiled,
        "desktop_activate",
        &json!({ "bundle_id": FINDER }),
        &actor,
        &mut facts,
        &mut executor,
    );
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
    assert_eq!(executor.activations(), 1);
}

#[test]
fn accepted_message_succeeds_from_the_provider_receipt() {
    let sources = message_sources();
    let compiled = sources.get("abi.message.send", "1.0.0").unwrap();
    let actor = actor();
    let mut executor = GovernedMessageExecutor::bind(
        compiled,
        message(),
        Sender(Ok("1100")),
        Arc::new(FixedClock::new(100)),
    )
    .unwrap();
    let mut facts = executor.observation_handle().decorate(upstream(
        compiled,
        &actor,
        Postconditions::Unverifiable,
    ));
    let report = approved_run(
        &sources,
        compiled,
        "message_send",
        &message().input_value(),
        &actor,
        &mut facts,
        &mut executor,
    );
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
    assert_eq!(executor.sends(), 1);
}

#[test]
fn permissive_upstream_cannot_make_a_lost_focus_succeed() {
    let sources = desktop_sources();
    let compiled = sources.get("abi.desktop.activate", "1.0.0").unwrap();
    let actor = actor();
    let mut executor = GovernedDesktopExecutor::bind(
        compiled,
        FINDER.into(),
        Desktop {
            after: "com.example.editor",
            activated: false,
        },
        Arc::new(FixedClock::new(100)),
    )
    .unwrap();
    // Upstream claims verified evidence for every predicate.
    let mut facts = executor.observation_handle().decorate(upstream(
        compiled,
        &actor,
        Postconditions::Verified,
    ));
    let report = approved_run(
        &sources,
        compiled,
        "desktop_activate",
        &json!({ "bundle_id": FINDER }),
        &actor,
        &mut facts,
        &mut executor,
    );
    assert!(
        report
            .outcomes
            .iter()
            .all(|outcome| outcome.state != OutcomeState::Succeeded),
        "{report:?}"
    );
    assert_eq!(executor.activations(), 1);
}
