//! Real OS effects behind the existing host's A1 approval and replay boundary.
use super::*;
use abi_agent_host::platform_process::{ProcessLimits, ProcessPlan};
use std::{path::PathBuf, sync::Arc};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "abi-governed-process-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
struct ProcessProvider(AtomicUsize);
impl ModelProvider for ProcessProvider {
    fn id(&self) -> &'static str {
        "process-provider"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            run.emit(&ModelEvent::ToolCall(ToolCall::new(
                "call",
                "process_cwd",
                "{}",
            )));
        }
        Ok(())
    }
}
struct Counted<E> {
    inner: E,
    entries: usize,
}
impl<E: GovernedLiveExecutor> GovernedLiveExecutor for Counted<E> {
    fn binding(&self) -> AdapterBinding {
        self.inner.binding()
    }
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        self.entries += 1;
        self.inner.execute(call, request, source, prepared, context)
    }
}
fn registry(plan: &ProcessPlan) -> SourceRegistry {
    let mut src = source();
    src.id = "local.process.cwd".into();
    src.resource_scope.resource_digest = plan.resource_digest();
    src.output_schema = json!({"type":"object","properties":{"cwd":{"type":"string"},"exit_code":{"const":0}},"required":["cwd","exit_code"],"additionalProperties":false});
    src.cancellation = CancellationContract::Cooperative;
    src.side_effect = SideEffect::PlatformRead;
    src.confirmation_policy = ApprovalLevel::A1Actor;
    src.approval_policy = ApprovalLevel::A1Actor;
    src.required_platform_permissions = BTreeSet::from([
        "process.pinned_exec_access".into(),
        "process.pinned_cwd_search".into(),
    ]);
    src.preconditions = BTreeSet::from(["process.ready".into()]);
    src.invariants = BTreeSet::from(["process.scoped".into()]);
    src.postconditions =
        BTreeSet::from(["process.exited_zero".into(), "process.cwd_matches".into()]);
    src.adapter_bindings = vec![AdapterBinding {
        adapter_id: "abi.pinned-process".into(),
        api_version: "1".into(),
    }];
    SourceRegistry::compile([src]).unwrap()
}
fn authority(compiled: &CompiledSource) -> (Request, Facts, Principal) {
    let actor = Principal::new("actor", PrincipalKind::HumanSubject).unwrap();
    let template = Request::new(
        "attempt",
        "call",
        actor.clone(),
        compiled.source().resource_scope.clone(),
        "local.process.cwd",
        "1.0.0",
        compiled.package().digest,
        Digest::default(),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let grant = Grant::active(
        "grant",
        Principal::new("owner", PrincipalKind::OrganizationOwner).unwrap(),
        actor.clone(),
        template.scope.clone(),
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
    let upstream = Facts {
        value: GovernedFacts {
            platform: PlatformFacts::new(BTreeSet::new(), 0, "p", "g", "s"),
            grants: vec![grant],
            used_decision_ids: BTreeSet::new(),
            predicates: BTreeSet::new(),
            captured_pre_state: None,
            observed_at_ms: 90,
            expires_at_ms: 2000,
            permitted_data_classes: BTreeSet::from([DataClass::Operational]),
            residency: ResidencyPolicy::LocalOnly,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        },
        reads: 0,
        change: false,
        post: Postconditions::Unverifiable,
    };
    (template, upstream, actor)
}
fn changed_host_inputs_refuse(
    compiled: &CompiledSource,
    template: &Request,
    approval: &ExactApproval,
    facts: &mut dyn GovernedFactProvider,
    ledger: &mut Ledger,
    executor: &mut Counted<abi_agent_host::platform_process::GovernedProcessExecutor>,
) {
    for mutation in ["principal", "resource", "source", "cancel"] {
        let mut src = compiled.source().clone();
        let mut request = template.clone();
        let token = CancellationToken::new();
        match mutation {
            "principal" => request.principal.id = "other".into(),
            "resource" => src.resource_scope.resource_digest = Digest::default(),
            "source" => src.deadline_ms += 1,
            "cancel" => token.cancel(),
            _ => unreachable!(),
        }
        let sources = SourceRegistry::compile([src]).unwrap();
        let selected = sources.get("local.process.cwd", "1.0.0").unwrap();
        request.scope = selected.source().resource_scope.clone();
        request.package_digest = selected.package().digest;
        request.request_digest = request.computed_digest();
        let provider = ProcessProvider(AtomicUsize::new(0));
        let host = GovernedHost::new(
            &provider,
            &sources,
            vec![GovernedRegistration {
                tool_name: "process_cwd".into(),
                request,
            }],
        )
        .unwrap();
        let report = host.run_live(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &token,
            GovernedOptions {
                clock: &FixedClock::new(100),
                host_budget: HostBudget::default(),
                approval: Some(approval.clone()),
                idempotency_key: None,
                requested_deadline_ms: 1000,
                requested_budgets: selected.source().budgets.clone(),
            },
            facts,
            ledger,
            executor,
        );
        assert!(
            !report.is_ok_and(|r| r.governance == GovernedStop::Completed),
            "{mutation}"
        );
        assert_eq!(executor.entries, 0, "{mutation}");
        assert_eq!(ledger.intents, 0, "{mutation}");
    }
}
fn actor_approval(pending: GovernedRunReport, actor: Principal) -> ExactApproval {
    let GovernedStop::PendingApproval(binding) = pending.governance else {
        panic!("{:?}", pending.governance)
    };
    ExactApproval {
        issued_at_ms: 90,
        approval: Approval {
            decision_id: "decision".into(),
            call_id: pending.pending_approval.unwrap().call_reference,
            call_digest: binding.request_digest,
            package_digest: binding.package_digest,
            grant_id: "grant".into(),
            approver: actor,
            coapprover: None,
            level: ApprovalLevel::A1Actor,
            expires_at_ms: 200,
            state: ApprovalState::Approved,
        },
        binding,
    }
}
#[test]
fn actual_process_requires_exact_actor_approval_and_never_replays() {
    let scratch = Scratch::new();
    let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
    let sources = registry(&plan);
    let compiled = sources.get("local.process.cwd", "1.0.0").unwrap();
    let (template, upstream, actor) = authority(compiled);
    let clock = Arc::new(FixedClock::new(100));
    let (executor, handle) = plan.bind(compiled, clock.clone()).unwrap();
    let mut executor = Counted {
        inner: executor,
        entries: 0,
    };
    let mut facts = handle.decorate(upstream);
    let mut ledger = Ledger::default();
    let submit = |approval,
                  facts: &mut dyn GovernedFactProvider,
                  ledger: &mut Ledger,
                  executor: &mut dyn GovernedLiveExecutor| {
        let provider = ProcessProvider(AtomicUsize::new(0));
        let host = GovernedHost::new(
            &provider,
            &sources,
            vec![GovernedRegistration {
                tool_name: "process_cwd".into(),
                request: template.clone(),
            }],
        )
        .unwrap();
        host.run_live(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &CancellationToken::new(),
            GovernedOptions {
                clock: clock.as_ref(),
                host_budget: HostBudget::default(),
                approval,
                idempotency_key: None,
                requested_deadline_ms: 1000,
                requested_budgets: compiled.source().budgets.clone(),
            },
            facts,
            ledger,
            executor,
        )
        .unwrap()
    };
    let pending = submit(None, &mut facts, &mut ledger, &mut executor);
    assert_eq!(executor.entries, 0);
    assert_eq!(ledger.intents, 0);
    let approval = actor_approval(pending, actor);
    for mutation in ["denied", "expired", "changed"] {
        let mut changed = approval.clone();
        match mutation {
            "denied" => changed.approval.state = ApprovalState::Denied,
            "expired" => changed.approval.expires_at_ms = 100,
            "changed" => changed.binding.scope_digest = Digest::default(),
            _ => unreachable!(),
        }
        let refused = submit(Some(changed), &mut facts, &mut ledger, &mut executor);
        assert_ne!(refused.governance, GovernedStop::Completed);
        assert_eq!(executor.entries, 0);
        assert_eq!(ledger.intents, 0);
    }
    changed_host_inputs_refuse(
        compiled,
        &template,
        &approval,
        &mut facts,
        &mut ledger,
        &mut executor,
    );
    let report = submit(
        Some(approval.clone()),
        &mut facts,
        &mut ledger,
        &mut executor,
    );
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
    assert_eq!(executor.entries, 1);
    assert_eq!(ledger.intents, 1);
    let replay = submit(Some(approval), &mut facts, &mut ledger, &mut executor);
    assert_ne!(replay.governance, GovernedStop::Completed);
    assert_eq!(executor.entries, 1);
    assert_eq!(ledger.intents, 1);
}
