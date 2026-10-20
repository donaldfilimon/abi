//! Fail-closed recording governance regressions.
use abi_agent_host::*;
use abi_agent_runtime::*;
use abi_capability::{lifecycle::*, source::*, *};
use serde_json::json;
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};
fn source() -> PackageSource {
    serde_json::from_value(json!({
        "id":"local.observe", "version":"1.0.0",
        "input_schema":{"type":"object","additionalProperties":false},
        "output_schema":{"type":"boolean"},"error_schema":{"type":"null"},
        "streaming":"None","cancellation":"BeforeEffect", "required_platform_permissions":[],
        "resource_scope":{"organization_id":"org","deployment_id":"deploy","tenant_id":"tenant","platform":"local","guild_ref":null,"resource_digest":Digest::from_bytes([1;32]),"subject_digest":Digest::from_bytes([2;32])},
        "side_effect":"None","reversibility":"ReversibleWithLoss","risk_class":"Low",
        "data_classes":["Operational"],"residency":"LocalOnly","retention":"Session",
        "preconditions":["fresh"],"invariants":["scoped"],"deadline_ms":1000,"rate_class":"bounded",
        "budgets":{"max_input_bytes":512,"max_output_bytes":512,"max_events":4,"max_calls":1},
        "confirmation_policy":"A0None","delegation_policy":"Forbidden","expiry_policy_ms":1000,"revocation_policy":"RecheckBeforeEffect","approval_policy":"A0None",
        "idempotency":"NaturallyIdempotent","compensation":null,"rollback":"None","postconditions":[],
        "adapter_bindings":[{"adapter_id":"recording","api_version":"1"}],
        "receipt_requirements":{"redacted":true,"require_postcondition_evidence":true},
        "fixtures":{"deterministic":Digest::from_bytes([3;32]),"adversarial":Digest::from_bytes([4;32]),"failure":Digest::from_bytes([5;32])},
        "promotion_thresholds":{"minimum_passed":1,"maximum_failed":0},
        "expiration_criteria":{"expires_at_ms":10000,"disable_on_schema_drift":true,"disable_on_missing_evidence":true}
    })).unwrap()
}
struct Provider(AtomicUsize, String);
impl ModelProvider for Provider {
    fn id(&self) -> &'static str {
        "governed-fixture"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            run.emit(&ModelEvent::ToolCall(ToolCall::new(
                "call", "observe", &self.1,
            )));
        }
        Ok(())
    }
}
struct Facts {
    value: GovernedFacts,
    reads: usize,
    change: bool,
    post: Postconditions,
}
impl GovernedFactProvider for Facts {
    fn observe(&mut self, _: &Request) -> Result<GovernedFacts, LifecycleError> {
        self.reads += 1;
        let mut value = self.value.clone();
        if self.change && self.reads > 1 {
            value.platform.revocation_epoch += 1;
        }
        Ok(value)
    }
    fn postconditions(&mut self, _: &Request, _: &BTreeSet<String>) -> Postconditions {
        self.post
    }
}
#[derive(Default)]
struct Ledger {
    stages: Vec<StageAudit>,
    fail_stage: Option<u8>,
    fail_finish: bool,
    intents: usize,
    execution_ids: Vec<Digest>,
    outcome: Option<TerminalOutcome>,
    cancel_prepare: Option<CancellationToken>,
    cancel_stage: Option<(u8, CancellationToken)>,
}
impl DurableLifecycle for Ledger {
    fn append_stage(&mut self, record: &StageAudit) -> Result<(), LifecycleError> {
        if self.fail_stage == Some(record.stage) {
            return Err(LifecycleError::Capacity);
        }
        if let Some((stage, token)) = &self.cancel_stage
            && *stage == record.stage
        {
            token.cancel();
        }
        self.stages.push(record.clone());
        Ok(())
    }
    fn prepare(&mut self, request: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError> {
        assert!(request.is_valid());
        if self.execution_ids.contains(&request.execution_id) {
            return Ok(PrepareOutcome::Uncertain);
        }
        let stage = StageAudit {
            version: 1,
            execution_id: request.execution_id,
            stage: 11,
            binding_digest: Digest::default(),
            decision: Decision::Allow,
            reason: LifecycleReason::Authorization(ReasonCode::Authorized),
        };
        self.append_stage(&stage)?;
        self.intents += 1;
        self.execution_ids.push(request.execution_id);
        if let Some(token) = &self.cancel_prepare {
            token.cancel();
        }
        Ok(PrepareOutcome::Prepared(PreparedExecution::after_commit(
            request.execution_id,
            request.binding.clone(),
        )))
    }
    fn finish(
        &mut self,
        _: &PreparedExecution,
        outcome: &TerminalOutcome,
    ) -> Result<(), LifecycleError> {
        if self.fail_finish {
            return Err(LifecycleError::Unavailable);
        }
        self.outcome = Some(outcome.clone());
        Ok(())
    }
    fn lookup(&mut self, _: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        Ok(self.outcome.clone())
    }
}
fn run(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
) -> (GovernedRunReport, RecordingActuator, usize) {
    run_with(floor, ledger, change, post, |_| {}, "{}")
}
fn run_with(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
    mutate: impl FnOnce(&mut Facts),
    input: &str,
) -> (GovernedRunReport, RecordingActuator, usize) {
    run_approved(floor, ledger, change, post, mutate, input, None)
}
fn run_approved(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
    mutate: impl FnOnce(&mut Facts),
    input: &str,
    approval: Option<ExactApproval>,
) -> (GovernedRunReport, RecordingActuator, usize) {
    let mut src = source();
    src.approval_policy = floor;
    let sources = SourceRegistry::compile([src]).unwrap();
    let compiled = sources.get("local.observe", "1.0.0").unwrap();
    let principal = Principal::new("actor", PrincipalKind::Service).unwrap();
    let request = Request::new(
        "attempt",
        "call",
        principal.clone(),
        compiled.source().resource_scope.clone(),
        "local.observe",
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
        principal,
        request.scope.clone(),
        compiled.package(),
        0,
        2000,
        ApprovalLevel::A0None,
        0,
        "p",
        "g",
        "s",
    )
    .unwrap();
    let mut facts = Facts {
        value: GovernedFacts {
            platform: PlatformFacts::new(BTreeSet::new(), 0, "p", "g", "s"),
            grants: vec![grant],
            used_decision_ids: BTreeSet::new(),
            predicates: BTreeSet::from(["fresh".into(), "scoped".into()]),
            captured_pre_state: None,
            observed_at_ms: 90,
            expires_at_ms: 200,
            permitted_data_classes: BTreeSet::from([DataClass::Operational]),
            residency: ResidencyPolicy::LocalOnly,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        },
        reads: 0,
        change,
        post,
    };
    mutate(&mut facts);
    let provider = Provider(AtomicUsize::new(0), input.into());
    let host = GovernedHost::new(
        &provider,
        &sources,
        vec![GovernedRegistration {
            tool_name: "observe".into(),
            request,
        }],
    )
    .unwrap();
    let clock = FixedClock::new(100);
    let token = ledger
        .cancel_prepare
        .clone()
        .or_else(|| ledger.cancel_stage.as_ref().map(|(_, token)| token.clone()))
        .unwrap_or_default();
    let mut actuator = RecordingActuator::new(Vec::<String>::new());
    let report = host
        .run(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &token,
            GovernedOptions {
                clock: &clock,
                host_budget: HostBudget::default(),
                approval,
                idempotency_key: None,
                requested_deadline_ms: 1000,
                requested_budgets: compiled.source().budgets.clone(),
            },
            &mut facts,
            ledger,
            &mut actuator,
        )
        .unwrap();
    (report, actuator, provider.0.load(Ordering::SeqCst))
}
#[test]
fn all_seventeen_stages_and_correlated_continuation() {
    let mut ledger = Ledger::default();
    let (report, actuator, runs) = run(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
    );
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(runs, 2);
    assert_eq!(actuator.records().len(), 1);
    assert_eq!(
        ledger.stages.iter().map(|s| s.stage).collect::<Vec<_>>(),
        (0..17).collect::<Vec<_>>()
    );
    assert_eq!(ledger.outcome.unwrap().state, OutcomeState::Succeeded);
}
#[test]
fn approval_suspends_without_effect_or_continuation() {
    let mut ledger = Ledger::default();
    let (report, actuator, runs) = run(
        ApprovalLevel::A2Manager,
        &mut ledger,
        false,
        Postconditions::Verified,
    );
    assert!(matches!(
        report.governance,
        GovernedStop::PendingApproval(_)
    ));
    let preview = report
        .pending_approval
        .expect("structured approval preview");
    assert_eq!(preview.tool, "observe");
    assert_eq!(preview.capability, "local.observe");
    assert_eq!(preview.version, "1.0.0");
    assert_eq!(preview.risk, RiskClass::Low);
    assert_eq!(preview.effect, SideEffect::None);
    assert_eq!(preview.reversibility, Reversibility::ReversibleWithLoss);
    assert_eq!(preview.uncertainty, PreviewUncertainty::RecordingOnly);
    assert!(preview.expires_at_ms > 100 && preview.expires_at_ms <= 1100);
    assert!(preview.postconditions.len() <= 32);
    assert!(preview.compensation.is_none());
    assert!(preview.tool.len() <= 128 && preview.capability.len() <= 128);
    assert_eq!(runs, 1);
    assert_eq!(actuator.records(), []);
    assert_eq!(ledger.intents, 0);
    assert_eq!(ledger.stages.last().unwrap().stage, 9);
}
#[test]
fn every_pre_effect_audit_failure_blocks_effect() {
    for stage in 0..=12 {
        let mut ledger = Ledger {
            fail_stage: Some(stage),
            ..Ledger::default()
        };
        let (_, actuator, runs) = run(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
        );
        assert!(actuator.records().is_empty(), "stage {stage}");
        assert_eq!(runs, 1);
    }
}
#[test]
fn changed_facts_and_cancel_after_prepare_block_effects() {
    let mut ledger = Ledger::default();
    let (_, actuator, _) = run(
        ApprovalLevel::A0None,
        &mut ledger,
        true,
        Postconditions::Verified,
    );
    assert_eq!(actuator.records(), []);
    assert_eq!(ledger.intents, 0);
    let mut ledger = Ledger {
        cancel_prepare: Some(CancellationToken::new()),
        ..Ledger::default()
    };
    let (_, actuator, _) = run(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
    );
    assert_eq!(actuator.records(), []);
    assert_eq!(ledger.intents, 1);
    assert_eq!(
        ledger.outcome.unwrap().reason,
        LifecycleReason::CancelledBeforeDispatch
    );
}
#[test]
fn lost_outcome_stays_uncertain_and_postconditions_never_guess_success() {
    let mut ledger = Ledger {
        fail_finish: true,
        ..Ledger::default()
    };
    let (report, actuator, _) = run(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
    );
    assert_eq!(report.governance, GovernedStop::Uncertain);
    assert_eq!(actuator.records().len(), 1);
    for (post, state) in [
        (Postconditions::Failed, OutcomeState::Failed),
        (Postconditions::Unverifiable, OutcomeState::Unverifiable),
    ] {
        let mut ledger = Ledger::default();
        let (report, actuator, runs) = run(ApprovalLevel::A0None, &mut ledger, false, post);
        assert_eq!(report.outcomes[0].state, state);
        assert_eq!(actuator.records().len(), 1);
        assert_eq!(runs, 1);
    }
}

#[test]
fn missing_grants_stale_facts_privacy_and_safety_fail_closed() {
    let mutations: [fn(&mut Facts); 6] = [
        |f| f.value.grants.clear(),
        |f| f.value.expires_at_ms = 100,
        |f| f.value.observed_at_ms = 101,
        |f| f.value.permitted_data_classes.clear(),
        |f| f.value.safety_allows = false,
        |f| f.value.rate_allows = false,
    ];
    for mutate in mutations {
        let mut ledger = Ledger::default();
        let (report, actuator, runs) = run_with(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
            mutate,
            "{}",
        );
        assert!(matches!(report.governance, GovernedStop::Refused(_)));
        assert_eq!(actuator.records(), []);
        assert_eq!(ledger.intents, 0);
        assert_eq!(runs, 1);
    }
}

#[test]
fn invalid_schema_raw_canaries_do_not_enter_durable_audit() {
    for input in ["RAW_SECRET_CANARY", "{\"secret\":\"RAW_SECRET_CANARY\"}"] {
        let mut ledger = Ledger::default();
        let (report, actuator, _) = run_with(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
            |_| {},
            input,
        );
        assert!(matches!(report.governance, GovernedStop::Refused(_)));
        assert_eq!(actuator.records(), []);
        assert!(
            !serde_json::to_string(&ledger.stages)
                .unwrap()
                .contains("RAW_SECRET_CANARY")
        );
    }
}

#[test]
fn post_effect_audit_loss_preserves_uncertainty() {
    for stage in 13..=16 {
        let mut ledger = Ledger {
            fail_stage: Some(stage),
            ..Ledger::default()
        };
        let (report, actuator, runs) = run(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
        );
        assert_eq!(report.governance, GovernedStop::Uncertain);
        assert_eq!(actuator.records().len(), 1);
        assert_eq!(runs, 1);
    }
}

#[test]
fn explicit_resubmission_rechecks_exact_preview_roles_and_ttl() {
    let mut ledger = Ledger::default();
    let (pending, _, _) = run(
        ApprovalLevel::A2Manager,
        &mut ledger,
        false,
        Postconditions::Verified,
    );
    let GovernedStop::PendingApproval(binding) = pending.governance else {
        panic!("preview required")
    };
    let approval = ExactApproval {
        issued_at_ms: 90,
        approval: Approval {
            decision_id: "decision".into(),
            call_id: "call".into(),
            call_digest: binding.request_digest,
            package_digest: binding.package_digest,
            grant_id: "grant".into(),
            approver: Principal::new("manager", PrincipalKind::GuildManager).unwrap(),
            coapprover: None,
            level: ApprovalLevel::A2Manager,
            expires_at_ms: 200,
            state: ApprovalState::Approved,
        },
        binding,
    };
    let (report, actuator, _) = run_approved(
        ApprovalLevel::A2Manager,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        "{}",
        Some(approval.clone()),
    );
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(actuator.records().len(), 1);
    for changed in [
        {
            let mut a = approval.clone();
            a.binding.precondition_digest = Digest::default();
            a
        },
        {
            let mut a = approval.clone();
            a.approval.approver.kind = PrincipalKind::Service;
            a
        },
        {
            let mut a = approval.clone();
            a.approval.expires_at_ms = 100;
            a
        },
        {
            let mut a = approval.clone();
            a.approval.expires_at_ms = 2000;
            a
        },
    ] {
        let mut ledger = Ledger::default();
        let (report, actuator, _) = run_approved(
            ApprovalLevel::A2Manager,
            &mut ledger,
            false,
            Postconditions::Verified,
            |_| {},
            "{}",
            Some(changed),
        );
        assert_ne!(report.governance, GovernedStop::Completed);
        assert_eq!(actuator.records(), []);
        assert_eq!(ledger.intents, 0);
    }
}

struct CompensationProvider(AtomicUsize);
impl ModelProvider for CompensationProvider {
    fn id(&self) -> &'static str {
        "compensation"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        match self.0.fetch_add(1, Ordering::SeqCst) {
            0 => {
                run.emit(&ModelEvent::ToolCall(ToolCall::new(
                    "original", "observe", "{}",
                )));
            }
            1 => {
                run.emit(&ModelEvent::ToolCall(ToolCall::new(
                    "inverse", "inverse", "{}",
                )));
            }
            _ => {}
        }
        Ok(())
    }
}
#[test]
fn compensation_has_separate_intent_and_stops_on_first_failure() {
    compensation_case(Postconditions::Failed);
    compensation_case(Postconditions::Verified);
}
#[allow(clippy::too_many_lines)] // Full original/inverse fixture keeps authority boundaries visible.
fn compensation_case(post: Postconditions) {
    let mut original = source();
    original.compensation = Some(CompensationSpec {
        capability_id: "local.inverse".into(),
        version: "1.0.0".into(),
        pre_state: BTreeSet::from(["captured".into()]),
        preauthorized: false,
    });
    original.rollback = RollbackSpec::ReverseOrderStopOnFailure;
    let mut inverse = source();
    inverse.id = "local.inverse".into();
    let sources = SourceRegistry::compile([original, inverse]).unwrap();
    let principal = Principal::new("actor", PrincipalKind::Service).unwrap();
    let mut registrations = Vec::new();
    let mut grants = Vec::new();
    for (tool, id) in [("observe", "local.observe"), ("inverse", "local.inverse")] {
        let compiled = sources.get(id, "1.0.0").unwrap();
        let request = Request::new(
            "attempt",
            "template",
            principal.clone(),
            compiled.source().resource_scope.clone(),
            id,
            "1.0.0",
            compiled.package().digest,
            Digest::default(),
            "p",
            "g",
            "s",
            0,
        )
        .unwrap();
        grants.push(
            Grant::active(
                tool,
                Principal::new("owner", PrincipalKind::OrganizationOwner).unwrap(),
                principal.clone(),
                request.scope.clone(),
                compiled.package(),
                0,
                2000,
                ApprovalLevel::A0None,
                0,
                "p",
                "g",
                "s",
            )
            .unwrap(),
        );
        registrations.push(GovernedRegistration {
            tool_name: tool.into(),
            request,
        });
    }
    let mut facts = Facts {
        value: GovernedFacts {
            platform: PlatformFacts::new(BTreeSet::new(), 0, "p", "g", "s"),
            grants,
            used_decision_ids: BTreeSet::new(),
            predicates: BTreeSet::from(["fresh".into(), "scoped".into()]),
            captured_pre_state: Some(Digest::from_bytes([6; 32])),
            observed_at_ms: 90,
            expires_at_ms: 200,
            permitted_data_classes: BTreeSet::from([DataClass::Operational]),
            residency: ResidencyPolicy::LocalOnly,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        },
        reads: 0,
        change: false,
        post: Postconditions::Failed,
    };
    let provider = CompensationProvider(AtomicUsize::new(0));
    let host = GovernedHost::new(&provider, &sources, registrations).unwrap();
    let clock = FixedClock::new(100);
    let options = || GovernedOptions {
        clock: &clock,
        host_budget: HostBudget::default(),
        approval: None,
        idempotency_key: None,
        requested_deadline_ms: 1000,
        requested_budgets: source().budgets,
    };
    let mut ledger = Ledger::default();
    let mut actuator = RecordingActuator::new(Vec::<String>::new());
    let token = CancellationToken::new();
    let mut sink = CollectingSink::default();
    let model = ModelRequest::new("fixture");
    let first = host
        .run(
            &model,
            &mut sink,
            &token,
            options(),
            &mut facts,
            &mut ledger,
            &mut actuator,
        )
        .unwrap();
    assert_eq!(first.outcomes[0].state, OutcomeState::Failed);
    assert_eq!(ledger.intents, 1);
    facts.post = post;
    let compensated = host
        .run_compensation(
            "observe",
            &first.executions[0],
            &model,
            &mut sink,
            &token,
            options(),
            &mut facts,
            &mut ledger,
            &mut actuator,
        )
        .unwrap();
    assert_eq!(
        compensated.outcomes[0].state,
        if post == Postconditions::Verified {
            OutcomeState::Compensated
        } else {
            OutcomeState::CompensationFailed
        }
    );
    assert_eq!(ledger.intents, 2);
    assert_eq!(actuator.records().len(), 2);
    assert_eq!(
        provider.0.load(Ordering::SeqCst),
        if post == Postconditions::Verified {
            3
        } else {
            2
        }
    );
    // Unknown original outcomes cannot authorize any inverse attempt.
    ledger.outcome.as_mut().unwrap().state = OutcomeState::Uncertain;
    assert!(matches!(
        host.run_compensation(
            "observe",
            &first.executions[0],
            &model,
            &mut sink,
            &token,
            options(),
            &mut facts,
            &mut ledger,
            &mut actuator
        ),
        Err(GovernedCompensationError::InvalidOriginal)
    ));
    assert_eq!(
        provider.0.load(Ordering::SeqCst),
        if post == Postconditions::Verified {
            3
        } else {
            2
        }
    );
}

#[test]
fn audit_callback_cancellation_before_prepare_and_dispatch_never_records() {
    for stage in [10, 11, 12] {
        let mut ledger = Ledger {
            cancel_stage: Some((stage, CancellationToken::new())),
            ..Ledger::default()
        };
        let (_, actuator, runs) = run(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
        );
        assert_eq!(actuator.records(), []);
        assert_eq!(runs, 1);
        assert_eq!(ledger.intents, usize::from(stage > 10));
    }
}
