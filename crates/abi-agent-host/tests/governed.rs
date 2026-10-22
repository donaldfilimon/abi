//! Fail-closed recording governance regressions.
#[path = "governed/authority_expiry.rs"]
mod authority_expiry;
#[path = "governed/token_admission.rs"]
mod token_admission;

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
        "idempotency":"NaturallyIdempotent","compensation":null,"rollback":"None","postconditions":["observed"],
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
                expires_at_ms: 200,
                evidence_digest: Some(Digest::from_bytes([7; 32])),
                source: ObservationSource::RecordingFixture,
                status: self.post,
            })
            .collect())
    }
}
#[derive(Default)]
struct Ledger {
    stages: Vec<StageAudit>,
    prepares: Vec<PrepareRequest>,
    fail_stage: Option<u8>,
    fail_finish: bool,
    intents: usize,
    execution_ids: Vec<Digest>,
    compensated_origins: Vec<Digest>,
    invalidate_original_on_prepare: bool,
    outcome: Option<TerminalOutcome>,
    outcomes_by_id: std::collections::HashMap<Digest, TerminalOutcome>,
    cancel_prepare: Option<CancellationToken>,
    cancel_stage: Option<(u8, CancellationToken)>,
}
impl DurableLifecycle for Ledger {
    fn refuse_replay(&mut self, id: Digest) -> Result<bool, LifecycleError> {
        Ok(self.execution_ids.contains(&id))
    }
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
        self.prepares.push(request.clone());
        if self.execution_ids.contains(&request.execution_id) {
            return Ok(PrepareOutcome::Uncertain);
        }
        if let Some(origin) = &request.binding.compensation_origin
            && self.invalidate_original_on_prepare
        {
            self.outcomes_by_id
                .get_mut(&origin.original_execution_id)
                .unwrap()
                .state = OutcomeState::Uncertain;
        }
        if let Some(origin) = &request.binding.compensation_origin
            && (self
                .compensated_origins
                .contains(&origin.original_execution_id)
                || !self
                    .outcomes_by_id
                    .get(&origin.original_execution_id)
                    .is_some_and(|o| o.state == OutcomeState::Failed))
        {
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
        if let Some(origin) = &request.binding.compensation_origin {
            self.compensated_origins.push(origin.original_execution_id);
        }
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
        prepared: &PreparedExecution,
        outcome: &TerminalOutcome,
    ) -> Result<(), LifecycleError> {
        if self.fail_finish {
            return Err(LifecycleError::Unavailable);
        }
        self.outcome = Some(outcome.clone());
        self.outcomes_by_id
            .insert(prepared.execution_id(), outcome.clone());
        Ok(())
    }
    fn lookup(&mut self, id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        Ok(self.outcomes_by_id.get(&id).cloned())
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
    let provider = Provider(AtomicUsize::new(0), input.into());
    let (report, actuator) = run_provider_fixture(
        floor,
        ledger,
        change,
        post,
        mutate,
        approval,
        &provider,
        HostBudget::default(),
    );
    (report.unwrap(), actuator, provider.0.load(Ordering::SeqCst))
}
#[allow(clippy::too_many_arguments)]
fn run_provider_fixture(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
    mutate: impl FnOnce(&mut Facts),
    approval: Option<ExactApproval>,
    provider: &dyn ModelProvider,
    budget: HostBudget,
) -> (
    Result<GovernedRunReport, Box<GovernedRunFailure>>,
    RecordingActuator,
) {
    run_identity_fixture(
        floor,
        ledger,
        change,
        post,
        mutate,
        approval,
        provider,
        budget,
        ("attempt", "actor", "tenant"),
    )
}
#[allow(clippy::too_many_arguments)]
fn run_identity_fixture(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
    mutate: impl FnOnce(&mut Facts),
    approval: Option<ExactApproval>,
    provider: &dyn ModelProvider,
    budget: HostBudget,
    identity: (&str, &str, &str),
) -> (
    Result<GovernedRunReport, Box<GovernedRunFailure>>,
    RecordingActuator,
) {
    run_source_fixture(
        floor,
        ledger,
        change,
        post,
        mutate,
        approval,
        provider,
        budget,
        identity,
        source(),
    )
}
#[allow(clippy::too_many_arguments)]
fn run_source_fixture(
    floor: ApprovalLevel,
    ledger: &mut Ledger,
    change: bool,
    post: Postconditions,
    mutate: impl FnOnce(&mut Facts),
    approval: Option<ExactApproval>,
    provider: &dyn ModelProvider,
    budget: HostBudget,
    identity: (&str, &str, &str),
    mut src: PackageSource,
) -> (
    Result<GovernedRunReport, Box<GovernedRunFailure>>,
    RecordingActuator,
) {
    src.approval_policy = floor;
    src.resource_scope.tenant_id = identity.2.into();
    let sources = SourceRegistry::compile([src]).unwrap();
    let compiled = sources.get("local.observe", "1.0.0").unwrap();
    let principal = Principal::new(identity.1, PrincipalKind::Service).unwrap();
    let request = Request::new(
        identity.0,
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
    let host = GovernedHost::new(
        provider,
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
    let mut actuator = RecordingActuator::new(["observed"]);
    let report = host.run(
        &ModelRequest::new("fixture"),
        &mut CollectingSink::default(),
        &token,
        GovernedOptions {
            clock: &clock,
            host_budget: budget,
            approval,
            idempotency_key: None,
            requested_deadline_ms: 1000,
            requested_budgets: compiled.source().budgets.clone(),
        },
        &mut facts,
        ledger,
        &mut actuator,
    );
    (report, actuator)
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
            call_id: pending
                .pending_approval
                .as_ref()
                .unwrap()
                .call_reference
                .clone(),
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

struct FailAfterEffect(AtomicUsize, bool);
impl ModelProvider for FailAfterEffect {
    fn id(&self) -> &'static str {
        "failure-after-effect"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            run.emit(&ModelEvent::ToolCall(ToolCall::new(
                "call", "observe", "{}",
            )));
            return Ok(());
        }
        if self.1 {
            run.emit(&ModelEvent::ToolResult(ToolResult::with_limit(
                "fake",
                ToolStatus::Ok,
                "PRIVATE_CANARY",
                100,
            )));
            Ok(())
        } else {
            Err(RuntimeError::ScriptExhausted { scripted: 0 })
        }
    }
}
#[test]
fn later_driver_failure_retains_prepared_accounting() {
    for (forged, limited) in [(false, false), (true, false), (false, true)] {
        let provider = FailAfterEffect(AtomicUsize::new(0), forged);
        let mut budget = HostBudget::default();
        if limited {
            budget.max_provider_runs = 1;
        }
        let mut ledger = Ledger::default();
        let (result, actuator) = run_provider_fixture(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
            |_| {},
            None,
            &provider,
            budget,
        );
        assert_eq!(actuator.records().len(), 1);
        let failure = result.unwrap_err();
        assert_eq!(failure.executions.len(), 1);
        assert_eq!(failure.outcomes[0].state, OutcomeState::Succeeded);
        assert_eq!(
            failure.kind,
            if limited {
                GovernedDriverFailureKind::Budget(HostBudgetLimit::ProviderRuns)
            } else if forged {
                GovernedDriverFailureKind::Protocol
            } else {
                GovernedDriverFailureKind::Provider
            }
        );
        assert!(!format!("{failure:?}").contains("PRIVATE_CANARY"));
    }
}

#[path = "governed/compensation.rs"]
mod compensation;

#[test]
fn durable_identity_uses_trusted_operation_not_model_call_id_alone() {
    let mut ledger = Ledger::default();
    for identity in [
        ("attempt", "actor", "tenant"),
        ("attempt2", "actor", "tenant"),
        ("attempt", "actor2", "tenant"),
        ("attempt", "actor", "tenant2"),
    ] {
        let provider = Provider(AtomicUsize::new(0), "{}".into());
        let (result, actuator) = run_identity_fixture(
            ApprovalLevel::A0None,
            &mut ledger,
            false,
            Postconditions::Verified,
            |_| {},
            None,
            &provider,
            HostBudget::default(),
            identity,
        );
        assert_eq!(result.unwrap().governance, GovernedStop::Completed);
        assert_eq!(actuator.records().len(), 1);
    }
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let (result, actuator) = run_identity_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &provider,
        HostBudget::default(),
        ("attempt", "actor", "tenant"),
    );
    assert_eq!(
        result.unwrap().governance,
        GovernedStop::Refused(LifecycleReason::Authorization(ReasonCode::ApprovalReplayed))
    );
    assert_eq!(actuator.records(), []);
}
struct CanaryIdProvider(AtomicUsize);
impl ModelProvider for CanaryIdProvider {
    fn id(&self) -> &'static str {
        "canary-id"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            run.emit(&ModelEvent::ToolCall(ToolCall::new(
                "model_call_canary",
                "observe",
                "{}",
            )));
        }
        Ok(())
    }
}
#[test]
fn model_call_id_canary_stays_only_in_provider_correlation() {
    let mut ledger = Ledger::default();
    let provider = CanaryIdProvider(AtomicUsize::new(0));
    let (result, actuator) = run_provider_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &provider,
        HostBudget::default(),
    );
    let report = result.unwrap();
    assert_eq!(report.driver.tool_results[0].call_id, "model_call_canary");
    assert!(
        !serde_json::to_string(&ledger.prepares)
            .unwrap()
            .contains("model_call_canary")
    );
    assert!(
        !serde_json::to_string(&ledger.stages)
            .unwrap()
            .contains("model_call_canary")
    );
    assert!(
        !serde_json::to_string(actuator.records())
            .unwrap()
            .contains("model_call_canary")
    );
    assert!(
        !format!(
            "{:?}{:?}{:?}",
            report.executions, report.outcomes, report.pending_approval
        )
        .contains("model_call_canary")
    );
}

struct ForgedFirst;
impl ModelProvider for ForgedFirst {
    fn id(&self) -> &'static str {
        "forged-first"
    }
    fn run(&self, _: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        run.emit(&ModelEvent::ToolResult(ToolResult::with_limit(
            "RAW_REJECTED_ID",
            ToolStatus::Ok,
            "RAW_RESULT",
            100,
        )));
        Ok(())
    }
}
#[test]
fn prehook_protocol_refusal_has_fallible_redacted_terminal_audit() {
    let mut ledger = Ledger::default();
    let (result, actuator) = run_provider_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &ForgedFirst,
        HostBudget::default(),
    );
    assert!(result.is_err());
    assert_eq!(actuator.records(), []);
    assert_eq!(ledger.stages.len(), 1);
    assert_eq!(ledger.stages[0].stage, 0);
    assert_eq!(ledger.stages[0].decision, Decision::Deny);
    assert!(
        !serde_json::to_string(&ledger.stages)
            .unwrap()
            .contains("RAW_")
    );
}
#[test]
fn rejected_driver_and_failed_audit_remain_separate_closed_errors() {
    let mut ledger = Ledger {
        fail_stage: Some(0),
        ..Ledger::default()
    };
    let (result, _) = run_provider_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &ForgedFirst,
        HostBudget::default(),
    );
    let failure = result.unwrap_err();
    assert_eq!(failure.kind, GovernedDriverFailureKind::Protocol);
    assert_eq!(failure.audit_error, Some(LifecycleError::Capacity));
    assert!(!format!("{failure:?}").contains("RAW_"));
}

#[test]
fn recording_output_schema_failure_is_distinct_from_predicate_failure() {
    let mut src = source();
    src.output_schema = json!({"type":"string"});
    let mut ledger = Ledger::default();
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let (result, actuator) = run_source_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &provider,
        HostBudget::default(),
        ("attempt", "actor", "tenant"),
        src,
    );
    let report = result.unwrap();
    assert_eq!(actuator.records().len(), 1);
    assert_eq!(report.outcomes[0].state, OutcomeState::Failed);
    assert_eq!(report.outcomes[0].reason, LifecycleReason::AdapterFailure);
}
#[test]
fn driver_budget_stop_before_hook_is_also_audited() {
    let mut ledger = Ledger::default();
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let budget = HostBudget {
        max_output_tokens: 0,
        ..HostBudget::default()
    };
    let (result, actuator) = run_provider_fixture(
        ApprovalLevel::A0None,
        &mut ledger,
        false,
        Postconditions::Verified,
        |_| {},
        None,
        &provider,
        budget,
    );
    let report = result.unwrap();
    assert_ne!(report.governance, GovernedStop::Completed);
    assert_eq!(actuator.records(), []);
    assert_eq!(ledger.stages.len(), 1);
}
