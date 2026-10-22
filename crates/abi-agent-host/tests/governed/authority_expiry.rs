use super::*;
use std::sync::{Arc, atomic::AtomicU64};

struct MovingClock(Arc<AtomicU64>);
impl Clock for MovingClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Prepare,
    Stage12,
}
#[derive(Clone, Copy, Debug)]
enum Authority {
    Grant,
    Source,
    Facts,
    Approval,
}
struct ExpiryLedger {
    inner: Ledger,
    clock: Arc<AtomicU64>,
    boundary: Boundary,
    advance_to: u64,
    consumes: usize,
}
impl DurableLifecycle for ExpiryLedger {
    fn refuse_replay(&mut self, id: Digest) -> Result<bool, LifecycleError> {
        self.inner.refuse_replay(id)
    }
    fn append_stage(&mut self, record: &StageAudit) -> Result<(), LifecycleError> {
        if record.stage == 12 && matches!(self.boundary, Boundary::Stage12) {
            self.clock.store(self.advance_to, Ordering::SeqCst);
        }
        self.inner.append_stage(record)
    }
    fn prepare(&mut self, request: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError> {
        let result = self.inner.prepare(request)?;
        if matches!(result, PrepareOutcome::Prepared(_)) {
            self.consumes += usize::from(request.approval.is_some());
            if matches!(self.boundary, Boundary::Prepare) {
                self.clock.store(self.advance_to, Ordering::SeqCst);
            }
        }
        Ok(result)
    }
    fn finish(
        &mut self,
        prepared: &PreparedExecution,
        outcome: &TerminalOutcome,
    ) -> Result<(), LifecycleError> {
        assert!(outcome.is_valid_terminal());
        self.inner.finish(prepared, outcome)
    }
    fn lookup(&mut self, id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        self.inner.lookup(id)
    }
}
struct ExpiryFacts {
    inner: Facts,
    expires_latest: bool,
}
impl GovernedFactProvider for ExpiryFacts {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError> {
        let mut facts = self.inner.observe(request)?;
        // The guard must use the last validated snapshot, not the initial one.
        if self.expires_latest && self.inner.reads >= 3 {
            facts.expires_at_ms = 200;
        }
        Ok(facts)
    }
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        self.inner.postconditions(request, required)
    }
}

fn authority_fixture(
    authority: Authority,
    human: bool,
) -> (SourceRegistry, Request, GovernedFacts) {
    let mut src = source();
    if human {
        src.approval_policy = ApprovalLevel::A2Manager;
    }
    if matches!(authority, Authority::Source) {
        src.expiration_criteria.expires_at_ms = 200;
    }
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
        if matches!(authority, Authority::Grant) {
            200
        } else {
            2000
        },
        ApprovalLevel::A0None,
        0,
        "p",
        "g",
        "s",
    )
    .unwrap();
    let mut alternative = grant.clone();
    alternative.id = "z-alternative".into();
    alternative.expires_at_ms = 2000;
    let value = GovernedFacts {
        platform: PlatformFacts::new(BTreeSet::new(), 0, "p", "g", "s"),
        grants: vec![grant, alternative],
        used_decision_ids: BTreeSet::new(),
        predicates: BTreeSet::from(["fresh".into(), "scoped".into()]),
        captured_pre_state: None,
        observed_at_ms: 90,
        expires_at_ms: 2000,
        permitted_data_classes: BTreeSet::from([DataClass::Operational]),
        residency: ResidencyPolicy::LocalOnly,
        retention: RetentionClass::Session,
        safety_allows: true,
        rate_allows: true,
    };
    (sources, request, value)
}

fn exercise(
    authority: Authority,
    boundary: Boundary,
    human: bool,
    now: u64,
    failure: u8,
) -> (
    GovernedRunReport,
    RecordingActuator,
    ExpiryLedger,
    CancellationToken,
) {
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let (sources, request, value) = authority_fixture(authority, human);
    let compiled = sources.get("local.observe", "1.0.0").unwrap();
    let host = GovernedHost::new(
        &provider,
        &sources,
        vec![GovernedRegistration {
            tool_name: "observe".into(),
            request,
        }],
    )
    .unwrap();
    let clock = MovingClock(Arc::new(AtomicU64::new(100)));
    let token = CancellationToken::new();
    let options = |approval| GovernedOptions {
        clock: &clock,
        host_budget: HostBudget::default(),
        approval,
        idempotency_key: None,
        requested_deadline_ms: 1000,
        requested_budgets: compiled.source().budgets.clone(),
    };
    let facts = || ExpiryFacts {
        inner: Facts {
            value: value.clone(),
            reads: 0,
            change: false,
            post: Postconditions::Verified,
        },
        expires_latest: matches!(authority, Authority::Facts),
    };
    let approval = if human {
        let pending = host
            .run(
                &ModelRequest::new("fixture"),
                &mut CollectingSink::default(),
                &token,
                options(None),
                &mut facts(),
                &mut Ledger::default(),
                &mut RecordingActuator::new(["observed"]),
            )
            .unwrap();
        let preview = pending.pending_approval.unwrap();
        provider.0.store(0, Ordering::SeqCst);
        Some(ExactApproval {
            issued_at_ms: 90,
            approval: Approval {
                decision_id: "decision".into(),
                call_id: preview.call_reference,
                call_digest: preview.binding.request_digest,
                package_digest: preview.binding.package_digest,
                grant_id: "grant".into(),
                approver: Principal::new("manager", PrincipalKind::GuildManager).unwrap(),
                coapprover: None,
                level: ApprovalLevel::A2Manager,
                expires_at_ms: if matches!(authority, Authority::Approval) {
                    200
                } else {
                    1000
                },
                state: ApprovalState::Approved,
            },
            binding: preview.binding,
        })
    } else {
        None
    };
    let mut ledger = ExpiryLedger {
        inner: Ledger {
            fail_finish: failure == 1,
            fail_stage: (failure == 2).then_some(12),
            ..Ledger::default()
        },
        clock: clock.0.clone(),
        boundary,
        advance_to: now,
        consumes: 0,
    };
    let mut actuator = RecordingActuator::new(["observed"]);
    let report = host
        .run(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &token,
            options(approval),
            &mut facts(),
            &mut ledger,
            &mut actuator,
        )
        .unwrap();
    (report, actuator, ledger, token)
}

fn expired(authority: Authority, boundary: Boundary) {
    for human in [false, true] {
        if matches!(authority, Authority::Approval) && !human {
            continue;
        }
        let (report, actuator, ledger, token) = exercise(authority, boundary, human, 200, 0);
        assert!(
            actuator.records().is_empty(),
            "{authority:?} {boundary:?} human={human}"
        );
        assert!(
            !token.is_cancelled(),
            "host cancellation must not mutate shared latch"
        );
        assert_eq!(ledger.inner.prepares.len(), 1);
        assert_eq!(ledger.inner.intents, 1);
        assert_eq!(ledger.consumes, usize::from(human));
        let outcome = ledger.inner.outcome.unwrap();
        assert_eq!(outcome.state, OutcomeState::NotDispatched);
        assert_eq!(outcome.reason, LifecycleReason::CancelledBeforeDispatch);
        assert_eq!(outcome.result_digest, None);
        assert_eq!(report.outcomes, vec![outcome]);
        assert_eq!(report.executions.len(), 1);
        let denial = ledger.inner.stages.last().unwrap();
        assert_eq!(denial.stage, 12);
        assert_eq!(denial.decision, Decision::Deny);
        assert_ne!(denial.binding_digest, Digest::default());
    }
}

#[test]
fn grant_expires_during_prepare() {
    expired(Authority::Grant, Boundary::Prepare);
}
#[test]
fn grant_expires_during_stage12() {
    expired(Authority::Grant, Boundary::Stage12);
}
#[test]
fn source_expires_during_prepare() {
    expired(Authority::Source, Boundary::Prepare);
}
#[test]
fn source_expires_during_stage12() {
    expired(Authority::Source, Boundary::Stage12);
}
#[test]
fn latest_facts_expire_during_prepare() {
    expired(Authority::Facts, Boundary::Prepare);
}
#[test]
fn latest_facts_expire_during_stage12() {
    expired(Authority::Facts, Boundary::Stage12);
}
#[test]
fn approval_expires_during_prepare() {
    expired(Authority::Approval, Boundary::Prepare);
}
#[test]
fn approval_expires_during_stage12() {
    expired(Authority::Approval, Boundary::Stage12);
}

#[test]
fn just_before_expiry_still_records() {
    for authority in [
        Authority::Grant,
        Authority::Source,
        Authority::Facts,
        Authority::Approval,
    ] {
        for boundary in [Boundary::Prepare, Boundary::Stage12] {
            for human in [false, true] {
                if matches!(authority, Authority::Approval) && !human {
                    continue;
                }
                let (report, actuator, ledger, token) =
                    exercise(authority, boundary, human, 199, 0);
                assert_eq!(
                    actuator.records().len(),
                    1,
                    "{authority:?} {boundary:?} human={human}"
                );
                assert_eq!(report.governance, GovernedStop::Completed);
                assert_eq!(ledger.inner.prepares.len(), 1);
                assert_eq!(ledger.consumes, usize::from(human));
                assert!(!token.is_cancelled());
            }
        }
    }
}

#[test]
fn expiry_terminal_persistence_and_audit_failure_never_record() {
    for boundary in [Boundary::Prepare, Boundary::Stage12] {
        for human in [false, true] {
            for failure in [1, 2] {
                let (report, actuator, ledger, token) =
                    exercise(Authority::Grant, boundary, human, 200, failure);
                assert!(
                    actuator.records().is_empty(),
                    "{boundary:?} human={human} failure={failure}"
                );
                assert!(matches!(report.governance, GovernedStop::Uncertain));
                assert_eq!(ledger.inner.prepares.len(), 1);
                assert_eq!(ledger.consumes, usize::from(human));
                assert!(!token.is_cancelled());
            }
        }
    }
}
