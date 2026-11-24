use super::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[derive(Clone, Copy, Debug)]
enum Boundary {
    Prepare,
    Stage12,
}

#[derive(Clone, Copy, Debug)]
enum Change {
    GrantRevocation,
    Policy,
    Permission,
    Prerequisite,
    Safety,
    Rate,
}

impl Change {
    fn apply(self, facts: &mut GovernedFacts) {
        match self {
            Self::GrantRevocation => facts.grants[0].revocation = RevocationState::Revoked,
            Self::Policy => facts.platform.policy_version = "changed".into(),
            Self::Permission => facts.platform.permissions.clear(),
            Self::Prerequisite => {
                facts.predicates.remove("fresh");
            }
            Self::Safety => facts.safety_allows = false,
            Self::Rate => facts.rate_allows = false,
        }
    }
}

struct RefreshFacts {
    inner: Facts,
    changed: Arc<AtomicBool>,
    change: Option<Change>,
}

impl GovernedFactProvider for RefreshFacts {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError> {
        let mut facts = self.inner.observe(request)?;
        if self.changed.load(Ordering::SeqCst)
            && let Some(change) = self.change
        {
            change.apply(&mut facts);
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

struct RefreshLedger {
    inner: Ledger,
    changed: Arc<AtomicBool>,
    boundary: Boundary,
}

impl DurableLifecycle for RefreshLedger {
    fn refuse_replay(&mut self, id: Digest) -> Result<bool, LifecycleError> {
        self.inner.refuse_replay(id)
    }

    fn append_stage(&mut self, record: &StageAudit) -> Result<(), LifecycleError> {
        if record.stage == 12 && matches!(self.boundary, Boundary::Stage12) {
            self.changed.store(true, Ordering::SeqCst);
        }
        self.inner.append_stage(record)
    }

    fn prepare(&mut self, request: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError> {
        let result = self.inner.prepare(request)?;
        if matches!(result, PrepareOutcome::Prepared(_))
            && matches!(self.boundary, Boundary::Prepare)
        {
            self.changed.store(true, Ordering::SeqCst);
        }
        Ok(result)
    }

    fn finish(
        &mut self,
        prepared: &PreparedExecution,
        outcome: &TerminalOutcome,
    ) -> Result<(), LifecycleError> {
        self.inner.finish(prepared, outcome)
    }

    fn lookup(&mut self, id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        self.inner.lookup(id)
    }
}

fn exercise(
    boundary: Boundary,
    change: Option<Change>,
) -> (GovernedRunReport, RecordingActuator, RefreshLedger) {
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let mut src = source();
    src.required_platform_permissions = BTreeSet::from(["record".into()]);
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
    let changed = Arc::new(AtomicBool::new(false));
    let mut facts = RefreshFacts {
        inner: Facts {
            value: GovernedFacts {
                platform: PlatformFacts::new(BTreeSet::from(["record"]), 0, "p", "g", "s"),
                grants: vec![grant],
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
            },
            reads: 0,
            change: false,
            post: Postconditions::Verified,
        },
        changed: changed.clone(),
        change,
    };
    let host = GovernedHost::new(
        &provider,
        &sources,
        vec![GovernedRegistration {
            tool_name: "observe".into(),
            request,
        }],
    )
    .unwrap();
    let mut ledger = RefreshLedger {
        inner: Ledger::default(),
        changed,
        boundary,
    };
    let mut actuator = RecordingActuator::new(["observed"]);
    let report = host
        .run(
            &ModelRequest::new("fixture"),
            &mut CollectingSink::default(),
            &CancellationToken::new(),
            GovernedOptions {
                clock: &FixedClock::new(100),
                host_budget: HostBudget::default(),
                approval: None,
                idempotency_key: None,
                requested_deadline_ms: 1000,
                requested_budgets: compiled.source().budgets.clone(),
            },
            &mut facts,
            &mut ledger,
            &mut actuator,
        )
        .unwrap();
    (report, actuator, ledger)
}

#[test]
fn mutable_authority_changes_after_fallible_boundaries_never_record() {
    for boundary in [Boundary::Prepare, Boundary::Stage12] {
        for change in [
            Change::GrantRevocation,
            Change::Policy,
            Change::Permission,
            Change::Prerequisite,
            Change::Safety,
            Change::Rate,
        ] {
            let (report, actuator, ledger) = exercise(boundary, Some(change));
            assert!(actuator.records().is_empty(), "{boundary:?} {change:?}");
            assert_eq!(report.outcomes[0].state, OutcomeState::NotDispatched);
            assert_eq!(ledger.inner.intents, 1);
        }
    }
}

#[test]
fn unchanged_authority_still_records_after_each_refresh() {
    for boundary in [Boundary::Prepare, Boundary::Stage12] {
        let (report, actuator, ledger) = exercise(boundary, None);
        assert_eq!(actuator.records().len(), 1, "{boundary:?}");
        assert_eq!(report.governance, GovernedStop::Completed);
        assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
        assert_eq!(ledger.inner.intents, 1);
    }
}
