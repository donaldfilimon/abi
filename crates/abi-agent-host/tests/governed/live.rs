use super::*;
use std::sync::{Arc, atomic::AtomicBool};
fn fixture() -> (SourceRegistry, Request, GovernedFacts) {
    let mut src = source();
    src.adapter_bindings = vec![AdapterBinding {
        adapter_id: "live-fixture".into(),
        api_version: "1".into(),
    }];
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

struct Live {
    effects: usize,
    mode: u8,
}
impl GovernedLiveExecutor for Live {
    fn binding(&self) -> AdapterBinding {
        AdapterBinding {
            adapter_id: "live-fixture".into(),
            api_version: "1".into(),
        }
    }
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        _: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        assert_eq!(call.name, "observe");
        assert_eq!(prepared.binding().request_digest, request.request_digest);
        assert_eq!(prepared.binding().package_digest, source.package().digest);
        self.effects += 1;
        match self.mode {
            1 => Err(ToolExecutionError::new("unknown")),
            2 => panic!("adapter panic"),
            3 => Ok(ToolOutput::ok("{}")),
            4 => Ok(ToolOutput::ok("true ".repeat(200))),
            _ => Ok(ToolOutput::ok("true")),
        }
    }
}
struct MutatingLedger {
    inner: Ledger,
    changed: Arc<AtomicBool>,
    boundary: u8,
}
impl DurableLifecycle for MutatingLedger {
    fn refuse_replay(&mut self, id: Digest) -> Result<bool, LifecycleError> {
        self.inner.refuse_replay(id)
    }
    fn append_stage(&mut self, stage: &StageAudit) -> Result<(), LifecycleError> {
        if stage.stage == 12 && self.boundary == 12 {
            self.changed.store(true, Ordering::SeqCst);
        }
        self.inner.append_stage(stage)
    }
    fn prepare(&mut self, request: &PrepareRequest) -> Result<PrepareOutcome, LifecycleError> {
        let result = self.inner.prepare(request)?;
        if self.boundary == 11 {
            self.changed.store(true, Ordering::SeqCst);
        }
        Ok(result)
    }
    fn finish(&mut self, p: &PreparedExecution, o: &TerminalOutcome) -> Result<(), LifecycleError> {
        self.inner.finish(p, o)
    }
    fn lookup(&mut self, id: Digest) -> Result<Option<TerminalOutcome>, LifecycleError> {
        self.inner.lookup(id)
    }
}
struct LiveFacts {
    inner: Facts,
    changed: Arc<AtomicBool>,
    revoke: bool,
}
impl GovernedFactProvider for LiveFacts {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError> {
        let mut value = self.inner.observe(request)?;
        if self.changed.load(Ordering::SeqCst) {
            if self.revoke {
                value.platform.revocation_epoch += 1;
            } else {
                value.predicates.remove("fresh");
            }
        }
        Ok(value)
    }
    fn postconditions(
        &mut self,
        r: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        self.inner.postconditions(r, required)
    }
}
fn execute(
    ledger: &mut MutatingLedger,
    executor: &mut Live,
    revoke: bool,
) -> Result<GovernedRunReport, Box<GovernedRunFailure>> {
    let (sources, request, value) = fixture();
    let provider = Provider(AtomicUsize::new(0), "{}".into());
    let host = GovernedHost::new(
        &provider,
        &sources,
        vec![GovernedRegistration {
            tool_name: "observe".into(),
            request,
        }],
    )
    .unwrap();
    let mut facts = LiveFacts {
        inner: Facts {
            value,
            reads: 0,
            change: false,
            post: Postconditions::Verified,
        },
        changed: ledger.changed.clone(),
        revoke,
    };
    host.run_live(
        &ModelRequest::new("observe"),
        &mut CollectingSink::default(),
        &CancellationToken::new(),
        GovernedOptions {
            clock: &FixedClock::new(100),
            host_budget: HostBudget::default(),
            approval: None,
            idempotency_key: None,
            requested_deadline_ms: 1000,
            requested_budgets: sources
                .get("local.observe", "1.0.0")
                .unwrap()
                .source()
                .budgets
                .clone(),
        },
        &mut facts,
        ledger,
        executor,
    )
}
fn ledger(boundary: u8) -> MutatingLedger {
    MutatingLedger {
        inner: Ledger::default(),
        changed: Arc::new(AtomicBool::new(false)),
        boundary,
    }
}
#[test]
fn live_dispatch_returns_verified_actual_output() {
    let mut ledger = ledger(0);
    let mut live = Live {
        effects: 0,
        mode: 0,
    };
    let report = execute(&mut ledger, &mut live, false).unwrap();
    assert_eq!(report.governance, GovernedStop::Completed);
    assert_eq!(live.effects, 1);
    assert_eq!(report.outcomes[0].state, OutcomeState::Succeeded);
    assert_eq!(report.driver.tool_results[0].payload.text(), "true");
}
#[test]
fn prepare_and_stage12_authority_mutations_never_dispatch() {
    for boundary in [11, 12] {
        for revoke in [false, true] {
            let mut ledger = ledger(boundary);
            let mut live = Live {
                effects: 0,
                mode: 0,
            };
            let report = execute(&mut ledger, &mut live, revoke).unwrap();
            assert_eq!(live.effects, 0);
            assert_eq!(report.outcomes[0].state, OutcomeState::NotDispatched);
        }
    }
}
#[test]
fn ambiguous_errors_panics_and_lost_terminal_never_replay() {
    for mode in [0, 1, 2] {
        let mut ledger = ledger(0);
        ledger.inner.fail_finish = mode == 0;
        let mut live = Live { effects: 0, mode };
        let report = execute(&mut ledger, &mut live, false).unwrap();
        assert_eq!(report.governance, GovernedStop::Uncertain);
        assert_eq!(live.effects, 1);
        let report = execute(&mut ledger, &mut live, false).unwrap();
        assert!(matches!(report.governance, GovernedStop::Refused(_)));
        assert_eq!(live.effects, 1);
    }
}
#[test]
fn output_schema_and_bounds_reject_actual_live_output() {
    for mode in [3, 4] {
        let mut ledger = ledger(0);
        let mut live = Live { effects: 0, mode };
        let report = execute(&mut ledger, &mut live, false).unwrap();
        assert_eq!(report.outcomes[0].state, OutcomeState::Failed);
        assert_eq!(live.effects, 1);
    }
}
