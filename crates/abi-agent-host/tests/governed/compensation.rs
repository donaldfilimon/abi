use super::*;

struct CompensationProvider(AtomicUsize, bool);
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
                if self.1 {
                    run.emit(&ModelEvent::ToolCall(ToolCall::new(
                        "inverse-second",
                        "inverse",
                        "{}",
                    )));
                }
            }
            3 => {
                run.emit(&ModelEvent::ToolCall(ToolCall::new(
                    "inverse-fresh",
                    "inverse",
                    "{}",
                )));
            }
            _ => {}
        }
        Ok(())
    }
}
#[test]
fn compensation_has_separate_intent_and_stops_on_first_failure() {
    compensation_case(Postconditions::Failed, ApprovalLevel::A0None, false, 0);
    compensation_case(Postconditions::Verified, ApprovalLevel::A0None, false, 0);
}
#[allow(clippy::too_many_lines)] // Full original/inverse fixture keeps authority boundaries visible.
fn compensation_case(post: Postconditions, floor: ApprovalLevel, preauthorized: bool, retry: u8) {
    let mut original = source();
    original.approval_policy = floor;
    original.budgets.max_calls = 2;
    original.budgets.max_events = 8;
    original.compensation = Some(CompensationSpec {
        capability_id: "local.inverse".into(),
        version: "1.0.0".into(),
        pre_state: BTreeSet::from(["captured".into()]),
        preauthorized,
    });
    original.rollback = RollbackSpec::ReverseOrderStopOnFailure;
    let mut inverse = source();
    inverse.id = "local.inverse".into();
    inverse.budgets.max_calls = 2;
    inverse.budgets.max_events = 8;
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
    let provider = CompensationProvider(AtomicUsize::new(0), retry == 2);
    let host = GovernedHost::new(&provider, &sources, registrations).unwrap();
    let clock = FixedClock::new(100);
    let options = || GovernedOptions {
        clock: &clock,
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
    };
    let mut ledger = Ledger::default();
    let mut actuator = RecordingActuator::new(["observed"]);
    let token = CancellationToken::new();
    let mut sink = CollectingSink::default();
    let model = ModelRequest::new("fixture");
    let mut first = host
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
    if let GovernedStop::PendingApproval(binding) = &first.governance {
        let approved = ExactApproval {
            binding: binding.clone(),
            issued_at_ms: 90,
            approval: Approval {
                decision_id: "forward-decision".into(),
                call_id: first
                    .pending_approval
                    .as_ref()
                    .unwrap()
                    .call_reference
                    .clone(),
                call_digest: binding.request_digest,
                package_digest: binding.package_digest,
                grant_id: "observe".into(),
                approver: Principal::new("approver", PrincipalKind::OrganizationOwner).unwrap(),
                coapprover: None,
                level: floor,
                expires_at_ms: 200,
                state: ApprovalState::Approved,
            },
        };
        provider.0.store(0, Ordering::SeqCst);
        let mut approved_options = options();
        approved_options.approval = Some(approved);
        first = host
            .run(
                &model,
                &mut sink,
                &token,
                approved_options,
                &mut facts,
                &mut ledger,
                &mut actuator,
            )
            .unwrap();
    }
    assert_eq!(first.outcomes[0].state, OutcomeState::Failed);
    assert_eq!(ledger.intents, 1);
    facts.post = post;
    ledger.invalidate_original_on_prepare = retry == 3;
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
    if floor > ApprovalLevel::A0None && !preauthorized {
        assert!(
            matches!(compensated.governance, GovernedStop::PendingApproval(_)),
            "original approval floor must be retained"
        );
        assert_eq!(actuator.records().len(), 1);
        return;
    }
    if retry == 3 {
        assert_eq!(compensated.governance, GovernedStop::Uncertain);
        assert_eq!(actuator.records().len(), 1);
        return;
    }
    if retry == 2 {
        assert_eq!(compensated.governance, GovernedStop::Uncertain);
        assert_eq!(compensated.outcomes.len(), 1);
        assert_eq!(actuator.records().len(), 2);
        assert_eq!(ledger.intents, 2);
        return;
    }
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
    if retry == 1 {
        provider.0.store(3, Ordering::SeqCst);
        let second = host
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
        assert_ne!(second.governance, GovernedStop::Completed);
        assert_eq!(
            actuator.records().len(),
            2,
            "fresh inverse must not repeat original compensation"
        );
        return;
    }
    // Unknown original outcomes cannot authorize any inverse attempt.
    ledger
        .outcomes_by_id
        .get_mut(&first.executions[0].execution_id())
        .unwrap()
        .state = OutcomeState::Uncertain;
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
fn compensation_retains_original_floor_without_preauthorization() {
    compensation_case(Postconditions::Verified, ApprovalLevel::A4Owner, false, 0);
    compensation_case(Postconditions::Verified, ApprovalLevel::A4Owner, true, 0);
}
#[test]
fn compensation_origin_cannot_be_reused_with_a_fresh_call() {
    compensation_case(Postconditions::Verified, ApprovalLevel::A0None, false, 1);
}

#[test]
fn buffered_inverse_calls_cannot_repeat_the_original() {
    compensation_case(Postconditions::Verified, ApprovalLevel::A0None, false, 2);
}
#[test]
fn original_failed_state_is_revalidated_inside_prepare() {
    compensation_case(Postconditions::Verified, ApprovalLevel::A0None, false, 3);
}
