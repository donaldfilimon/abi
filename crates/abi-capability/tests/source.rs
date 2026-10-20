//! Full closed source compilation and narrowing regressions.
use abi_capability::source::*;
use abi_capability::{ApprovalLevel, Digest, RiskClass};
use serde_json::json;
use std::collections::BTreeSet;
fn source() -> PackageSource {
    serde_json::from_value(json!({
        "id":"local.observe", "version":"1.0.0",
        "input_schema":{"type":"object","additionalProperties":false},
        "output_schema":{"type":"boolean"},"error_schema":{"type":"null"},
        "streaming":"None","cancellation":"BeforeEffect", "required_platform_permissions":[],
        "resource_scope":{"organization_id":"org","deployment_id":"deploy","tenant_id":"tenant","platform":"local","guild_ref":null,"resource_digest":Digest::from_bytes([1;32]),"subject_digest":Digest::from_bytes([2;32])},
        "side_effect":"None","reversibility":"Irreversible","risk_class":"Low",
        "data_classes":["Operational"],"residency":"LocalOnly","retention":"Session",
        "preconditions":["fresh"],"invariants":["scoped"],"deadline_ms":1000,"rate_class":"bounded",
        "budgets":{"max_input_bytes":512,"max_output_bytes":512,"max_events":4,"max_calls":1},
        "confirmation_policy":"A4Owner","delegation_policy":"Forbidden","expiry_policy_ms":1000,"revocation_policy":"RecheckBeforeEffect","approval_policy":"A4Owner",
        "idempotency":"NaturallyIdempotent","compensation":null,"rollback":"None","postconditions":[],
        "adapter_bindings":[{"adapter_id":"recording","api_version":"1.0.0"}],
        "receipt_requirements":{"redacted":true,"require_postcondition_evidence":true},
        "fixtures":{"deterministic":Digest::from_bytes([3;32]),"adversarial":Digest::from_bytes([4;32]),"failure":Digest::from_bytes([5;32])},
        "promotion_thresholds":{"minimum_passed":1,"maximum_failed":0},
        "expiration_criteria":{"expires_at_ms":10000,"disable_on_schema_drift":true,"disable_on_missing_evidence":true}
    })).unwrap()
}
#[test]
fn full_source_is_closed_schema_bound_and_deterministic() {
    let src = source();
    let a = SourceRegistry::compile([src.clone()]).unwrap();
    let b = SourceRegistry::compile([src.clone()]).unwrap();
    let a = a.get("local.observe", "1.0.0").unwrap();
    assert_eq!(
        a.package().digest,
        b.get("local.observe", "1.0.0").unwrap().package().digest
    );
    assert!(a.validates_input(&json!({})));
    assert!(!a.validates_input(&json!({"secret":"canary"})));
    assert!(a.validates_output(&json!(true)));
    assert!(!a.validates_output(&json!("canary")));
    let mut changed = src.clone();
    changed.budgets.max_events += 1;
    assert_ne!(
        a.package().digest,
        SourceRegistry::compile([changed])
            .unwrap()
            .get("local.observe", "1.0.0")
            .unwrap()
            .package()
            .digest
    );
    let mut json = serde_json::to_value(src).unwrap();
    json["unknown"] = json!(true);
    assert!(serde_json::from_value::<PackageSource>(json).is_err());
}
#[test]
fn references_and_compensators_fail_closed() {
    for schema in [
        json!({"$ref":"https://example.com/schema"}),
        json!({"$ref":"#/missing"}),
        json!({"type":"not_a_type"}),
    ] {
        let mut src = source();
        src.input_schema = schema;
        assert!(matches!(
            SourceRegistry::compile([src]),
            Err(SourceError::Schema)
        ));
    }
    let mut src = source();
    src.reversibility = abi_capability::Reversibility::Reversible;
    assert!(SourceRegistry::compile([src.clone()]).is_err());
    src.compensation = Some(CompensationSpec {
        capability_id: "local.inverse".into(),
        version: "1.0.0".into(),
        pre_state: BTreeSet::from(["state".into()]),
        preauthorized: true,
    });
    src.rollback = RollbackSpec::ReverseOrderStopOnFailure;
    assert!(matches!(
        SourceRegistry::compile([src]),
        Err(SourceError::Compensator)
    ));
}
#[test]
fn constitution_narrows_and_rejects_unknown_widening_and_prohibited_entries() {
    let src = source();
    let mut constitution = GuildConstitution {
        version: "v1".into(),
        scope: src.resource_scope.clone(),
        allowed: BTreeSet::from([(src.id.clone(), src.version.clone())]),
        risk_ceiling: RiskClass::Low,
        approval_floor: ApprovalLevel::A5DualControl,
        data_classes: src.data_classes.clone(),
        max_deadline_ms: 500,
        max_approval_ttl_ms: 500,
        budgets: src.budgets.clone(),
    };
    let registry = SourceRegistry::compile([src.clone()]).unwrap();
    let narrowed = registry.narrow(&constitution).unwrap();
    let entry = narrowed.get(&src.id, &src.version).unwrap();
    assert_eq!(entry.source().deadline_ms, 500);
    assert_eq!(entry.package().approval_floor, ApprovalLevel::A5DualControl);
    assert_ne!(
        entry.package().digest,
        registry
            .get(&src.id, &src.version)
            .unwrap()
            .package()
            .digest
    );
    constitution.budgets.max_calls = 2;
    assert!(matches!(
        registry.narrow(&constitution),
        Err(SourceError::Widening)
    ));
    constitution.budgets.max_calls = 1;
    constitution.scope.tenant_id = "other".into();
    assert!(matches!(
        registry.narrow(&constitution),
        Err(SourceError::Widening)
    ));
    constitution.scope = src.resource_scope.clone();
    constitution
        .allowed
        .insert(("missing.tool".into(), "1.0.0".into()));
    assert!(matches!(
        registry.narrow(&constitution),
        Err(SourceError::Widening)
    ));
    let mut prohibited = src;
    prohibited.risk_class = RiskClass::Prohibited;
    constitution
        .allowed
        .remove(&("missing.tool".into(), "1.0.0".into()));
    assert!(matches!(
        SourceRegistry::compile([prohibited])
            .unwrap()
            .narrow(&constitution),
        Err(SourceError::Widening)
    ));
}

#[test]
fn source_prerequisites_reject_wrong_scope_missing_facts_expiry_and_budget() {
    use abi_capability::{Principal, PrincipalKind, Request};
    let source = source();
    let registry = SourceRegistry::compile([source.clone()]).unwrap();
    let compiled = registry.get(&source.id, &source.version).unwrap();
    let mut request = Request::new(
        "attempt",
        "call",
        Principal::new("actor", PrincipalKind::HumanSubject).unwrap(),
        source.resource_scope.clone(),
        &source.id,
        &source.version,
        compiled.package().digest,
        Digest::from_bytes([9; 32]),
        "policy",
        "guild",
        "safety",
        1,
    )
    .unwrap();
    let predicates = BTreeSet::from(["fresh".into(), "scoped".into()]);
    let input = json!({});
    let mut facts = SourceFacts {
        input: &input,
        predicates: &predicates,
        now_ms: 100,
        requested_deadline_ms: 500,
        requested_budgets: &source.budgets,
        adapter: &source.adapter_bindings[0],
        idempotency_key: None,
        captured_pre_state: None,
    };
    assert!(compiled.check_prerequisites(&request, &facts).is_ok());
    request.scope.tenant_id = "other".into();
    assert!(compiled.check_prerequisites(&request, &facts).is_err());
    request.scope = source.resource_scope.clone();
    facts.now_ms = 10_000;
    assert!(compiled.check_prerequisites(&request, &facts).is_err());
    facts.now_ms = 100;
    facts.requested_deadline_ms = 1001;
    assert!(compiled.check_prerequisites(&request, &facts).is_err());
    facts.requested_deadline_ms = 500;
    let empty = BTreeSet::new();
    facts.predicates = &empty;
    assert!(compiled.check_prerequisites(&request, &facts).is_err());
    facts.predicates = &predicates;
    let mut excessive = source.budgets.clone();
    excessive.max_calls = 2;
    facts.requested_budgets = &excessive;
    assert!(compiled.check_prerequisites(&request, &facts).is_err());
    let mut tighter = source.budgets.clone();
    tighter.max_input_bytes = 1;
    facts.requested_budgets = &tighter;
    assert!(
        compiled.check_prerequisites(&request, &facts).is_err(),
        "two-byte input exceeds requested one-byte ceiling"
    );
}

#[test]
fn malformed_unicode_digest_is_a_refusal_not_a_panic() {
    let text = format!("sha256:aé{}", "0".repeat(61));
    assert!(serde_json::from_value::<Digest>(json!(text)).is_err());
    for text in [
        format!("sha256:{}", "A".repeat(64)),
        format!("sha256:{}", "g".repeat(64)),
    ] {
        assert!(serde_json::from_value::<Digest>(json!(text)).is_err());
    }
}

fn constitution_for(s: &PackageSource) -> GuildConstitution {
    GuildConstitution {
        version: "v1".into(),
        scope: s.resource_scope.clone(),
        allowed: BTreeSet::from([(s.id.clone(), s.version.clone())]),
        risk_ceiling: s.risk_class,
        approval_floor: s.approval_policy,
        data_classes: s.data_classes.clone(),
        max_deadline_ms: s.deadline_ms,
        max_approval_ttl_ms: 500,
        budgets: s.budgets.clone(),
    }
}

#[test]
fn mixed_catalogue_intersection_keeps_original_risk_and_data_declarations() {
    let mut low = source();
    low.data_classes = BTreeSet::from([DataClass::Public]);
    let mut medium = source();
    medium.id = "local.medium".into();
    medium.risk_class = RiskClass::Medium;
    let registry = SourceRegistry::compile([low.clone(), medium.clone()]).unwrap();
    let mut constitution = constitution_for(&low);
    constitution
        .allowed
        .insert((medium.id.clone(), medium.version.clone()));
    constitution.risk_ceiling = RiskClass::Medium;
    constitution.data_classes = BTreeSet::from([DataClass::Public, DataClass::Operational]);
    let narrowed = registry
        .narrow(&constitution)
        .expect("catalogue ceiling does not rewrite individual declarations");
    assert_eq!(
        narrowed
            .get(&low.id, &low.version)
            .unwrap()
            .source()
            .risk_class,
        RiskClass::Low
    );
    assert_eq!(
        narrowed
            .get(&low.id, &low.version)
            .unwrap()
            .source()
            .data_classes,
        low.data_classes
    );
    assert!(narrowed.get(&medium.id, &medium.version).is_some());
    constitution.risk_ceiling = RiskClass::Low;
    assert!(
        registry
            .narrow(&constitution)
            .unwrap()
            .get(&medium.id, &medium.version)
            .is_none()
    );
    constitution.risk_ceiling = RiskClass::Medium;
    constitution.data_classes = BTreeSet::from([DataClass::Public]);
    assert!(
        registry
            .narrow(&constitution)
            .unwrap()
            .get(&medium.id, &medium.version)
            .is_none()
    );
}

#[test]
fn source_startup_rejects_platform_write_without_postconditions() {
    let mut s = source();
    s.side_effect = abi_capability::SideEffect::PlatformWrite;
    assert!(SourceRegistry::compile([s]).is_err());
}

#[test]
fn canonical_source_evaluation_enforces_package_and_narrowed_approval_ttl() {
    use abi_agent_runtime::CancellationToken;
    use abi_capability::lifecycle::{AuthorizationBinding, ExactApproval};
    use abi_capability::{
        Approval, ApprovalState, AuthorizationContext, Decision, FixedClock, Grant, Kernel,
        PlatformFacts, Principal, PrincipalKind, Request,
    };
    let mut s = source();
    s.resource_scope.guild_ref = Some("guild".into());
    let registry = SourceRegistry::compile([s.clone()]).unwrap();
    let narrowed = registry.narrow(&constitution_for(&s)).unwrap();
    for sources in [&registry, &narrowed] {
        let compiled = sources.get(&s.id, &s.version).unwrap();
        let registry = sources.capability_registry().unwrap();
        let kernel = Kernel::new(&registry);
        let actor = Principal::new("actor", PrincipalKind::HumanSubject).unwrap();
        let owner = Principal::new("owner", PrincipalKind::GuildOwner).unwrap();
        let request = Request::new(
            "attempt",
            "call",
            actor.clone(),
            s.resource_scope.clone(),
            &s.id,
            &s.version,
            compiled.package().digest,
            Digest::from_bytes([9; 32]),
            "policy",
            "guild",
            "safety",
            1,
        )
        .unwrap();
        let grants = [Grant::active(
            "grant",
            owner.clone(),
            actor,
            s.resource_scope.clone(),
            compiled.package(),
            0,
            10000,
            ApprovalLevel::A4Owner,
            1,
            "policy",
            "guild",
            "safety",
        )
        .unwrap()];
        let clock = FixedClock::new(100);
        let cancellation = CancellationToken::new();
        let used = BTreeSet::new();
        let platform = PlatformFacts::new(BTreeSet::new(), 1, "policy", "guild", "safety");
        let context = AuthorizationContext {
            now_ms: &clock,
            cancellation: &cancellation,
            grants: &grants,
            approval: None,
            used_decision_ids: &used,
            platform: &platform,
        };
        let pending = kernel.evaluate(&request, &context);
        let precondition = Digest::from_bytes([8; 32]);
        let binding = AuthorizationBinding::from_evaluation(
            &request,
            &pending,
            compiled.schema_digest(),
            precondition,
        )
        .unwrap();
        let mut exact = ExactApproval {
            approval: Approval {
                decision_id: "decision".into(),
                call_id: "call".into(),
                call_digest: request.request_digest,
                package_digest: request.package_digest,
                grant_id: "grant".into(),
                approver: owner,
                coapprover: None,
                level: ApprovalLevel::A4Owner,
                expires_at_ms: compiled.source().expiry_policy_ms,
                state: ApprovalState::Approved,
            },
            binding,
            issued_at_ms: 0,
        };
        assert_eq!(
            kernel
                .evaluate_source_bound(&request, &context, compiled, precondition, Some(&exact))
                .decision,
            Decision::Allow
        );
        exact.approval.expires_at_ms += 1;
        assert_eq!(
            kernel
                .evaluate_source_bound(&request, &context, compiled, precondition, Some(&exact))
                .decision,
            Decision::ApprovalRequired,
            "source-specific TTL must reject before durable prepare"
        );
    }
}
