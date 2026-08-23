//! Privacy, failure-path, hierarchy, and deterministic replay qualification.

use abi_agent_runtime::CancellationToken;
use abi_capability::{
    Approval, ApprovalLevel, ApprovalState, AuditError, AuditRecord, AuditSink,
    AuthorizationContext, BoundedMemoryAuditSink, CapabilityPackage, CapabilityRegistry,
    CredentialRef, Decision, Digest, FixedClock, Grant, Kernel, KernelError, PlatformFacts,
    Principal, PrincipalKind, RecordingActuator, Request, Reversibility, RiskClass, Scope,
    SideEffect, TenantCredentialResolver,
};
use std::collections::BTreeSet;

fn digest(byte: u8) -> Digest {
    Digest::from_bytes([byte; 32])
}

fn principal(id: &str, kind: PrincipalKind) -> Principal {
    Principal::new(id, kind).expect("bounded synthetic principal")
}

fn package() -> CapabilityPackage {
    CapabilityPackage::new(
        "discord.channel.organize",
        "2.0.0",
        digest(3),
        SideEffect::PlatformWrite,
        Reversibility::Reversible,
        RiskClass::Medium,
        ApprovalLevel::A1Actor,
        ["manage_channels"],
        ["channel_state_matches"],
    )
    .expect("valid synthetic package")
}

fn scope(tenant: &str) -> Scope {
    Scope::new(
        "org_ref",
        "deploy_ref",
        tenant,
        "discord_recording",
        Some("guild_ref"),
        digest(8),
        digest(9),
    )
    .expect("valid synthetic scope")
}

fn request() -> Request {
    Request::new(
        "attempt_v2",
        "call_v2",
        principal("member_ref", PrincipalKind::HumanSubject),
        scope("tenant_ref"),
        "discord.channel.organize",
        "2.0.0",
        digest(3),
        digest(5),
        "tenant_v2",
        "guild_v2",
        "safety_v2",
        4,
    )
    .expect("valid synthetic request")
}

fn grant(package: &CapabilityPackage) -> Grant {
    Grant::active(
        "grant_v2",
        principal("owner_ref", PrincipalKind::GuildOwner),
        request().principal,
        scope("tenant_ref"),
        package,
        0,
        2_000,
        ApprovalLevel::A1Actor,
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    )
    .expect("valid synthetic grant")
}

fn approval(level: ApprovalLevel, approver: &str) -> Approval {
    Approval {
        decision_id: "decision_v2".to_owned(),
        call_id: "call_v2".to_owned(),
        call_digest: request().request_digest,
        package_digest: digest(3),
        grant_id: "grant_v2".to_owned(),
        approver: principal(approver, PrincipalKind::GuildManager),
        coapprover: None,
        level,
        expires_at_ms: 1_500,
        state: ApprovalState::Approved,
    }
}

fn dual_control_approval(request: &Request) -> Approval {
    Approval {
        decision_id: "decision_a5".to_owned(),
        call_id: request.call_id.clone(),
        call_digest: request.request_digest,
        package_digest: request.package_digest,
        grant_id: "grant_a5".to_owned(),
        approver: principal("admin_one", PrincipalKind::GuildAdministrator),
        coapprover: Some(principal("service_ref", PrincipalKind::Service)),
        level: ApprovalLevel::A5DualControl,
        expires_at_ms: 1_500,
        state: ApprovalState::Approved,
    }
}

#[test]
fn actor_self_confirmation_is_allowed_but_safety_floor_raises_to_manager() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let actor = approval(ApprovalLevel::A1Actor, "member_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    )
    .with_approval_floors(ApprovalLevel::A0None, ApprovalLevel::A2Manager);
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: Some(&actor),
        used_decision_ids: &used,
        platform: &platform,
    };
    let mut audit = BoundedMemoryAuditSink::new(4, 4096);
    let mut actuator = RecordingActuator::new(["channel_state_matches"]);

    let outcome = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect("closed approval response");

    assert_eq!(outcome.receipt.decision, Decision::ApprovalRequired);
    assert_eq!(outcome.receipt.required_approval, ApprovalLevel::A2Manager);
    assert_eq!(actuator.records().len(), 0);
}

#[test]
fn exact_manager_approval_records_one_digest_only_effect() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: Some(&approval),
        used_decision_ids: &used,
        platform: &platform,
    };
    let mut audit = BoundedMemoryAuditSink::new(4, 4096);
    let mut actuator = RecordingActuator::new(["channel_state_matches"]);

    let outcome = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect("authorized recording succeeds");

    assert_eq!(outcome.receipt.decision, Decision::Allow);
    assert_eq!(actuator.records().len(), 1);
    assert!(outcome.receipt.result_digest.is_some());
}

struct FailedAudit;
impl AuditSink for FailedAudit {
    fn record(&mut self, _: &AuditRecord) -> Result<(), AuditError> {
        Err(AuditError::Unavailable)
    }
}

#[test]
fn audit_failure_and_cancellation_each_prevent_recording() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: Some(&approval),
        used_decision_ids: &used,
        platform: &platform,
    };
    let mut actuator = RecordingActuator::new(["channel_state_matches"]);

    let error = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut FailedAudit, &mut actuator)
        .expect_err("audit failure closes the attempt");
    assert_eq!(error, KernelError::Audit(AuditError::Unavailable));
    assert_eq!(actuator.records(), []);

    cancellation.cancel();
    let mut audit = BoundedMemoryAuditSink::new(4, 4096);
    let outcome = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect("cancellation is a closed result");
    assert_eq!(outcome.receipt.decision, Decision::Pause);
    assert_eq!(actuator.records(), []);
}

#[test]
fn frozen_inputs_replay_byte_identically_without_raw_material() {
    fn run() -> Vec<u8> {
        let package = package();
        let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
        let grants = [grant(&package)];
        let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
        let clock = FixedClock::new(1_000);
        let cancellation = CancellationToken::new();
        let used = BTreeSet::new();
        let platform = PlatformFacts::new(
            BTreeSet::from(["manage_channels"]),
            4,
            "tenant_v2",
            "guild_v2",
            "safety_v2",
        );
        let context = AuthorizationContext {
            now_ms: &clock,
            cancellation: &cancellation,
            grants: &grants,
            approval: Some(&approval),
            used_decision_ids: &used,
            platform: &platform,
        };
        let mut audit = BoundedMemoryAuditSink::new(4, 4096);
        let mut actuator = RecordingActuator::new(["channel_state_matches"]);
        let outcome = Kernel::new(&registry)
            .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
            .expect("replay succeeds");
        serde_json::to_vec(&(outcome, audit.records(), actuator.records()))
            .expect("closed evidence encodes")
    }

    let first = run();
    assert_eq!(first, run());
    let serialized = String::from_utf8(first).expect("JSON is UTF-8");
    for forbidden in [
        "synthetic raw message canary",
        "synthetic transcript canary",
        "synthetic audio canary",
        "synthetic secret canary",
        "synthetic output canary",
    ] {
        assert!(!serialized.contains(forbidden), "forbidden material leaked");
    }
}

#[test]
fn credential_reference_is_exact_tenant_and_version_without_fallback() {
    let reference =
        CredentialRef::new("key_ref", "tenant_ref", "provider_ref", 3).expect("bounded reference");
    let resolver = TenantCredentialResolver::new([(reference.clone(), digest(12))]);

    assert_eq!(
        resolver.resolve(&reference, "tenant_ref", 3),
        Some(digest(12))
    );
    assert_eq!(resolver.resolve(&reference, "other_tenant", 3), None);
    assert_eq!(resolver.resolve(&reference, "tenant_ref", 4), None);
}

#[test]
fn exact_scope_policy_epoch_expiry_permission_approval_and_postcondition_mismatches_fail_closed() {
    let cases = [
        ("expired", "grant_expired"),
        ("scope", "scope_mismatch"),
        ("policy", "policy_version_mismatch"),
        ("epoch", "revocation_epoch_mismatch"),
        ("permission", "platform_permission_missing"),
        ("replay", "approval_replayed"),
        ("self_approval", "self_approval"),
        ("postcondition", "postcondition_failed"),
    ];

    for (case, expected) in cases {
        let package = package();
        let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
        let mut grant = grant(&package);
        let request = request();
        let mut approval = approval(ApprovalLevel::A2Manager, "manager_ref");
        let mut permissions = BTreeSet::from(["manage_channels"]);
        let mut platform_epoch = 4;
        let mut platform_policy = "tenant_v2";
        let mut used = BTreeSet::new();
        let mut postconditions = BTreeSet::from(["channel_state_matches"]);
        match case {
            "expired" => grant.expires_at_ms = 1_000,
            "scope" => grant.scope = scope("other_tenant"),
            "policy" => platform_policy = "tenant_v3",
            "epoch" => platform_epoch = 5,
            "permission" => permissions.clear(),
            "replay" => {
                used.insert("decision_v2".to_owned());
            }
            "self_approval" => {
                approval.approver = principal("member_ref", PrincipalKind::GuildManager);
            }
            "postcondition" => postconditions.clear(),
            _ => unreachable!(),
        }
        let grants = [grant];
        let clock = FixedClock::new(1_000);
        let cancellation = CancellationToken::new();
        let platform = PlatformFacts::new(
            permissions,
            platform_epoch,
            platform_policy,
            "guild_v2",
            "safety_v2",
        );
        let context = AuthorizationContext {
            now_ms: &clock,
            cancellation: &cancellation,
            grants: &grants,
            approval: Some(&approval),
            used_decision_ids: &used,
            platform: &platform,
        };
        let mut audit = BoundedMemoryAuditSink::new(4, 4096);
        let mut actuator = RecordingActuator::new(postconditions);

        let outcome = Kernel::new(&registry)
            .authorize_and_record(&request, &context, &mut audit, &mut actuator)
            .expect("mismatch is a closed decision");

        assert_eq!(outcome.receipt.reason.as_str(), expected, "{case}");
        assert!(actuator.records().is_empty(), "{case}");
    }
}

#[test]
fn bounded_audit_capacity_is_fallible_and_precedes_recording() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: Some(&approval),
        used_decision_ids: &used,
        platform: &platform,
    };
    let mut audit = BoundedMemoryAuditSink::new(0, 0);
    let mut actuator = RecordingActuator::new(["channel_state_matches"]);

    let error = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect_err("zero audit capacity fails closed");

    assert_eq!(error, KernelError::Audit(AuditError::Capacity));
    assert_eq!(actuator.records(), []);
}

#[test]
fn grant_issuance_is_owner_admin_scoped_and_prohibited_is_ungrantable() {
    let medium = package();
    let member = principal("member_ref", PrincipalKind::HumanSubject);
    assert!(
        Grant::active(
            "grant_v2",
            principal("service_ref", PrincipalKind::Service),
            member.clone(),
            scope("tenant_ref"),
            &medium,
            0,
            2_000,
            ApprovalLevel::A2Manager,
            4,
            "tenant_v2",
            "guild_v2",
            "safety_v2"
        )
        .is_err()
    );
    assert!(
        Grant::active(
            "grant_v2",
            principal("admin_ref", PrincipalKind::GuildAdministrator),
            member.clone(),
            scope("tenant_ref"),
            &medium,
            0,
            2_000,
            ApprovalLevel::A2Manager,
            4,
            "tenant_v2",
            "guild_v2",
            "safety_v2"
        )
        .is_err()
    );

    let low = CapabilityPackage::new(
        "discord.channel.inspect",
        "2.0.0",
        digest(13),
        SideEffect::PlatformRead,
        Reversibility::Reversible,
        RiskClass::Low,
        ApprovalLevel::A0None,
        ["view_channels"],
        std::iter::empty::<&str>(),
    )
    .expect("low package compiles");
    assert!(
        Grant::active(
            "grant_low",
            principal("admin_ref", PrincipalKind::GuildAdministrator),
            member.clone(),
            scope("tenant_ref"),
            &low,
            0,
            2_000,
            ApprovalLevel::A0None,
            1,
            "tenant_v2",
            "guild_v2",
            "safety_v2"
        )
        .is_ok()
    );

    let prohibited = CapabilityPackage::new(
        "discord.guild.destroy",
        "2.0.0",
        digest(14),
        SideEffect::PlatformWrite,
        Reversibility::Irreversible,
        RiskClass::Prohibited,
        ApprovalLevel::A5DualControl,
        ["administrator"],
        ["guild_absent"],
    )
    .expect("prohibited package remains representable");
    assert!(
        Grant::active(
            "grant_no",
            principal("owner_ref", PrincipalKind::GuildOwner),
            member,
            scope("tenant_ref"),
            &prohibited,
            0,
            2_000,
            ApprovalLevel::A5DualControl,
            1,
            "tenant_v2",
            "guild_v2",
            "safety_v2"
        )
        .is_err()
    );
}

#[test]
fn grant_input_order_cannot_let_an_earlier_mismatch_mask_a_later_exact_grant() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let exact = grant(&package);
    let mut wrong_scope = exact.clone();
    wrong_scope.id = "grant_wrong".to_owned();
    wrong_scope.scope = scope("other_tenant");
    let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );

    for grants in [
        [wrong_scope.clone(), exact.clone()],
        [exact.clone(), wrong_scope.clone()],
    ] {
        let context = AuthorizationContext {
            now_ms: &clock,
            cancellation: &cancellation,
            grants: &grants,
            approval: Some(&approval),
            used_decision_ids: &used,
            platform: &platform,
        };
        let mut audit = BoundedMemoryAuditSink::new(4, 4096);
        let mut actuator = RecordingActuator::new(["channel_state_matches"]);
        let outcome = Kernel::new(&registry)
            .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
            .expect("an exact later grant authorizes deterministically");
        assert_eq!(outcome.receipt.decision, Decision::Allow);
        assert_eq!(actuator.records().len(), 1);
    }
}

#[test]
fn numeric_approval_level_cannot_be_claimed_by_a_weak_principal_kind() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );

    for kind in [PrincipalKind::Service, PrincipalKind::HumanSubject] {
        let mut claimed = approval(ApprovalLevel::A2Manager, "weak_ref");
        claimed.approver = principal("weak_ref", kind);
        let context = AuthorizationContext {
            now_ms: &clock,
            cancellation: &cancellation,
            grants: &grants,
            approval: Some(&claimed),
            used_decision_ids: &used,
            platform: &platform,
        };
        let mut audit = BoundedMemoryAuditSink::new(4, 4096);
        let mut actuator = RecordingActuator::new(["channel_state_matches"]);
        let outcome = Kernel::new(&registry)
            .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
            .expect("weak approver is a closed response");
        assert_eq!(outcome.receipt.reason.as_str(), "approval_insufficient");
        assert_eq!(actuator.records(), []);
    }
}

#[test]
fn caller_mutation_cannot_detach_request_commitment_from_typed_fields() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grants = [grant(&package)];
    let approval = approval(ApprovalLevel::A2Manager, "manager_ref");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let mut tampered = request();
    tampered.scope = scope("other_tenant");
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: Some(&approval),
        used_decision_ids: &used,
        platform: &platform,
    };
    let mut audit = BoundedMemoryAuditSink::new(4, 4096);
    let mut actuator = RecordingActuator::new(["channel_state_matches"]);

    let outcome = Kernel::new(&registry)
        .authorize_and_record(&tampered, &context, &mut audit, &mut actuator)
        .expect("tampered commitment fails closed");

    assert_eq!(outcome.receipt.reason.as_str(), "request_digest_mismatch");
    assert_eq!(actuator.records(), []);
}

#[test]
fn a1_is_actor_confirmation() {
    let actor = principal("member_ref", PrincipalKind::HumanSubject);
    let actor_package = CapabilityPackage::new(
        "discord.channel.inspect",
        "2.0.0",
        digest(20),
        SideEffect::PlatformRead,
        Reversibility::Reversible,
        RiskClass::Low,
        ApprovalLevel::A1Actor,
        ["view_channels"],
        std::iter::empty::<&str>(),
    )
    .expect("A1 package compiles");
    let actor_request = Request::new(
        "attempt_a1",
        "call_a1",
        actor.clone(),
        scope("tenant_ref"),
        "discord.channel.inspect",
        "2.0.0",
        digest(20),
        digest(21),
        "tenant_v2",
        "guild_v2",
        "safety_v2",
        1,
    )
    .expect("A1 request is sealed");
    let actor_grants = [Grant::active(
        "grant_a1",
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        actor.clone(),
        scope("tenant_ref"),
        &actor_package,
        0,
        2_000,
        ApprovalLevel::A1Actor,
        1,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    )
    .expect("administrator may issue low-risk grant")];
    let actor_approval = Approval {
        decision_id: "decision_a1".to_owned(),
        call_id: "call_a1".to_owned(),
        call_digest: actor_request.request_digest,
        package_digest: digest(20),
        grant_id: "grant_a1".to_owned(),
        approver: actor,
        coapprover: None,
        level: ApprovalLevel::A1Actor,
        expires_at_ms: 1_500,
        state: ApprovalState::Approved,
    };
    let actor_registry = CapabilityRegistry::compile([actor_package]).expect("registry compiles");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let actor_platform = PlatformFacts::new(
        BTreeSet::from(["view_channels"]),
        1,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let actor_context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &actor_grants,
        approval: Some(&actor_approval),
        used_decision_ids: &used,
        platform: &actor_platform,
    };
    let mut actor_audit = BoundedMemoryAuditSink::new(4, 4096);
    let mut actor_actuator = RecordingActuator::new(std::iter::empty::<&str>());
    let actor_outcome = Kernel::new(&actor_registry)
        .authorize_and_record(
            &actor_request,
            &actor_context,
            &mut actor_audit,
            &mut actor_actuator,
        )
        .expect("actor confirmation is accepted");
    assert_eq!(actor_outcome.receipt.decision, Decision::Allow);
}

#[test]
fn a5_requires_two_distinct_admin_or_owner_principals() {
    let dual_package = CapabilityPackage::new(
        "discord.guild.restructure",
        "2.0.0",
        digest(22),
        SideEffect::PlatformWrite,
        Reversibility::ReversibleWithLoss,
        RiskClass::High,
        ApprovalLevel::A5DualControl,
        ["administrator"],
        ["guild_structure_matches"],
    )
    .expect("A5 package compiles");
    let dual_request = Request::new(
        "attempt_a5",
        "call_a5",
        principal("member_ref", PrincipalKind::HumanSubject),
        scope("tenant_ref"),
        "discord.guild.restructure",
        "2.0.0",
        digest(22),
        digest(23),
        "tenant_v2",
        "guild_v2",
        "safety_v2",
        1,
    )
    .expect("A5 request is sealed");
    let dual_grants = [Grant::active(
        "grant_a5",
        principal("owner_ref", PrincipalKind::GuildOwner),
        dual_request.principal.clone(),
        scope("tenant_ref"),
        &dual_package,
        0,
        2_000,
        ApprovalLevel::A5DualControl,
        1,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    )
    .expect("owner may issue high-risk grant")];
    let dual_approval = dual_control_approval(&dual_request);
    let dual_registry = CapabilityRegistry::compile([dual_package]).expect("registry compiles");
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let used = BTreeSet::new();
    let dual_platform = PlatformFacts::new(
        BTreeSet::from(["administrator"]),
        1,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let mut dual_audit = BoundedMemoryAuditSink::new(4, 4096);
    let mut dual_actuator = RecordingActuator::new(["guild_structure_matches"]);
    let dual_context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &dual_grants,
        approval: Some(&dual_approval),
        used_decision_ids: &used,
        platform: &dual_platform,
    };
    let weak = Kernel::new(&dual_registry)
        .authorize_and_record(
            &dual_request,
            &dual_context,
            &mut dual_audit,
            &mut dual_actuator,
        )
        .expect("weak coapprover fails closed");
    assert_eq!(weak.receipt.reason.as_str(), "approval_insufficient");
    assert_eq!(dual_actuator.records(), []);

    let mut strong_approval = dual_approval.clone();
    strong_approval.coapprover = Some(principal("admin_two", PrincipalKind::GuildAdministrator));
    let strong_context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &dual_grants,
        approval: Some(&strong_approval),
        used_decision_ids: &used,
        platform: &dual_platform,
    };
    let strong = Kernel::new(&dual_registry)
        .authorize_and_record(
            &dual_request,
            &strong_context,
            &mut dual_audit,
            &mut dual_actuator,
        )
        .expect("two strong distinct approvers satisfy dual control");
    assert_eq!(strong.receipt.decision, Decision::Allow);
    assert_eq!(dual_actuator.records().len(), 1);
}
